import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn import datasets
from sparse import COO

from parameters import Parameters
from plots import plot_activation_map, plot_gradient_surrogates, plot_weight_hist, plot_relu_alpha
from plots import plot_losses
from results_output import init_out_directory, result_out
from spike_functions import SpikeFunctionRelu, SpikeFunctionFastSigmoid, SpikeFunctionPiecewise, SpikeFunctionSigmoid, SpikeFunctionPiecewiseSymetrique
from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_grad')
DATA_DIR = './data'

NB_CLASSES = 10
IMAGE_SIZE = 28 * 28
STEP = 2


def load_data(p: Parameters):
    Stopwatch.starting('load_data')

    save_path = Path(DATA_DIR, 'mnist_grad.p')

    if save_path.is_file():
        LOGGER.info(f'Loading MNIST database with post processing from local file ({save_path})...')
        images, labels = pickle.load(open(save_path, 'rb'))
    else:
        LOGGER.info('Downloading MNIST database from distant server...')

        # Let's download the MNIST dataset, available at https://www.openml.org/d/554
        # Don't use cache because it's slow
        images, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, cache=False)

        # Convert the labels (string) to integers for convenience
        labels = np.array(labels, dtype=np.int)

        # We'll normalize our input data in the range [0., 1[.
        images = images / pow(2, 8)  # / 256

        # Store in file using pickle
        LOGGER.debug(f'Saving MNIST database with post processing in local file ({save_path})...')
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        pickle.dump((images, labels), open(save_path, 'wb'))

    # And convert the data to a spike train
    time_of_spike = (1 - images) * p.absolute_duration  # The brighter the white, the earlier the spike
    # "Remove" the spikes associated with darker pixels (Presumably less information)
    time_of_spike[images < .25] = 0

    sample_id, neuron_idx = np.nonzero(time_of_spike)

    # We use a sparse COO array to store the spikes for memory requirements
    # You can use the spike_train variable as if it were a tensor
    # of shape (nb_of_samples, nb_of_features, absolute_duration)
    images_spike_train = COO((sample_id, neuron_idx, time_of_spike[sample_id, neuron_idx]),
                             np.ones_like(sample_id), shape=(len(images), IMAGE_SIZE, p.absolute_duration))

    time_msg = Stopwatch.stopping('load_data')
    LOGGER.info(f'MNIST database loaded: {len(images)} images of dimension {images[0].shape}. {time_msg}.')

    return images_spike_train, labels


def init_net_params(run_parameters: Parameters, device):
    LOGGER.info(f'Initialize network parameters on {len(run_parameters.size_hidden_layers)} hidden layers')
    LOGGER.info('Network shape : ' + str(IMAGE_SIZE) + ' -> ' +
                ' -> '.join(map(str, run_parameters.size_hidden_layers))
                + ' -> ' + str(NB_CLASSES))

    # Model's parameters
    params = []
    nb_params = 0

    # Input -> first hidden
    weights_input_hidden = torch.empty((IMAGE_SIZE, run_parameters.size_hidden_layers[0]),
                                       device=device, dtype=torch.float, requires_grad=True)
    if run_parameters.extreme_learning:
        weights_input_hidden.requires_grad_(False)

    torch.nn.init.normal_(weights_input_hidden, mean=0., std=.1)
    params.append(weights_input_hidden)
    nb_params += IMAGE_SIZE * run_parameters.size_hidden_layers[0]

    # All others hidden -> hidden
    for layer_index in range(len(run_parameters.size_hidden_layers) - 1):
        weights_hidden_hidden = torch.empty(
            (run_parameters.size_hidden_layers[layer_index], run_parameters.size_hidden_layers[layer_index + 1]),
            device=device, dtype=torch.float, requires_grad=True)
        if run_parameters.extreme_learning and layer_index < run_parameters.size_hidden_layers:
            weights_hidden_hidden.requires_grad_(False)

        torch.nn.init.normal_(weights_hidden_hidden, mean=0., std=.1)
        params.append(weights_hidden_hidden)
        nb_params += run_parameters.size_hidden_layers[layer_index] * run_parameters.size_hidden_layers[layer_index + 1]

    # Last layer -> output
    weights_hidden_output = torch.empty((run_parameters.size_hidden_layers[-1], NB_CLASSES),
                                        device=device, dtype=torch.float, requires_grad=True)
    torch.nn.init.normal_(weights_hidden_output, mean=0., std=.1)
    params.append(weights_hidden_output)
    nb_params += run_parameters.size_hidden_layers[-1] * NB_CLASSES

    LOGGER.info(f'Network parameters initialized ({nb_params})')

    return params


def run_spiking_layer(input_spike_train, layer_weights, device, p: Parameters):
    """Here we implement a current-LIF dynamic in pytorch"""

    # First, we multiply the input spike train by the weights of the current layer to get the current that will be added
    # We can calculate this beforehand because the weights are constant in the forward pass (no plasticity)
    # Equivalent to a matrix multiplication for tensors of dim > 2 using Einstein's Notation
    input_current = torch.einsum("abc,bd->adc", (input_spike_train, layer_weights))

    recorded_spikes = []  # Array of the output spikes at each time t
    membrane_potential_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), device=device,
                                          dtype=torch.float)
    membrane_current_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), device=device,
                                        dtype=torch.float)

    for t in range(p.absolute_duration):  # For every timestep
        # Apply the leak
        # Using tau_v with euler or exact method
        membrane_potential_at_t = (1 - int(p.delta_t) / int(p.tau_v)) * membrane_potential_at_t
        # Using tau_i with euler or exact method
        membrane_current_at_t = (1 - int(p.delta_t) / int(p.tau_i)) * membrane_current_at_t

        # Select the input current at time t
        input_at_t = input_current[:, :, t]

        # Integrate the input current
        membrane_current_at_t += input_at_t

        # Integrate the input to the membrane potential
        membrane_potential_at_t += membrane_current_at_t

        # Select the surrogate function based on the parameters
        spike_functions = None
        if p.surrogate_gradient == 'relu':
            spike_functions = SpikeFunctionRelu
        elif p.surrogate_gradient == 'fast_sigmoid':
            spike_functions = SpikeFunctionFastSigmoid
        elif p.surrogate_gradient == 'piecewise':
            spike_functions = SpikeFunctionPiecewise
        elif  p.surrogate_gradient == 'sigmoid':
            spike_functions = SpikeFunctionSigmoid
        elif  p.surrogate_gradient == 'piecewiseSymetrique':
            spike_functions = SpikeFunctionPiecewiseSymetrique

        # Apply the non-differentiable function
        recorded_spikes_at_t = spike_functions.apply(membrane_potential_at_t - p.v_threshold)

        recorded_spikes.append(recorded_spikes_at_t)

        # Reset the spiked neurons
        membrane_potential_at_t[membrane_potential_at_t > p.v_threshold] = 0

    recorded_spikes = torch.stack(recorded_spikes, dim=2)  # Stack over time axis (Array -> Tensor)
    return recorded_spikes


def run(p: Parameters):
    init_out_directory(p)

    LOGGER.info(f'Beginning of run "{p.run_name}"')
    timer = Stopwatch('run')
    timer.start()

    # Reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Use the GPU unless there is none available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    LOGGER.debug(f'pyTorch device selected: {device}')

    # Load the datasets
    images_spike_train, labels = load_data(p)

    # Initialize the network parameters
    params = init_net_params(p, device)

    # Setup the optimizer and the loss function
    optimizer = torch.optim.Adam(params, lr=p.learning_rate, amsgrad=True)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # ============================ Training ============================

    LOGGER.info(f'Start training on {p.nb_train_samples} images...')
    Stopwatch.starting('training')

    train_indices = np.arange(p.nb_train_samples)
    # Shuffle the train indices
    np.random.shuffle(train_indices)

    nb_train_batches = len(train_indices) // p.batch_size
    losses_evolution = []
    # Activation map of the all except first layer
    layer_to_measure = 0  # Index of the layer to register for activation map
    activation_map_data = []

    for epoch in range(p.nb_epoch):
        LOGGER.info(f'Start epoch {epoch + 1}/{p.nb_epoch} ({epoch / p.nb_epoch * 100:4.1f}%)')
        epoch_loss = 0
        for i, batch_indices in enumerate(np.array_split(train_indices, nb_train_batches)):
            # Select batch and convert to tensors
            batch_spike_train = torch.FloatTensor(images_spike_train[batch_indices].todense()).to(device)
            batch_labels = torch.LongTensor(labels[batch_indices, np.newaxis]).to(device)

            # Here we create a target spike count (10 spikes for wrong label, 100 spikes for true label)
            # in a one-hot fashion
            # This approach is seen in Shrestha & Orchard (2018) https://arxiv.org/pdf/1810.08646.pdf
            # Code available at https://github.com/bamsumit/slayerPytorch
            min_spike_count = 10 * torch.ones((len(batch_labels), 10), device=device, dtype=torch.float)
            target_output = min_spike_count.scatter_(1, batch_labels, 100.0)

            # Forward propagation through each layers
            next_layer_input = batch_spike_train
            a = 0  # Counter for the for loop
            for layer_params in params:
                next_layer_input = run_spiking_layer(next_layer_input, layer_params, device, p)
                # We measure the spikes of layer a for each i sample
                if layer_to_measure == a and i % STEP == 0 and len(activation_map_data) <= 8:
                    activation_map_data.append(torch.sum(next_layer_input, 2)[layer_to_measure])
                a += 1

            # Count the spikes over time axis from the last layer output
            network_output = torch.sum(next_layer_input, 2)

            if LOGGER.isEnabledFor(logging.DEBUG):
                # Show result for the first image of the batch
                inferred_label = torch.argmax(network_output[0])
                correct_label = int(batch_labels[0])
                net_output_str = " | ".join(map(lambda x: f'{x[0]}:{int(x[1]):02}', enumerate(network_output[0])))
                LOGGER.debug(f'Example - spikes per label: {net_output_str}')
                LOGGER.debug(f'Example - inferred ({inferred_label}) for label ({correct_label}) '
                             f'{"[GOOD]" if inferred_label == correct_label else "[BAD]"}')

            loss = loss_fn(network_output, target_output)
            losses_evolution.append(float(loss))

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            LOGGER.debug(f'Batch {i + 1:03}/{nb_train_batches} completed with loss : {loss:.4f}')

        LOGGER.info(f'Epoch loss: {epoch_loss / nb_train_batches:.4f}')

    time_msg = Stopwatch.stopping('training', p.nb_train_samples * p.nb_epoch)
    LOGGER.info(f'Training completed. {time_msg}.')

    # ============================ Plotting ============================

    LOGGER.info(f'Post training plotting.')

    plot_losses(losses_evolution, p)

    LOGGER.info(f'Post training plotting completed and saved.')

    # ============================= Testing ============================

    # Select the indices depending of test or validation run parameter
    if p.use_validation:
        start_indices = p.nb_train_samples + p.nb_test_samples
        end_indices = start_indices + p.nb_validation_samples
    else:
        start_indices = p.nb_train_samples
        end_indices = start_indices + p.nb_test_samples

    test_indices = np.arange(start_indices, end_indices)
    nb_test = len(test_indices)

    LOGGER.info(f'Start {"validation" if p.use_validation else "test"} on {nb_test} images...')
    Stopwatch.starting('testing')

    nb_test_batches = nb_test // p.batch_size
    correct_label_count = 0
    y_pred = []
    # We only need to batchify the test set for memory requirements
    for batch_indices in np.array_split(test_indices, nb_test_batches):
        batch_spike_test = torch.FloatTensor(images_spike_train[batch_indices].todense()).to(device)

        # Forward propagation through each layers
        next_layer_input = batch_spike_test
        for layer_params in params:
            next_layer_input = run_spiking_layer(next_layer_input, layer_params, device, p)

        # Count the spikes over time axis from the last layer output
        network_output = torch.sum(next_layer_input, 2)

        # Do the prediction by selecting the output neuron with the most number of spikes
        _, am = torch.max(network_output, 1)
        inferred_labels = am.detach().cpu().numpy()
        correct_label_count += np.sum(inferred_labels == labels[batch_indices])
        y_pred.append(inferred_labels)

        if LOGGER.isEnabledFor(logging.DEBUG):
            # Show result for the first image of the batch
            inferred_label = inferred_labels[0]
            correct_label = int(labels[batch_indices[0]])
            net_output_str = " | ".join(map(lambda x: f'{x[0]}:{int(x[1]):02}', enumerate(network_output[0])))
            LOGGER.debug(f'Example - spikes per label: {net_output_str}')
            LOGGER.debug(f'Example - inferred ({inferred_label}) for label ({correct_label}) '
                         f'{"[GOOD]" if inferred_label == correct_label else "[BAD]"}')

    time_msg = Stopwatch.stopping('testing', nb_test)
    LOGGER.info(f'{"Validation" if p.use_validation else "Testing"} completed. {time_msg}.')

    accuracy = correct_label_count / nb_test
    LOGGER.info(f'Final accuracy on {nb_test} images: {accuracy * 100:.5}%')

    # ============================ Plotting ============================

    LOGGER.info(f'Post {"validation" if p.use_validation else "testing"} plotting.')

    y_true = labels[test_indices]
    y_pred = np.array(y_pred).reshape(1,-1)[0,:]
    # plot_post_test(y_pred, y_true, p)
    plot_gradient_surrogates(p)
    plot_weight_hist(params, p)
    plot_activation_map(activation_map_data, p)
    plot_relu_alpha(p)

    LOGGER.info(f'Post {"validation" if p.use_validation else "testing"} plotting completed and saved.')

    timer.stop()
    time_msg = timer.log()
    LOGGER.info(f'End of run "{p.run_name}". {time_msg}.')

    result_out(p, accuracy, time_msg)
