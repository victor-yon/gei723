import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn import datasets
from sparse import COO

from network import SpikeFunction
from parameters import Parameters
from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_grad')
DATA_DIR = './data'

NB_CLASSES = 10
IMAGE_SIZE = 28 * 28


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
        membrane_potential_at_t = (1 - t / int(p.tau_v)) * membrane_potential_at_t
        # Using tau_i with euler or exact method
        membrane_current_at_t = (1 - t / int(p.tau_i)) * membrane_current_at_t

        # Select the input current at time t
        input_at_t = input_current[:, :, t]

        # Integrate the input current
        membrane_current_at_t += input_at_t

        # Integrate the input to the membrane potential
        membrane_potential_at_t += membrane_current_at_t / int(p.tau_v)

        # Apply the non-differentiable function
        recorded_spikes_at_t = SpikeFunction.apply(membrane_potential_at_t - p.v_threshold)
        recorded_spikes.append(recorded_spikes_at_t)

        # Reset the spiked neurons
        membrane_potential_at_t[membrane_potential_at_t > p.v_threshold] = 0

    recorded_spikes = torch.stack(recorded_spikes, dim=2)  # Stack over time axis (Array -> Tensor)
    return recorded_spikes


def run(p: Parameters):
    # Reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    LOGGER.info(f'Beginning of run "{p.run_name}"')
    timer = Stopwatch('run')
    timer.start()

    # Use the GPU unless there is none available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    LOGGER.debug(f'pyTorch device selected: {device}')

    # Load the datasets
    images_spike_train, labels = load_data(p)

    # Initialise model's parameters
    w1 = torch.empty((IMAGE_SIZE, p.nb_hidden_neurons), device=device, dtype=torch.float, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0., std=.1)

    w2 = torch.empty((p.nb_hidden_neurons, NB_CLASSES), device=device, dtype=torch.float, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0., std=.1)

    params = [w1, w2]  # Trainable parameters
    optimizer = torch.optim.Adam(params, lr=p.learning_rate, amsgrad=True)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # ============================ Training ============================

    LOGGER.info(f'Start training on {p.nb_train_samples} images...')
    Stopwatch.starting('training')

    train_indices = np.arange(p.nb_train_samples)
    # Shuffle the train indices
    np.random.shuffle(train_indices)

    nb_train_batches = len(train_indices) // p.batch_size

    for epoch in range(p.nb_epoch):
        LOGGER.info(f'Start epoch {epoch + 1:02}/{p.nb_epoch} ({epoch / p.nb_epoch * 100:4.1f}%)')
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

            # Forward propagation
            layer_1_spikes = run_spiking_layer(batch_spike_train, w1, device, p)
            layer_2_spikes = run_spiking_layer(layer_1_spikes, w2, device, p)
            network_output = torch.sum(layer_2_spikes, 2)  # Count the spikes over time axis

            if LOGGER.isEnabledFor(logging.DEBUG):
                # Show result for the first image of the batch
                inferred_label = torch.argmax(network_output[0])
                correct_label = int(batch_labels[0])
                net_output_str = " | ".join(map(lambda x: f'{x[0]}:{int(x[1])}', enumerate(network_output[0])))
                LOGGER.debug(f'Example - spikes per label: {net_output_str}')
                LOGGER.debug(f'Example - inferred ({inferred_label}) for label ({correct_label}) '
                             f'{"[GOOD]" if inferred_label == correct_label else "[BAD]"}')

            loss = loss_fn(network_output, target_output)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            LOGGER.debug(f'Batch {i + 1:03}/{nb_train_batches} completed with loss : {loss:.4f}')

        LOGGER.info(f'Epoch loss: {epoch_loss / nb_train_batches:.4f}')

    time_msg = Stopwatch.stopping('training', p.nb_train_samples * p.nb_epoch)
    LOGGER.info(f'Training completed. {time_msg}.')

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
    # We only need to batchify the test set for memory requirements
    for batch_indices in np.array_split(test_indices, nb_test_batches):
        test_spike_train = torch.FloatTensor(images_spike_train[batch_indices].todense()).to(device)

        # Same forward propagation as before
        layer_1_spikes = run_spiking_layer(test_spike_train, w1, device, p)
        layer_2_spikes = run_spiking_layer(layer_1_spikes, w2, device, p)
        network_output = torch.sum(layer_2_spikes, 2)  # Count the spikes over time axis

        # Do the prediction by selecting the output neuron with the most number of spikes
        _, am = torch.max(network_output, 1)
        inferred_labels = am.detach().cpu().numpy()
        correct_label_count += np.sum(inferred_labels == labels[test_indices])
        print(correct_label_count)

        if LOGGER.isEnabledFor(logging.DEBUG):
            # Show result for the first image of the batch
            inferred_label = inferred_labels[0]
            correct_label = int(labels[test_indices[0]])
            net_output_str = " | ".join(map(lambda x: f'{x[0]}:{int(x[1])}', enumerate(network_output[0])))
            LOGGER.debug(f'Example - spikes per label: {net_output_str}')
            LOGGER.debug(f'Example - inferred ({inferred_label}) for label ({correct_label}) '
                         f'{"[GOOD]" if inferred_label == correct_label else "[BAD]"}')

    time_msg = Stopwatch.stopping('testing', nb_test)
    LOGGER.info(f'{"Validation" if p.use_validation else "Testing"} completed. {time_msg}.')

    LOGGER.info(f'Final accuracy on {nb_test} images: {correct_label_count / nb_test * 100:.5}%')

    timer.stop()
    time_msg = timer.log()
    LOGGER.info(f'End of run "{p.run_name}". {time_msg}.')