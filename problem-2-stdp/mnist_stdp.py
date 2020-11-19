"""
authors: Victor Yon, Soline Bernard et Antoine Marion
date: 19/11/2020
version history: See Github
description: Création du réseau de neurones avec Biran2, boucles d'entrainement et de test. Chargement des données,
normalisation et choix des neurones de classification.
D'après les formules de Peter U. Diehl et Matthew Cook (https://doi.org/10.3389/fncom.2015.00099)
Code inspiré de : https://github.com/zxzhijia/Brian2STDPMNIST
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from brian2 import prefs, units, NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network, StateMonitor
from sklearn import datasets

from mnist_stdp_out import init_out_directory, result_out
from mnist_stdp_plots import plot_post_training, COURBES, img_show, plot_post_testing
from simulation_parameters import SimulationParameters
from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_stdp')
DATA_DIR = './data'
INFO_FREQUENCY = 50


def load_data():
    Stopwatch.starting('load_data')

    save_path = Path(DATA_DIR, 'mnist_784.p')

    if save_path.is_file():
        LOGGER.info(f'Loading MNIST database from local file ({save_path})...')
        images, labels = pickle.load(open(save_path, 'rb'))
    else:
        LOGGER.info('Downloading MNIST database from distant server...')
        # Don't use cache because it's slow
        images, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, cache=False)

        # Store in file using pickle
        LOGGER.debug(f'Saving MNIST database in local file ({save_path})...')
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        pickle.dump((images, labels), open(save_path, 'wb'))

    time_msg = Stopwatch.stopping('load_data')
    LOGGER.info(f'MNIST database loaded: {len(images)} images of dimension {images[0].shape}. {time_msg}.')

    # Show the first 9 images
    img_show(images[:9].reshape(9, 28, 28), 'Exemples d\'images du jeu de données MNIST', labels[:9])

    return images, labels


def normalize_weights(synapses_input_e):
    len_source = len(synapses_input_e.source)
    len_target = len(synapses_input_e.target)
    connection = np.zeros((len_source, len_target))
    connection[synapses_input_e.i, synapses_input_e.j] = synapses_input_e.w
    temp_conn = np.copy(connection)
    col_sums = np.sum(temp_conn, axis=0)
    col_factors = 78. / col_sums  # Because why not?!
    for j in range(len_target):
        temp_conn[:, j] *= col_factors[j]
    synapses_input_e.w = temp_conn[synapses_input_e.i, synapses_input_e.j]


def chose_labeled_neurons(spike_activities, parameters):
    """
    Chose the best neuron to recognise each label.

    :return: A list a index of excitatory neuron that spike the most for each label.
    """

    if parameters.classification_type == 'single':
        return np.argmax(spike_activities, axis=1)

    elif parameters.classification_type == 'group':
        # Liste pour chaque classe des indices des neurones qui ont déchargé le plus pour elle pendnat l'entrainement
        labeled_neuron_group = [[], [], [], [], [], [], [], [], [], []]

        for i in range(len(spike_activities[0])):
            labeled_neuron_group[np.argmax(spike_activities[:, i])].append(i)

        return labeled_neuron_group

    else:
        raise ValueError(f'Unknown classification type {parameters.classification_type}.')


def infer_label(spike_activity, labeled_neurons, parameters):
    """
    Infer the label of an image based on the spiking activity of the excitatory neurons.
    Using the labeled neurons.

    :param spike_activity: The spiking activity of each neurons of the excitatory group for an input.
    :param labeled_neurons: The list of labeled neurons computed.
    :return: The best guess of label.
    """

    if parameters.classification_type == 'single':
        spikes_interest = np.array([spike_activity[x] for x in labeled_neurons])
        return np.argmax(spikes_interest)

    elif parameters.classification_type == 'group':
        # On récupére la moyenne des spikes des neurones correspondant aux classes
        average = np.zeros(10)
        for j in range(10):
            neurone_interet = np.array([int(spike_activity[x]) for x in labeled_neurons[j]])
            if len(neurone_interet) != 0:
                average[j] = np.mean(neurone_interet)

        # On trouve le label (égal à l'indice de la plus grosse moyenne)
        return np.argmax(average)

    else:
        raise ValueError(f'Unknown classification type {parameters.classification_type}.')


def build_network(input_size, parameters):
    # Initialise the network, then add every part to it.
    net = Network()

    # ========================= Neurons excitator ==========================

    neuron_eqs_e = '''
        dv/dt  = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) : volt (unless refractory)
        I_synE = ge * nS * -v                                       : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        dtheta/dt = -theta / (tc_theta)                             : volt
        dtimer/dt = 0.1                                             : second
        '''

    neurons_e = NeuronGroup(parameters.nb_excitator_neurons,
                            neuron_eqs_e,
                            threshold='(v > (theta - offset + v_thresh_e)) and (timer > refrac_e)',
                            refractory=parameters.refrac_e,
                            reset='v = v_reset_e; theta += theta_plus; timer = 0*ms',
                            method='euler',
                            name='neurons_e')
    neurons_e.v = parameters.v_rest_e - 40. * units.mV
    neurons_e.theta = np.ones(parameters.nb_excitator_neurons) * 20.0 * units.mV
    net.add(neurons_e)

    # ========================= Neurons inhibitor ==========================

    neuron_eqs_i = '''
        dv/dt  = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) : volt (unless refractory)
        I_synE = ge * nS * -v                                      : amp
        I_synI = gi * nS * (-85.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                      : 1
        dgi/dt = -gi/(2.0*ms)                                      : 1
        '''

    neurons_i = NeuronGroup(parameters.nb_inhibitor_neurons,
                            neuron_eqs_i,
                            threshold='v > v_thresh_i',
                            refractory=parameters.refrac_i,
                            reset='v = v_reset_i',
                            method='euler',
                            name='neurons_i')
    neurons_i.v = parameters.v_rest_i - 40. * units.mV
    net.add(neurons_i)

    # ======================= Excitator => Inhibitor =======================

    synapses_e_i = Synapses(neurons_e, neurons_i, model='w : 1', on_pre='ge_post += w', name='synapses_e_i')
    synapses_e_i.connect(condition='i==j')  # One to one (diagonal only)
    synapses_e_i.w = '10.4'
    net.add(synapses_e_i)

    # ======================= Inhibitor => Excitator =======================

    synapses_i_e = Synapses(neurons_i, neurons_e, model='w : 1', on_pre='gi_post += w', name='synapses_i_e')
    synapses_i_e.connect(condition='i!=j')  # All except one (not diagonal only)
    synapses_i_e.w = '17.0'
    net.add(synapses_i_e)

    # =========================== Neurons input ============================

    neurons_input = PoissonGroup(input_size, 0 * units.Hz, name='neurons_input')
    net.add(neurons_input)

    # ========================= Input => Excitator =========================

    eqs_stdp = '''
        w                               : 1
        post2before                     : 1
        dpre/dt    = -pre/(tc_pre)      : 1 (event-driven)
        dpost1/dt  = -post1/(tc_post_1) : 1 (event-driven)
        dpost2/dt  = -post2/(tc_post_2) : 1 (event-driven)
    '''

    eqs_stdp_pre = '''
            ge_post += w
            pre = 1
            w = clip(w - nu_pre * post1, 0, wmax)
        '''

    eqs_stdp_post = '''
            post2before = post2
            w = clip(w + nu_post * pre * post2before, 0, wmax)
            post1 = 1
            post2 = 1
        '''

    synapses_input_e = Synapses(neurons_input, neurons_e, model=eqs_stdp, on_pre=eqs_stdp_pre, on_post=eqs_stdp_post,
                                name='synapses_input_e')
    synapses_input_e.connect(True)  # All to all
    synapses_input_e.w = 'rand() * 0.3'
    min_delay = 0 * units.ms
    max_delay = 10 * units.ms
    delta_delay = max_delay - min_delay
    synapses_input_e.delay = 'min_delay + rand() * delta_delay'
    net.add(synapses_input_e)

    return net


def setup_network_monitors(net):
    # Spike counter
    spike_counters_input = SpikeMonitor(net['neurons_input'], record=False, name='spike_counters_input')
    spike_counters_e = SpikeMonitor(net['neurons_e'], record=False, name='spike_counters_e')
    spike_counters_i = SpikeMonitor(net['neurons_i'], record=False, name='spike_counters_i')
    net.add(spike_counters_input, spike_counters_e, spike_counters_i)

    # Voltage monitors
    volt_mon_e = StateMonitor(net['neurons_e'], 'v', record=[0, 1], name='volt_mon_e')
    volt_mon_i = StateMonitor(net['neurons_i'], 'v', record=[0, 1], name='volt_mon_i')
    net.add(volt_mon_e, volt_mon_i)

    # Weight monitors
    mon_input = StateMonitor(net['synapses_input_e'], 'w', record=[0, 1], name='mon_input')
    mon_e_i = StateMonitor(net['synapses_e_i'], 'w', record=[0, 1], name='mon_e_i')
    mon_i_e = StateMonitor(net['synapses_i_e'], 'w', record=[0, 1], name='mon_i_e')
    net.add(mon_input, mon_e_i, mon_i_e)


def train(net, images, labels, parameters):
    nb_train_samples = len(images)
    LOGGER.info(f'Start training on {len(images)} images...')
    Stopwatch.starting('training')

    current_input_intensity = parameters.input_intensity

    # Array to store spike activity for each excitatory neurons
    spike_per_label_input = np.zeros((10, len(net['neurons_input'])))
    spike_per_label_e = np.zeros((10, len(net['neurons_e'])))
    count_activation_map = np.zeros([1, len(net['neurons_e'])])

    # Array to store previous spike activity
    previous_spike_count_input = np.zeros(len(net['neurons_input']))
    previous_spike_count_e = np.zeros(len(net['neurons_e']))
    previous_spike_count_i = np.zeros(len(net['neurons_i']))

    average_spike_evolution_e = []
    average_spike_evolution_i = []
    average_weights_evolution = []

    net['neurons_input'].rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second, namespace=parameters.get_namespace())  # Why?

    for epoch in range(parameters.nb_epoch):
        LOGGER.info(f'Start epoch {epoch + 1}/{parameters.nb_epoch}')

        for i, (image, label) in enumerate(zip(images, labels)):
            LOGGER.log(logging.INFO if i % INFO_FREQUENCY == 0 else logging.DEBUG,
                       f'Start training step {i + 1:03}/{nb_train_samples} ({i / nb_train_samples * 100:5.2f}%)')
            enough_spikes = False

            while not enough_spikes:
                # Input => Excitator weight normalization
                if parameters.normalization:
                    normalize_weights(net['synapses_input_e'])

                average_weights_evolution.append(np.average(net['synapses_input_e'].w))

                input_rates = image.reshape(len(net['neurons_input'])) / 8 * current_input_intensity
                net['neurons_input'].rates = input_rates * units.Hz

                # Run the network
                net.run(parameters.exposition_time, namespace=parameters.get_namespace())

                current_spike_count_input = net['spike_counters_input'].count - previous_spike_count_input
                current_spike_count_e = net['spike_counters_e'].count - previous_spike_count_e
                current_spike_count_i = net['spike_counters_i'].count - previous_spike_count_i

                average_spike_evolution_e.append(np.average(current_spike_count_e))
                average_spike_evolution_i.append(np.average(current_spike_count_i))

                previous_spike_count_input = np.copy(net['spike_counters_input'].count)
                previous_spike_count_e = np.copy(net['spike_counters_e'].count)
                previous_spike_count_i = np.copy(net['spike_counters_i'].count)

                current_sum_spikes_e = np.sum(current_spike_count_e)
                current_sum_spikes_i = np.sum(current_spike_count_i)

                LOGGER.debug(
                    f'Number of excitatory spikes: {current_sum_spikes_e} | inhibitor spikes: {current_sum_spikes_i}')

                # Check if enough spike triggered, if under the limit start again with the same image
                if current_sum_spikes_e < 5:
                    current_input_intensity += 1
                    net['neurons_input'].rates = 0 * units.Hz
                    net.run(parameters.resting_time, namespace=parameters.get_namespace())
                    LOGGER.debug(
                        f'Not enough spikes ({current_sum_spikes_e}), '
                        f'retry with higher intensity level ({current_input_intensity})')
                else:
                    # Store spike activity
                    spike_per_label_e[int(label)] += current_spike_count_e
                    spike_per_label_input[int(label)] += current_spike_count_input

                    # Reset network
                    net['neurons_input'].rates = 0 * units.Hz
                    net.run(parameters.resting_time, namespace=parameters.get_namespace())
                    current_input_intensity = parameters.input_intensity

                    enough_spikes = True

                if enough_spikes is True and np.size(count_activation_map, axis=0) < COURBES:
                    count_activation_map = np.concatenate((count_activation_map, current_spike_count_e.reshape(1, -1)))

    time_msg = Stopwatch.stopping('training', len(images))
    LOGGER.info(f'Training completed. {time_msg}.')

    return spike_per_label_e, spike_per_label_input, average_spike_evolution_e, average_spike_evolution_i, \
           count_activation_map, average_weights_evolution


def test(net, images, labels, labeled_neurons, parameters):
    nb_test_samples = len(images)
    LOGGER.info(f'Start testing on {nb_test_samples} images...')
    Stopwatch.starting('testing')

    current_input_intensity = parameters.input_intensity

    # TODO disable learning
    net.store()  # If remove the also re enable previous_spike_count_e

    previous_spike_count_e = np.copy(net['spike_counters_e'].count)
    nb_correct = 0
    y_pred = np.zeros(parameters.nb_test_samples)

    net['neurons_input'].rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second, namespace=parameters.get_namespace())  # Why?

    for i, (image, label) in enumerate(zip(images, labels)):
        LOGGER.log(logging.INFO if i % INFO_FREQUENCY == 0 else logging.DEBUG,
                   f'Start testing step {i + 1:03}/{nb_test_samples} ({i / nb_test_samples * 100:5.2f}%)')
        enough_spikes = False

        # Restore the network everytime because we don't disable the learning
        net.restore()

        while not enough_spikes:
            input_rates = image.reshape((len(net['neurons_input']))) / 8 * current_input_intensity
            net['neurons_input'].rates = input_rates * units.Hz

            # Run the network
            net.run(parameters.exposition_time, namespace=parameters.get_namespace())

            current_spike_count_e = net['spike_counters_e'].count - previous_spike_count_e
            # Since the network is reset everytime, the previous_spike_count_e can stay the same
            # previous_spike_count_e = np.copy(net['spike_counters_e'].count)

            current_sum_spikes_e = np.sum(current_spike_count_e)

            # Check if enough spike triggered, if under the limit start again with the same image
            if current_sum_spikes_e < 5:
                current_input_intensity += 1
                net['neurons_input'].rates = 0 * units.Hz
                net.run(parameters.resting_time, namespace=parameters.get_namespace())
                LOGGER.debug(
                    f'Not enough spikes ({current_sum_spikes_e}), '
                    f'retry with higher intensity level ({current_input_intensity})')
            else:
                # Reset network
                net['neurons_input'].rates = 0 * units.Hz
                net.run(parameters.resting_time, namespace=parameters.get_namespace())
                current_input_intensity = parameters.input_intensity

                inferred_label = infer_label(current_spike_count_e, labeled_neurons, parameters)
                y_pred[i] = inferred_label

                if inferred_label == int(label):
                    nb_correct += 1
                    LOGGER.debug(f'Correctly classified {label} - Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')
                else:
                    LOGGER.debug(f'Badly classified {label} (inferred {inferred_label}) '
                                 f'- Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')

                enough_spikes = True

    time_msg = Stopwatch.stopping('testing', nb_test_samples)
    LOGGER.info(f'Testing completed. {time_msg}.')
    y_true = np.zeros(parameters.nb_test_samples)
    for i in range(len(y_true)):
        y_true[i] = labels[i]

    y_pred = np.array(y_pred)
    # TODO move plot_post_testing in run()
    plot_post_testing(y_pred, y_true, parameters)
    LOGGER.info(f'Final accuracy on {nb_test_samples} images: {nb_correct / nb_test_samples * 100:.5}%')

    return nb_correct / nb_test_samples


def run(parameters: SimulationParameters):
    init_out_directory(parameters)

    LOGGER.info(f'Beginning of run "{parameters.run_name}"')
    timer = Stopwatch('run')
    timer.start()

    # Brian code generation target
    prefs.codegen.target = 'cython'

    # Fix seed
    np.random.seed(0)

    # Load MNIST dataset
    images, labels = load_data()

    # ===================================== Create Network =====================================

    LOGGER.info('Creating network...')
    Stopwatch.starting('network_creation')

    # Build the network
    input_size = len(images[0])
    net = build_network(input_size, parameters)

    # Attach the monitors
    setup_network_monitors(net)

    # Prepare the network
    net._clocks = []  # Fix a bug from brian 2.4.2
    net.before_run(parameters.get_namespace())

    time_msg = Stopwatch.stopping('network_creation')
    LOGGER.info(f'Network created. {time_msg}.')

    # ======================================== Training ========================================

    train_images = images[:parameters.nb_train_samples]
    train_labels = labels[:parameters.nb_train_samples]

    # Start the training loop
    train_stats = train(net, train_images, train_labels, parameters)

    # Find the labeled neurons
    labeled_neurons = chose_labeled_neurons(train_stats[0], parameters)

    # Plot at the end of the training
    plot_post_training(net, train_stats, parameters)

    # ======================================== Testing =========================================

    test_images = images[60000:60000 + parameters.nb_test_samples]
    test_labels = labels[60000:60000 + parameters.nb_test_samples]

    # Start the training loop
    accuracy = test(net, test_images, test_labels, labeled_neurons, parameters)

    timer.stop()
    time_msg = timer.log()
    LOGGER.info(f'End of run "{parameters.run_name}". {time_msg}.')

    result_out(parameters, accuracy, time_msg)
