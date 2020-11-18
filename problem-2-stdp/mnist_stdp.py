"""
Original paper from : Peter U. Diehl and Matthew Cook (https://doi.org/10.3389/fncom.2015.00099)
Code adapted from : https://github.com/zxzhijia/Brian2STDPMNIST
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import prefs, units, NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network, StateMonitor
from sklearn import datasets

from simulation_parameters import SimulationParameters
from stopwatch import Stopwatch
from util_plots import img_show

LOGGER = logging.getLogger('mnist_stdp')
DATA_DIR = './data'


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
    img_show(images[:9].reshape(9, 28, 28), 'Examples d\'images du jeu de données MNIST', labels[:9])

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


def chose_labeled_neurons(spike_activities):
    """
    Chose the best neuron to recognise each label.

    :param spike_activities:
    :return: A list a index of excitatory neuron that spike the most for each label.
    """
    return np.argmax(spike_activities, axis=1)


def infer_label(spike_activity, labeled_neurons):
    """
    Infer the label of an image based on the spiking activity of the excitatory neurons.
    Using the labeled neurons.

    :param spike_activity: The spiking activity of each neurons of the excitatory group for an input.
    :param labeled_neurons: The list of labeled neurons computed.
    :return: The best guess of label.
    """
    spikes_interest = np.array([spike_activity[x] for x in labeled_neurons])
    return np.argmax(spikes_interest)


def train(net, images, labels, parameters):
    nb_train_samples = len(images)
    LOGGER.info(f'Start training on {len(images)} images...')
    Stopwatch.starting('training')

    current_input_intensity = parameters.input_intensity

    previous_spike_count_e = np.zeros(len(net['neurons_i']))
    previous_spike_count_i = np.zeros(len(net['neurons_i']))

    # Array to store spike activity for each excitatory neurons
    spike_per_label = np.zeros((10, len(net['neurons_e'])))
    count_activation_map = np.zeros([1, len(net['neurons_e'])])

    average_spike_evolution_e = []
    average_spike_evolution_i = []
    evolution_moyenne_matrice_poids = []

    net['neurons_input'].rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second, namespace=parameters.get_namespace())  # Why?

    # TODO add epoch
    for i, (image, label) in enumerate(zip(images, labels)):
        LOGGER.debug(f'Start training step {i + 1:03}/{nb_train_samples} ({i / nb_train_samples * 100:5.2f}%)')
        enough_spikes = False

        while not enough_spikes:
            normalize_weights(net['synapses_input_e'])
            evolution_moyenne_matrice_poids.append(np.average(net['synapses_input_e'].w))

            input_rates = image.reshape(len(net['neurons_input'])) / 8 * current_input_intensity
            net['neurons_input'].rates = input_rates * units.Hz

            # Run the network
            net.run(parameters.exposition_time, namespace=parameters.get_namespace())

            current_spike_count_e = net['spike_counters_e'].count - previous_spike_count_e
            current_spike_count_i = net['spike_counters_i'].count - previous_spike_count_i

            average_spike_evolution_e.append(np.average(current_spike_count_e))
            average_spike_evolution_i.append(np.average(current_spike_count_i))

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
                    f'Not enough spikes ({current_sum_spikes_e}), retry with higher intensity level ({current_input_intensity})')
            else:
                # Store spike activity
                spike_per_label[int(label)] += current_spike_count_e

                # Reset network
                net['neurons_input'].rates = 0 * units.Hz
                net.run(parameters.resting_time, namespace=parameters.get_namespace())
                current_input_intensity = parameters.input_intensity

                enough_spikes = True

            # if enough_spikes is True and np.size(count_activation_map, axis=0) < 5:
            #     count_activation_map = np.concatenate((count_activation_map, current_spike_count_e))

    time_msg = Stopwatch.stopping('training', len(images))
    LOGGER.info(f'Training completed. {time_msg}.')

    return spike_per_label, average_spike_evolution_e, average_spike_evolution_i


def run(parameters: SimulationParameters):
    if parameters.nb_train_samples > 60000:
        raise ValueError('The number of train sample can\'t be more than 60000')
    if parameters.nb_test_samples > 10000:
        raise ValueError('The number of test sample can\'t be more than 10000')

    LOGGER.info('Beginning of execution')

    # Brian code generation target (optimization?)
    prefs.codegen.target = 'cython'

    # Fix seed
    np.random.seed(0)

    # Load MNIST data
    images, labels = load_data()

    # ================================== Simulation parameters =================================
    nb_input_neurons = len(images[0])
    COURBES = 10

    neuron_eqs_e = '''
        dv/dt  = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) : volt (unless refractory)
        I_synE = ge * nS * -v                                       : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        dtheta/dt = -theta / (tc_theta)                             : volt
        dtimer/dt = 0.1                                             : second
        '''

    neuron_eqs_i = '''
        dv/dt  = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) : volt (unless refractory)
        I_synE = ge * nS * -v                                      : amp
        I_synI = gi * nS * (-85.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                      : 1
        dgi/dt = -gi/(2.0*ms)                                      : 1
        '''

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

    # ==================================== Network creation ====================================

    LOGGER.info('Creating network...')
    Stopwatch.starting('network_creation')

    # Initialise the network, then add every part to it.
    net = Network()

    # ========================= Neurons excitator ==========================
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
    neurons_input = PoissonGroup(nb_input_neurons, 0 * units.Hz, name='neurons_input')
    net.add(neurons_input)

    # ========================= Input => Excitator =========================
    synapses_input_e = Synapses(neurons_input, neurons_e, model=eqs_stdp, on_pre=eqs_stdp_pre, on_post=eqs_stdp_post,
                                name='synapses_input_e')
    synapses_input_e.connect(True)  # All to all
    synapses_input_e.w = 'rand() * 0.3'
    min_delay = 0 * units.ms
    max_delay = 10 * units.ms
    delta_delay = max_delay - min_delay
    synapses_input_e.delay = 'min_delay + rand() * delta_delay'
    net.add(synapses_input_e)

    # ============================== Monitors ==============================

    # Spike counter
    spike_counters_e = SpikeMonitor(neurons_e, record=False, name='spike_counters_e')
    spike_counters_i = SpikeMonitor(neurons_i, record=False, name='spike_counters_i')
    net.add(spike_counters_e, spike_counters_i)

    # Voltage monitors
    volt_mon_e = StateMonitor(neurons_e, 'v', record=[0, 1], name='volt_mon_e')
    volt_mon_i = StateMonitor(neurons_i, 'v', record=[0, 1], name='volt_mon_i')
    net.add(volt_mon_e, volt_mon_i)

    # Weight monitors
    mon_input = StateMonitor(synapses_input_e, 'w', record=[0, 1], name='mon_input')
    mon_e_i = StateMonitor(synapses_e_i, 'w', record=[0, 1], name='mon_e_i')
    mon_i_e = StateMonitor(synapses_i_e, 'w', record=[0, 1], name='mon_i_e')

    # Feed the network with parameter variables
    net._clocks = []  # Fix a bug from brian 2.4.2
    net.before_run(parameters.get_namespace())

    time_msg = Stopwatch.stopping('network_creation')
    LOGGER.info(f'Network created. {time_msg}.')

    # ======================================== Training ========================================

    train_images = images[:parameters.nb_train_samples]
    train_labels = labels[:parameters.nb_train_samples]
    spike_per_label, average_spike_evolution_e, average_spike_evolution_i = train(net,
                                                                                  train_images,
                                                                                  train_labels,
                                                                                  parameters)

    labeled_neurons = chose_labeled_neurons(spike_per_label)

    # ========================================= Plots ==========================================

    Stopwatch.starting('plotting')
    LOGGER.info(f'Start plotting...')

    plt.figure()
    plt.plot(average_spike_evolution_e, label="exitateur")
    plt.plot(average_spike_evolution_i, label="inhibiteur")
    plt.title("Evolution de la moyenne des décharge exitateur")
    plt.legend()
    # plt.subplot(212)
    # plt.plot(evolution_moyenne_matrice_poids)
    # plt.title("Evolution de la moyenne des poids")
    plt.show()

    plt.figure()
    plt.plot(volt_mon_e.t / units.second, volt_mon_e.v[0], label='excitator')
    plt.plot(volt_mon_i.t / units.second, volt_mon_i.v[0], label='inhibitor')
    plt.title('Potentiel firtst neuron exc et inhib')
    plt.legend()
    plt.show()

    # Activation map graph
    # plt.figure()
    # for i in range(len(count_activation_map)):
    #     plt.plot(range(len(current_spike_count_e)), count_activation_map[i])
    # # plot chaque ligne de la matrice spike pour la carte d'activation
    # plt.title('Carte d'' activation')
    # plt.show()

    # Courbes d'accord
    # plt.figure()
    # plt.plot(cou)
    # # plot les colonnes de spikes pour avoir les courbes d'accord. Pas toutes les faire car beaucoup trop
    # plt.title(f"Échantillon des courbes d''accord des {COURBES} premiers neurones")
    # plt.show()

    # Histogramme des courbes d'accord
    fig = plt.figure()
    plt.hist(synapses_input_e.w, bins=10, edgecolor='black')
    bins = range(1, 11)
    plt.title(f"répartition des poids selon leur valeur après {parameters.nb_train_samples} itérations")
    plt.show()

    # Average of weights with the number of samples
    # plt.figure()
    # plt.title(f"Moyenne des poids après {nb_train_samples} itération")
    # plt.plot(evolution_moyenne_matrice_poids)
    # plt.show()

    plt.subplot(211)
    plt.title('Synapses input to exc')
    plt.plot(synapses_input_e.w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(mon_input.t / units.second, mon_input.w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.title('Synapses exc to inh')
    plt.plot(synapses_e_i.w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(mon_e_i.t / units.second, mon_e_i.w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.title('Synapses inh to exc')
    plt.plot(synapses_i_e.w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(mon_i_e.t / units.second, mon_i_e.w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    time_msg = Stopwatch.stopping('plotting')
    LOGGER.info(f'Plotting completed. {time_msg}.')

    # ======================================== Testing =========================================

    test_images = images[60000:60000 + parameters.nb_test_samples]
    test_labels = labels[60000:60000 + parameters.nb_test_samples]

    nb_test_samples = len(test_images)
    Stopwatch.starting('testing')
    LOGGER.info(f'Start testing on {nb_test_samples} images...')

    current_input_intensity = parameters.input_intensity

    # TODO disable learning
    net.store()  # If remove the also re enable previous_spike_count_e

    previous_spike_count_e = np.copy(net['spike_counters_e'].count)
    nb_correct = 0

    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second, namespace=parameters.get_namespace())  # Why?

    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        LOGGER.debug(f'Start testing step {i + 1:03}/{nb_test_samples} ({i / nb_test_samples * 100:5.2f}%)')
        enough_spikes = False

        # Restore the network everytime because we don't disable the learning
        net.restore()

        while not enough_spikes:
            input_rates = image.reshape((nb_input_neurons)) / 8 * current_input_intensity
            neurons_input.rates = input_rates * units.Hz

            # Run the network
            net.run(parameters.exposition_time, namespace=parameters.get_namespace())

            current_spike_count_e = spike_counters_e.count - previous_spike_count_e
            # Since the network is reset everytime, the previous_spike_count_e can stay the same
            # previous_spike_count_e = np.copy(spike_counters_e.count)

            current_sum_spikes_e = np.sum(current_spike_count_e)

            # Check if enough spike triggered, if under the limit start again with the same image
            if current_sum_spikes_e < 5:
                current_input_intensity += 1
                neurons_input.rates = 0 * units.Hz
                net.run(parameters.resting_time, namespace=parameters.get_namespace())
                LOGGER.debug(
                    f'Not enough spikes ({current_sum_spikes_e}), retry with higher intensity level ({current_input_intensity})')
            else:
                # Reset network
                neurons_input.rates = 0 * units.Hz
                net.run(parameters.resting_time, namespace=parameters.get_namespace())
                current_input_intensity = parameters.input_intensity

                inferred_label = infer_label(current_spike_count_e, labeled_neurons)

                if inferred_label == int(label):
                    nb_correct += 1
                    LOGGER.debug(f'Correctly classified {label} - Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')
                else:
                    LOGGER.debug(f'Badly classified {label} (inferred {inferred_label}) '
                                 f'- Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')

                enough_spikes = True

    time_msg = Stopwatch.stopping('testing', nb_test_samples)
    LOGGER.info(f'Testing completed. {time_msg}.')

    LOGGER.info(f'Final accuracy on {nb_test_samples} images: {nb_correct / nb_test_samples * 100:.5}%')
