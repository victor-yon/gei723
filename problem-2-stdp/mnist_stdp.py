"""
Original paper from : Peter U. Diehl and Matthew Cook (https://doi.org/10.3389/fncom.2015.00099)
Code adapted from : https://github.com/zxzhijia/Brian2STDPMNIST
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import prefs, units, NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network, StateMonitor
from sklearn import datasets

from stopwatch import Stopwatch
from util_plots import img_show

LOGGER = logging.getLogger(__name__)
DATA_DIR = './data'


def load_data():
    Stopwatch.starting('load_data')

    save_path = Path(DATA_DIR, 'mnist_784.p')

    if save_path.is_file():
        LOGGER.info(f'Loading MNIST database from local file ({save_path}) ...')
        images, labels = pickle.load(open(save_path, 'rb'))
    else:
        LOGGER.info('Downloading MNIST database from distant server ...')
        # Don't use cache because it's slow
        images, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, cache=False)

        # Store in file using pickle
        LOGGER.debug(f'Saving MNIST database in local file ({save_path}) ...')
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


def run(nb_train_samples: int = 60000, nb_test_samples: int = 10000):
    if nb_train_samples > 60000:
        raise ValueError('The number of train sample can\'t be more than 60000')
    if nb_test_samples > 10000:
        raise ValueError('The number of test sample can\'t be more than 10000')

    # Brian code generation target (optimization?)
    prefs.codegen.target = 'cython'

    # Fix seed
    np.random.seed(0)

    # Load MNIST data
    images, labels = load_data()

    # ================================== Simulation parameters =================================
    nb_input_neurons = len(images[0])
    nb_excitator_neurons: int = 400  # a faire varier
    nb_inhibitor_neurons: int = 400  # a faire varier
    single_example_time = 0.35 * units.second
    resting_time = 0.15 * units.second
    input_intensity = 2
    start_input_intensity = input_intensity
    COURBES = 10 # Nombre de courbes d'activation que l'on souhaite
    WEIGHTS_TO_RECORD = [4000, 60000]

    offset = 20.0 * units.mV
    v_rest_e = -65. * units.mV
    v_rest_i = -60. * units.mV
    v_reset_e = -65. * units.mV
    v_reset_i = 'v=-45.*mV'
    v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'
    v_thresh_i = 'v>-40.*mV'
    refrac_e = 5. * units.ms
    refrac_i = 2. * units.ms

    tc_pre_ee = 20 * units.ms
    tc_post_1_ee = 20 * units.ms
    tc_post_2_ee = 40 * units.ms
    nu_ee_pre = 0.0001  # learning rate à faire varier
    nu_ee_post = 0.01  # learning rate à faire varier
    wmax_ee = 1.0  # à faire varier

    # Training only
    tc_theta = 1e7 * units.ms
    theta_plus_e = 0.05 * units.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS * -v                                       : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        '''

    # Training only
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
    #
    neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

    neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS * -v                                      : amp
        I_synI = gi * nS * (-85.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                      : 1
        dgi/dt = -gi/(2.0*ms)                                      : 1
        '''

    eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
    eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
    eqs_stdp_post_ee = \
        'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    # ==================================== Network creation ====================================

    LOGGER.info('Creating excitator and inhibitor neurons...')
    Stopwatch.starting('network_creation')

    neurons_e = NeuronGroup(nb_excitator_neurons, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e,
                            method='euler')
    neurons_i = NeuronGroup(nb_inhibitor_neurons, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                            reset=v_reset_i, method='euler')

    spike_counters_e = SpikeMonitor(neurons_e, record=False)
    spike_counters_i = SpikeMonitor(neurons_i, record=False)

    neurons_e.v = v_rest_e - 40. * units.mV
    neurons_i.v = v_rest_i - 40. * units.mV

    # Training only
    neurons_e.theta = np.ones(nb_excitator_neurons) * 20.0 * units.mV

    synapses_e_i = Synapses(neurons_e, neurons_i, model='w : 1', on_pre='ge_post += w')
    synapses_e_i.connect(condition='i==j')  # One to one (diagonal only)
    synapses_e_i.w = '10.4'  # Not random?

    synapses_i_e = Synapses(neurons_i, neurons_e, model='w : 1', on_pre='gi_post += w')
    synapses_i_e.connect(condition='i!=j')  # All except one (not diagonal only)
    synapses_i_e.w = '17.0'

    LOGGER.info('Creating input neurons...')

    neurons_input = PoissonGroup(nb_input_neurons, 0 * units.Hz)

    model = 'w : 1'
    on_pre = 'ge_post += w'
    on_post = eqs_stdp_post_ee

    # Training only
    model += eqs_stdp_ee
    on_pre += '; ' + eqs_stdp_pre_ee

    synapses_input_e = Synapses(neurons_input, neurons_e, model=model, on_pre=on_pre, on_post=on_post)
    synapses_input_e.connect(True)  # All to all
    synapses_input_e.w = 'rand() * 0.3'

    volt_mon_e = StateMonitor(neurons_e, 'v', record=range(COURBES))
    volt_mon_i = StateMonitor(neurons_i, 'v', record=range(COURBES))

    mon_input = StateMonitor(synapses_input_e, 'w', record=WEIGHTS_TO_RECORD)

    min_delay = 0 * units.ms
    max_delay = 10 * units.ms
    delta_delay = max_delay - min_delay

    # Construct the network
    net = Network(neurons_input, neurons_i, neurons_e, synapses_i_e, synapses_e_i, synapses_input_e, spike_counters_e, spike_counters_i, volt_mon_e, volt_mon_i, mon_input)

    time_msg = Stopwatch.stopping('network_creation')
    LOGGER.info(f'Network created. {time_msg}.')

    # ======================================== Training ========================================

    LOGGER.info(f'Start training on {nb_train_samples} images...')
    Stopwatch.starting('training')

    previous_spike_count_e = np.zeros(nb_excitator_neurons)
    previous_spike_count_i = np.zeros(nb_inhibitor_neurons)

    evolution_moyenne_spike_e = []
    evolution_moyenne_spike_i = []
    evolution_moyenne_matrice_poids = []

    # Array to store spike activity for each excitatory neurons
    spike_per_label = np.zeros((10, nb_excitator_neurons))

    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second)  # Why?
    count_activation_map = np.zeros([1,nb_excitator_neurons])

    # TODO add epoch
    for i, (image, label) in enumerate(zip(images[:nb_train_samples], labels[:nb_train_samples])):
        LOGGER.debug(f'Start training step {i + 1:03}/{nb_train_samples} ({i / nb_train_samples * 100:5.2f}%)')
        enough_spikes = False

        while not enough_spikes:
            normalize_weights(synapses_input_e)
            evolution_moyenne_matrice_poids.append(np.average(synapses_input_e.w))

            input_rates = image.reshape((nb_input_neurons)) / 8 * input_intensity
            neurons_input.rates = input_rates * units.Hz

            # Run the network
            net.run(single_example_time)

            current_spike_count_e = spike_counters_e.count - previous_spike_count_e
            current_spike_count_i = spike_counters_i.count - previous_spike_count_i

            evolution_moyenne_spike_e.append(np.average(current_spike_count_e))
            evolution_moyenne_spike_i.append(np.average(current_spike_count_i))

            previous_spike_count_e = np.copy(spike_counters_e.count)
            previous_spike_count_i = np.copy(spike_counters_i.count)
            sum_spikes = np.sum(current_spike_count_e)

            # Check if enough spike triggered, if under the limit start again with the same image
            if sum_spikes < 5:
                input_intensity += 1
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                LOGGER.debug(
                    f'Not enough spikes ({sum_spikes}), retry with higher intensity level ({input_intensity})')
            else:
                # Store spike activity
                spike_per_label[int(label)] += current_spike_count_e

                # Reset network
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                input_intensity = start_input_intensity

                enough_spikes = True

            if enough_spikes is True and np.size(count_activation_map, axis=0) < COURBES:
                count_activation_map = np.concatenate((count_activation_map, current_spike_count_e.reshape(1, -1)))

    labeled_neurons = chose_labeled_neurons(spike_per_label)
    ax_average_weights = mon_input.t[0:-1:5000]
    time_msg = Stopwatch.stopping('training', nb_train_samples)
    LOGGER.info(f'Training completed. {time_msg}.')

    # ========================================= Plots ==========================================

    Stopwatch.starting('plotting')
    LOGGER.info(f'Start plotting...')

    plt.figure()
    plt.plot(evolution_moyenne_spike_e, linestyle='--',linewidth=3, label="excitateur")
    plt.plot(evolution_moyenne_spike_i, label="inhibiteur")
    plt.title("Moyenne des décharge des neurones excitateurs avec les échantillons")
    plt.xlabel('Nombre d\'échantillons')
    plt.ylabel('Moyenne des décharges')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for i in range(2):
        plt.plot(volt_mon_e.t[:10000] / units.second, volt_mon_e.v[i,:10000], label=f'excitator {i}')
    #     plt.plot(volt_mon_i.t / units.second, volt_mon_i.v[i], label=f'inhibitor {i}')
    plt.title(f'Potentiel {COURBES} premiers neurones excitateurs')
    plt.xlabel('Temps (s)')
    plt.ylabel('voltage des neurones en V')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Activation map graph
    plt.figure()
    for i in range(1, np.size(count_activation_map, axis=0)):
        plt.plot(range(len(current_spike_count_e)), count_activation_map[i,:], label=f'exemple {i}')
    # plot chaque ligne de la matrice spike pour la carte d'activation
    plt.xlabel('indice du neurone de la couche excitatrice')
    plt.ylabel('Nombre de décharge par neurones')
    plt.title('Carte d\'activation de 9 exemples d\'entraînement')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.ylim([0, 10])
    plt.tight_layout()
    plt.show()

    # Courbes d'accord
    plt.figure()
    for i in range(COURBES):
        plt.plot(range(10), spike_per_label[:, i*30], label=f'neurone {i+1}')
    plt.title(f"Échantillon des courbes d\'accord de {COURBES} neurones")
    plt.xlabel('valeur de l\'étiquette')
    plt.ylabel('nombre de déchange')
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=[1.25, 1])
    plt.show()

    # Histogramme des courbes d'accord
    plt.figure()
    plt.hist(synapses_input_e.w, bins=20, edgecolor='black', label='')
    plt.title(f"répartition des poids selon leur valeur après {nb_train_samples} itérations")
    plt.xlabel('Valeur des poids')
    plt.ylabel('Quantité dans chaque catégorie')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.subplot(211)
    plt.title('Synapses échantillonées entrée vers excitateur')
    plt.plot(synapses_input_e.w[0:-1:200], '.k', label='poids des synapses')
    plt.ylabel('Poids')
    plt.xlabel('Indices des synapses')
    plt.subplot(212)
    plt.title('évolution des poids dans le temps')
    for i in range(len(WEIGHTS_TO_RECORD)):
        plt.plot(mon_input.t / units.second, mon_input.w[i], label=f'synapse {WEIGHTS_TO_RECORD[i]}')
    plt.plot(ax_average_weights, evolution_moyenne_matrice_poids, label=f'moyenne des poids')
    plt.xlabel('Temps (s)')
    plt.ylabel('Poids')
    plt.ylim([0, 0.20])
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=[0.85, 0.90], prop={'size':6})
    plt.show()


    time_msg = Stopwatch.stopping('plotting')
    LOGGER.info(f'Plotting completed. {time_msg}.')

    # ======================================== Testing =========================================

    Stopwatch.starting('testing')
    LOGGER.info(f'Start testing on {nb_test_samples} images...')

    # TODO disable learning
    net.store()  # If remove the also re enable previous_spike_count_e

    nb_correct = 0

    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second)  # Why?
    y_pred = np.zeros([1,nb_test_samples])
    for i, (image, label) in enumerate(
            zip(images[60000:60000 + nb_test_samples], labels[60000:60000 + nb_test_samples])):
        LOGGER.debug(f'Start testing step {i + 1:03}/{nb_test_samples} ({i / nb_test_samples * 100:5.2f}%)')
        enough_spikes = False

        # Restore the network everytime because we don't disable the learning
        net.restore()

        while not enough_spikes:
            input_rates = image.reshape((nb_input_neurons)) / 8 * input_intensity
            neurons_input.rates = input_rates * units.Hz

            # Run the network
            net.run(single_example_time)

            current_spike_count_e = spike_counters_e.count - previous_spike_count_e
            # Since the network is reset everytime, the previous_spike_count_e can stay the same
            # previous_spike_count_e = np.copy(spike_counters_e.count)

            sum_spikes = np.sum(current_spike_count_e)

            # Check if enough spike triggered, if under the limit start again with the same image
            if sum_spikes < 5:
                input_intensity += 1
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                LOGGER.debug(
                    f'Not enough spikes ({sum_spikes}), retry with higher intensity level ({input_intensity})')
            else:
                # Reset network
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                input_intensity = start_input_intensity

                inferred_label = infer_label(current_spike_count_e, labeled_neurons)
                print(inferred_label)
                # y_pred[i] = inferred_label
                if inferred_label == int(label):
                    nb_correct += 1
                    LOGGER.debug(f'Correctly classified {label} - Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')
                else:
                    LOGGER.debug(f'Badly classified {label} (inferred {inferred_label}) '
                                 f'- Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')

                enough_spikes = True
    # Matrice de confusion
    # y_true = labels[60000:nb_test_samples]
    # print('y_true')
    # print(y_true)
    # print('y_pred')
    # print(y_pred)

    time_msg = Stopwatch.stopping('testing', nb_test_samples)
    LOGGER.info(f'Testing completed. {time_msg}.')

    LOGGER.info(f'Final accuracy on {nb_test_samples} images: {nb_correct / nb_test_samples * 100:.5}%')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.info('Beginning of execution')

    run(nb_train_samples=10, nb_test_samples=20)
