"""
Original paper from : Peter U. Diehl and Matthew Cook (https://doi.org/10.3389/fncom.2015.00099)
Code adapted from : https://github.com/zxzhijia/Brian2STDPMNIST
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from brian2 import prefs, units, NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network, StateMonitor
from sklearn import datasets

from util_plots import img_show

LOGGER = logging.getLogger(__name__)


def load_data():
    LOGGER.info('Loading MNIST database...')
    images, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data')
    LOGGER.info(f'MNIST database loaded: {len(images)} images of dimension {images[0].shape}')

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
    COURBES = 10

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

    volt_mon_e = StateMonitor(neurons_e, 'v', record=[0, 1])
    volt_mon_i = StateMonitor(neurons_i, 'v', record=[0, 1])

    mon_input = StateMonitor(synapses_input_e, 'w', record=[0, 1])
    mon_e_i = StateMonitor(synapses_e_i, 'w', record=[0, 1])
    mon_i_e = StateMonitor(synapses_i_e, 'w', record=[0, 1])

    min_delay = 0 * units.ms
    max_delay = 10 * units.ms
    delta_delay = max_delay - min_delay

    # Construct the network
    net = Network(neurons_input, neurons_i, neurons_e, synapses_i_e, synapses_e_i, synapses_input_e, spike_counters_e)

    # ======================================== Training ========================================
    LOGGER.info(f'Start training on {nb_train_samples} images')

    previous_spike_count_e = np.zeros(nb_excitator_neurons)
    previous_spike_count_i = np.zeros(nb_inhibitor_neurons)

    evolution_moyenne_spike_e = []
    evolution_moyenne_spike_i = []
    evolution_moyenne_matrice_poids = []

    # Array to store spike activity for each excitatory neurons
    spike_per_label = np.zeros((10, nb_excitator_neurons))

    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second)  # Why?
    count_activation_map = []

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

            if enough_spikes == True & len(count_activation_map) < 5 * nb_excitator_neurons:
                count_activation_map.append(current_spike_count_e)

    labeled_neurons = chose_labeled_neurons(spike_per_label)

    # ========================================= Plots ==========================================

    plt.subplot(211)
    plt.plot(evolution_moyenne_spike_e, label="exitateur")
    plt.plot(evolution_moyenne_spike_i, label="inhibiteur")
    plt.title("Evolution de la moyenne des décharge exitateur")
    plt.legend()
    plt.subplot(212)
    plt.plot(evolution_moyenne_matrice_poids)
    plt.title("Evolution de la moyenne des poids")
    plt.show()

    LOGGER.info(f'Training completed')

    plt.figure()
    plt.plot(volt_mon_e.t / units.second, volt_mon_e.v[0])
    plt.plot(volt_mon_i.t / units.second, volt_mon_i.v[0])
    plt.title('Potentiel neuron exc et inhib')
    plt.show()

    # Activation map graph
    plt.figure()
    plt.plot(range(len(current_spike_count_e)), count_activation_map[0])
    # plot chaque ligne de la matrice spike pour la carte d'activation
    plt.title('Carte d'' activation')
    plt.show()

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
    plt.title(f"répartition des poids selon leur valeur après {nb_train_samples} itérations")
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

    # ======================================== Testing =========================================

    LOGGER.info(f'Start testing on {nb_test_samples} images')

    # TODO disable learning
    net.store()  # If remove the also re enable previous_spike_count_e

    nb_correct = 0

    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second)  # Why?

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

                if inferred_label == int(label):
                    nb_correct += 1
                    LOGGER.debug(f'Correctly classified {label} - Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')
                else:
                    LOGGER.debug(f'Badly classified {label} (inferred {inferred_label}) '
                                 f'- Current accuracy: {nb_correct / (i + 1) * 100:5.2f}%')

                enough_spikes = True

    LOGGER.info(f'Testing completed')
    LOGGER.info(f'Final accuracy on {nb_test_samples} images: {nb_correct / nb_test_samples * 100:.5}%')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.info('Beginning of execution')

    run(nb_train_samples=50, nb_test_samples=50)
