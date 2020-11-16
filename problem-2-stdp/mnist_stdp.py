"""
Original paper from : Peter U. Diehl and Matthew Cook (https://doi.org/10.3389/fncom.2015.00099)
Code adapted from : https://github.com/zxzhijia/Brian2STDPMNIST
"""

import logging

import numpy as np
from brian2 import prefs, units, NeuronGroup, Synapses, SpikeMonitor, PoissonGroup, Network
from sklearn import datasets


def load_data():
    logging.info('Loading MNIST database...')
    images, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data')
    logging.info(f'MNIST database loaded: {len(images)} images of dimension {images[0].shape}')
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
    nb_excitator_neurons: int = 400
    nb_inhibitor_neurons: int = 400
    single_example_time = 0.35 * units.second
    resting_time = 0.15 * units.second
    input_intensity = 2
    start_input_intensity = input_intensity

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
    nu_ee_pre = 0.0001  # learning rate
    nu_ee_post = 0.01  # learning rate
    wmax_ee = 1.0

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
    eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    # ==================================== Network creation ====================================

    logging.info('Creating excitator and inhibitor neurons...')

    neurons_e = NeuronGroup(nb_excitator_neurons, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e,
                            method='euler')
    neurons_i = NeuronGroup(nb_inhibitor_neurons, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                            reset=v_reset_i, method='euler')

    spike_counters_e = SpikeMonitor(neurons_e, record=False)

    neurons_e.v = v_rest_e - 40. * units.mV
    neurons_i.v = v_rest_i - 40. * units.mV

    # Training only
    neurons_e.theta = np.ones(nb_excitator_neurons) * 20.0 * units.mV

    synapses_e_i = Synapses(neurons_e, neurons_i, model='w : 1', on_pre='ge_post += w')
    synapses_e_i.connect('i==j')  # One to one (diagonal only)
    synapses_e_i.w = '10.4'  # Not random?

    synapses_i_e = Synapses(neurons_i, neurons_e, model='w: 1', on_pre='gi_post += w')
    synapses_i_e.connect('i!=j')  # All except one (not diagonal only)
    synapses_i_e.w = '17.0'

    logging.info('Creating input neurons...')

    neurons_input = PoissonGroup(nb_input_neurons, 0 * units.Hz)

    model = 'w : 1'
    on_pre = 'ge_post += w'
    on_post = ''

    # Training only
    model += eqs_stdp_ee
    on_pre += '; ' + eqs_stdp_pre_ee
    on_post = eqs_stdp_post_ee

    synapses_input_e = Synapses(neurons_input, neurons_e, model=model, on_pre=on_pre, on_post=on_post)
    synapses_input_e.connect(True)  # All to all
    synapses_input_e.w = 'rand() * 0.3'

    min_delay = 0 * units.ms
    max_delay = 10 * units.ms
    delta_delay = max_delay - min_delay

    # Construct the network
    net = Network(neurons_input, neurons_i, neurons_e, synapses_i_e, synapses_e_i, synapses_input_e, spike_counters_e)

    # ======================================== Training ========================================
    logging.info(f'Start training on {nb_train_samples} images')

    previous_spike_count = np.zeros(nb_excitator_neurons)
    neurons_input.rates = 0 * units.Hz  # Necessary?
    net.run(0 * units.second)  # Why?

    # TODO add epoch
    for i, image in enumerate(images[:nb_train_samples]):
        logging.debug(f'Start step {i:03}/{nb_train_samples} ({i / nb_train_samples * 100:5.2f}%)')
        enough_spikes = False

        while not enough_spikes:
            normalize_weights(synapses_input_e)

            input_rates = image.reshape((nb_input_neurons)) / 8 * input_intensity
            neurons_input.rates = input_rates * units.Hz

            # Run the network
            net.run(single_example_time)

            current_spike_count = spike_counters_e.count - previous_spike_count
            previous_spike_count = np.copy(spike_counters_e.count)
            sum_spikes = np.sum(current_spike_count)

            # Check if enough spike triggered, if under the limit start again with the same image
            if sum_spikes < 5:
                input_intensity += 1
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                logging.debug(
                    f'Not enough spikes ({sum_spikes}), retry with higher intensity level ({input_intensity})')
            else:
                neurons_input.rates = 0 * units.Hz
                net.run(resting_time)
                input_intensity = start_input_intensity
                enough_spikes = True

    logging.info(f'Training is over')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info('Beginning of execution')

    run(nb_train_samples=50, nb_test_samples=50)
