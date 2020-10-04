from brian2 import *

SENSOR_INPUT = 0
OSCILLATOR_INPUT_INDEX = 1


def build_ground_contact_nn():
    start_scope()
    duration = 150 * ms

    # ======= Frontal sensor =======
    # 1 LIF neuron
    eqs_sensor = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''

    frontal_sensor = NeuronGroup(1, eqs_sensor, threshold='v >= 1 ', reset='v = 0', method='exact')
    frontal_sensor.I = [SENSOR_INPUT]
    frontal_sensor.tau = [10] * ms

    # ======== Core Network ========
    # 2 LIF neuron
    eqs_core = '''
    dv/dt = -v/tau : 1
    tau : second
    th : 1
    '''

    core = NeuronGroup(2, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core.tau = [50, 50] * ms
    core.th = [0.5, 1.5]

    # ======== Ground Motor ========
    # 1 LIF neuron
    eqs_motor = '''
    dv/dt = -v/tau : 1
    tau : second
    '''

    motor = NeuronGroup(1, eqs_motor, threshold='v > 0.1', reset='v = 0', method='exact')
    motor.tau = [50] * ms

    # ======= Sensor to Core =======
    syn_sensor_core = Synapses(frontal_sensor, core, 'w : 1', on_pre='v_post += w')
    syn_sensor_core.connect(i=0, j=[0, 1])
    syn_sensor_core.w = [-1, 1]

    # ======= Core to Motor ========
    syn_core_motor = Synapses(core, motor, on_pre='v_post += 1')
    syn_core_motor.connect(i=[0, 1], j=0)

    # Fake oscillator for testing
    eqs_osci = '''
    dv/dt = I/tau : 1
    I : 1
    tau : second
    '''

    fake_osci = NeuronGroup(1, eqs_osci, threshold='v >= 1 ', reset='v = 0', method='euler')
    fake_osci.I = 2
    fake_osci.tau = [50] * ms
    state_mon_osci = StateMonitor(fake_osci, 'v', record=True)
    syn_osci_core = Synapses(fake_osci, core, on_pre='v_post += 1')
    syn_osci_core.connect(i=0, j=OSCILLATOR_INPUT_INDEX)

    # Monitoring
    state_mon_sensor = StateMonitor(frontal_sensor, 'v', record=True)
    state_mon_core = StateMonitor(core, 'v', record=True)
    state_mon_motor = StateMonitor(motor, 'v', record=True)

    run(duration)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(state_mon_osci.t / ms, state_mon_osci.v[0], color='magenta', label='Fake Oscillator')
    ax1.plot(state_mon_sensor.t / ms, state_mon_sensor.v[0], '--', color='black', label='Frontal Sensor')
    ax2.plot(state_mon_core.t / ms, state_mon_core.v[0], color='blue', label='Core 0')
    ax2.plot(state_mon_core.t / ms, state_mon_core.v[1], color='green', label='Core 1')
    ax3.plot(state_mon_motor.t / ms, state_mon_motor.v[0], color='red', label='Motor')
    xlabel('Time (ms)')
    ylabel('v')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    show()


if __name__ == '__main__':
    build_ground_contact_nn()
