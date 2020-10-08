from brian2 import *

SENSOR_INPUT = 0.2
TRAIN_SIZE = 4


def build_direction_nn():
    start_scope()
    duration = 150 * ms

    # ========== Core Main =========
    # 1 IF neuron
    eqs_core = '''
    dv/dt = -v/tau : 1
    tau : second
    th : 1
    '''

    core_main = NeuronGroup(1, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core_main.tau = 10 * ms
    core_main.th = 0.8

    # ========= Core Inhib =========
    # 1 LIF neuron
    eqs_core_inhib = '''
    dv/dt = (I-v)/tau : 1
    tau : second
    I : 1
    th : 1
    '''

    core_inhib = NeuronGroup(1, eqs_core_inhib, threshold='v > th', reset='v = I', method='exact')
    core_inhib.tau = 10 * ms
    core_inhib.th = 1
    core_inhib.I = SENSOR_INPUT

    # ======== Main to Inhib =======
    syn_main_inhib = Synapses(core_inhib, core_main, on_pre='v_post -= 1')
    syn_main_inhib.connect(i=0, j=0)

    # Fake oscillator for testing
    eqs_osci = '''
    dv/dt = I/tau : 1
    I : 1
    tau : second
    '''

    fake_osci = NeuronGroup(1, eqs_osci, threshold='v >= 1 ', reset='v = 0', method='euler')
    fake_osci.I = 2
    fake_osci.tau = [50] * ms

    syn_osci_core = Synapses(fake_osci, core_main, on_pre='v_post += 1')
    syn_osci_core.connect(i=0, j=0)
    # Delay required because if the 2 pulse (-1 and +1) arrived at the same time only one is take into account
    syn_osci_core.delay = 1 * ms
    syn_osci_inhib = Synapses(fake_osci, core_inhib, 'w : 1', on_pre='v_post += w')
    syn_osci_inhib.connect(i=0, j=0)
    syn_osci_inhib.w = 1 / TRAIN_SIZE

    state_mon_osci = StateMonitor(fake_osci, 'v', record=True)

    # Monitoring
    state_mon_main = StateMonitor(core_main, 'v', record=True)
    state_mon_inhib = StateMonitor(core_inhib, 'v', record=True)

    run(duration)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(state_mon_osci.t / ms, state_mon_osci.v[0], color='black', label='Fake Oscillator')
    ax1.axhline(y=SENSOR_INPUT, color='magenta', linestyle='--', label='Frontal Sensor (I)')
    ax2.plot(state_mon_inhib.t / ms, state_mon_inhib.v[0], color='blue', label='Core Inhib')
    ax3.plot(state_mon_main.t / ms, state_mon_main.v[0], color='red', label='Core Main')
    xlabel('Time (ms)')
    ylabel('v')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    show()


if __name__ == '__main__':
    build_direction_nn()
