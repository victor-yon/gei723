from brian2 import *


# Oscillations with Leaky Integrate & Fire neurons
def oscillations_lif():
    start_scope()
    N = 2
    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''

    captor_grp = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')
    captor_grp.I = [4]
    captor_grp.tau = [100] * ms
    state_mon_c = StateMonitor(captor_grp, 'v', record=True)
    spike_mon_c = SpikeMonitor(captor_grp, record=True)

    main_grp = NeuronGroup(N, eqs, threshold='v > 0.8', reset='v = 0', method='exact')
    main_grp.I = [0, 0]
    main_grp.tau = [100, 100] * ms
    state_mon = StateMonitor(main_grp, 'v', record=True)
    spike_mon = SpikeMonitor(main_grp, record=True)

    synapses_main = Synapses(main_grp, main_grp, 'w : 1', on_pre='v_post += w')
    synapses_main.connect(i=0, j=1)
    synapses_main.connect(i=1, j=0)
    synapses_main.w = [0.3, 0.3]
    synapses_main.delay = 10 * ms

    synapses_captor = Synapses(captor_grp, main_grp, 'w : 1', on_pre='v_post += w')
    synapses_captor.connect(i=0, j=0)
    synapses_captor.w = [1]
    synapses_captor.delay = 5 * ms

    run(200 * ms)

    for i in range(N):
        plot(state_mon.t / ms, state_mon.v[i], label=f'Neurone {i}')

    plot(state_mon_c.t / ms, state_mon_c.v[0], '--', label=f'Neurone Captor')

    xlabel('Time (ms)')
    ylabel('v')
    legend()
    show()


if __name__ == '__main__':
    oscillations_lif()
