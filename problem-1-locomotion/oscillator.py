from brian2 import *


# Oscillations with 2 Integrate & Fire neurons
def oscillator_if():
    start_scope()

    duration = 200 * ms
    N = 2
    eqs = '''
    v : 1
    '''

    main_grp = NeuronGroup(N, eqs, threshold='v >= 1 ', reset='v = 0', method='exact')
    main_grp.v = [1, 0]
    state_mon = StateMonitor(main_grp, 'v', record=True)

    synapses_main = Synapses(main_grp, main_grp, 'w : 1', on_pre='v_post += w')
    synapses_main.connect(i=0, j=1)
    synapses_main.connect(i=1, j=0)
    synapses_main.w = [1, 1]
    synapses_main.delay = 10 * ms

    run(duration)

    for i in range(N):
        plot(state_mon.t / ms, state_mon.v[i], label=f'Neurone {i}')

    xlabel('Time (ms)')
    ylabel('v')
    legend()
    show()


if __name__ == '__main__':
    oscillator_if()
