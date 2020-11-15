# Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
# https://brian2.readthedocs.io/en/latest/examples/synapses.STDP.html

from brian2 import *

if __name__ == '__main__':
    # Input parameters
    nb_neurons_input = 1000
    frequency_input = 15 * Hz

    taum = 10 * ms
    taupre = 20 * ms
    taupost = taupre
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms
    gmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax

    # ======================= Input =======================

    input_neurons = PoissonGroup(nb_neurons_input, rates=frequency_input)
    input_monitor = SpikeMonitor(input_neurons)

    # ======================== Core =======================

    eqs_neurons = '''
    dv/dt = (ge * (Ee-v) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    core = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr', method='euler')

    # ====================== Synapses =====================
    # Synapses: input -> core
    eqs_synapses = '''
    w : 1
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
    '''

    on_pre = '''
    ge += w
    Apre += dApre
    w = clip(w + Apost, 0, gmax)
    '''

    on_post = '''
    Apost += dApost
    w = clip(w + Apre, 0, gmax)
    '''

    synapses = Synapses(input_neurons, core, eqs_synapses, on_pre=on_pre, on_post=on_post, )
    synapses.connect()
    synapses.w = 'rand() * gmax'
    syn_monitor = StateMonitor(synapses, 'w', record=[0, 1])  # Record the 2 first synapses

    # ======================== Run ========================

    run(100 * second, report='text')

    # ======================= Plot ========================

    subplot(311)
    plot(synapses.w / gmax, '.k')
    ylabel('Weight / gmax')
    xlabel('Synapse index')
    subplot(312)
    hist(synapses.w / gmax, 20)
    ylabel('Number of synapses')
    xlabel('Weight / gmax')
    subplot(313)
    plot(syn_monitor.t / second, syn_monitor.w.T / gmax)
    xlabel('Time (s)')
    ylabel('Weight / gmax')
    tight_layout()
    show()
