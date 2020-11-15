from brian2 import *

from util_plots import stdp_shape


def stdp_with_time():
    eqs_stdp_test = '''
        w : 1
        t_spike_a : second
        t_spike_b : second
        tau_a = 20 * ms : second
        tau_b = 20 * ms : second
        A = 0.01 : 1
        B = -0.01 : 1
        t0 = 0 * second : second
    '''
    # On peut avoir accès au temps avec la variable t dans la syntaxe des équations de Brian2
    on_pre_test = '''
        v_post += w
        t_spike_a = t
        w = w + int(t_spike_b > t0) * B * exp((t_spike_b - t_spike_a)/tau_b) # le cas Delta t < 0
        t_spike_b = t0
    '''
    on_post_test = '''
        t_spike_b = t
        w = w + int(t_spike_a > t0) * A * exp(-(t_spike_b - t_spike_a)/tau_a) # le cas Delta t > 0
        t_spike_a = t0
    '''

    stdp_shape(eqs_stdp=eqs_stdp_test, on_pre=on_pre_test, on_post=on_post_test)


def stdp_with_trace():
    # Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
    # https://brian2.readthedocs.io/en/latest/examples/synapses.STDP.html

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


if __name__ == '__main__':
    stdp_with_time()
