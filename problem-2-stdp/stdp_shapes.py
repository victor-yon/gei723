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
    run_time = 100 * second

    # ======================= Input =======================

    input_neurons = PoissonGroup(nb_neurons_input, rates=frequency_input)

    # ======================== Core =======================

    eqs_neurons = '''
        vt = -54 * mV : volt
        vr = -60 * mV : volt
        Ee = 0 * mV : volt
        El = -74 * mV : volt
        taum = 10 * ms : second
        taue = 5 * ms : second
        
        dv/dt = (ge * (Ee-v) + El - v) / taum : volt
        dge/dt = -ge / taue : 1
    '''

    core_neuron = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr', method='euler')

    # ====================== Synapses =====================
    # Synapses: input -> core
    eqs_synapses = '''
        w : 1
        gmax = 0.01 : 1
        taupre = 20 * ms : second
        taupost = taupre : second
        dApre = 0.01 * gmax : 1
        dApost = -dApre * taupre / taupost * 1.05 : 1  # include gmax from dApre
        
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

    # Compute and show the STDP shape
    stdp_shape(eqs_stdp=eqs_synapses, on_pre=on_pre, on_post=on_post, neuron_variable='ge : 1')

    synapses = Synapses(input_neurons, core_neuron, eqs_synapses, on_pre=on_pre, on_post=on_post)
    synapses.connect()
    synapses.w = 'rand() * gmax'
    syn_monitor = StateMonitor(synapses, 'w', record=[0, 1])  # Record the 2 first synapses

    # ======================== Run ========================

    run(run_time, report='text')

    # ======================= Plot ========================

    subplot(311)
    plot(synapses.w / synapses.gmax, '.k')
    ylabel('Weight / gmax')
    xlabel('Synapse index')
    subplot(312)
    hist(synapses.w / synapses.gmax, 20)
    ylabel('Number of synapses')
    xlabel('Weight / gmax')
    subplot(313)
    plot(syn_monitor.t / second, syn_monitor.w.T / synapses.gmax)
    xlabel('Time (s)')
    ylabel('Weight / gmax')
    tight_layout()
    show()


def diehl_cook():
    stdp_synapse_model = '''
        w : 1
        
        tc_pre_ee = 20 * ms : second
        tc_post_1_ee = 20 * ms : second
        tc_post_2_ee = 40 * ms : second
        
        nu_ee_presyn = 0.1 : 1
        nu_ee_postsyn = 0.1 : 1
    
        plastic : boolean (shared) # Activer/désactiver la plasticité
        
        post2before : 1  # x_tar ?
    
        dpre/dt    = -pre/(tc_pre_ee) : 1 (event-driven)
    
        dpost1/dt  = -post1/(tc_post_1_ee) : 1 (event-driven)
    
        dpost2/dt  = -post2/(tc_post_2_ee) : 1 (event-driven)  # (?)
    
        wmax = 10 : 1  # Completed
    
        mu = 1 : 1  # Completed
    '''

    # Completed
    stdp_pre = '''
        ge_post += w
        
        pre = 1.
        
        w = clip(w + (nu_ee_presyn * post1), 0, wmax)
    '''

    # Completed
    # w = clip(w + nu_ee_postsyn*(pre-post2before)*(wmax - w)**mu, 0, wmax)
    stdp_post = '''
        post2before = post2
        
        w = clip(w + (nu_ee_postsyn * pre), 0, wmax)
        
        post1 = -1.
    
        post2 = 1.
    '''

    stdp_shape(eqs_stdp=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post, neuron_variable='ge : 1')


if __name__ == '__main__':
    stdp_with_trace()
