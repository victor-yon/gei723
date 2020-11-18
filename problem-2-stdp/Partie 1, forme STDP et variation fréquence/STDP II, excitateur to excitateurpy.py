from brian2 import *

N = 1000
taum = 10*ms
taupre1 = 10*ms
taupre2 = 30*ms
taupost = 40*ms #taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 63*Hz  #15Hz, 6Hz et 10 Hz
gmax = .01
dApre1 = 0.12
dApre2 = -0.06
dApost = -0.06  #-dApre * taupre / taupost * 1.05
#dApost *= gmax
#dApre *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

# Cette variable nous permet de réinitialiser les instants de décharge après que la STDP s'opère dans la synapse
# On va utiliser la condition int(t_spike_a > t0) pour évaluer si oui ou non on opère le changement de poids
t0 = 0*second

eqs_stdp = '''
    w : 1
    t_spike_a : second 
    t_spike_b : second
'''
# On peut avoir accès au temps avec la variable t dans la syntaxe des équations de Brian2
on_pre = '''
    ge += w
    t_spike_a = t
    w = w + int(t_spike_b > t0) * dApost * exp((t_spike_b - t_spike_a)/taupost)      # le cas Delta t < 0
    t_spike_b = t0
'''
on_post = '''
    t_spike_b = t
    w = w + int(t_spike_a > t0) * dApre1 * exp(-(t_spike_b - t_spike_a)/taupre1) + int(t_spike_a > t0) * dApre2 * exp(-(t_spike_b - t_spike_a)/taupre2)    # le cas Delta t > 0
    t_spike_a = t0
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler')
S = Synapses(input, neurons,
             model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler'
             )
S.connect()
S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)

run(100*second, report='text')

suptitle('Fréquence de {} Hz'.format(F), fontsize=20)

subplot(311)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()


show()
