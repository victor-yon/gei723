from brian2 import *
import numpy as np


N = 1000
taum = 10*ms
taupre = 10*ms #200*ms
taupost = 200*ms #taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
gmax = .01
dApre = 0.01
dApost = -0.5 #-dApre * taupre / taupost #* 1.05
dApost *= gmax
dApre *= gmax
Point = np.array([ 500, 500, 500 ,500 , 500, 100, 75, 7, 6, 0.6, 0.05,0.05, 0.05, 0.05, 0.05, 0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.071, -0.072, -0.073, -0.074, -0.075, -0.072, -0.07, -0.065, -0.06,  -0.05, -0.045, -0.04, -0.035, -0.03, -0.02, -0.01, -0.005 -0.0015, -0.0012, -0.001, -0.00050, -0.0005 ,0])*100
x = np.array([-3000,-2000,-1000,-900,-800,-750,-500,-400, -250, -150, 0, 1, 2, 3, 6,12, 15, 17, 20, 22, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,55, 60, 65, 70, 75, 80, 90, 100,125, 150, 175, 1000])
z = np.polyfit(x, Point,15)
p = np.poly1d(z)

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
    w =  int(t_spike_b > t0) * dApost * exp((t_spike_b - t_spike_a)/taupost)      # le cas Delta t < 0
    t_spike_b = t0
'''
on_post = '''
    t_spike_b = t
    w = w + int((t_spike_b - t_spike_a) < 20 )*int(t_spike_a > t0)*dApre * exp(-(t_spike_b - t_spike_a)/taupre) + int(20<= (t_spike_b - t_spike_a ))*int((t_spike_b - t_spike_a )<= 50)*nt(t_spike_a > t0)*p(t_spike_b - t_spike_a)
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

##[Trouver l'approximation polynomial]

z = np.polyfit(L, Cor, 10)
#    p = np.poly1d(z)
