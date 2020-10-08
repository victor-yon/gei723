## [OSCILLATEUR IetD avec courant de fuite]

## Import
from brian2 import *
import matplotlib as plt

## [Neurone avec fuite]

## [Paramétres]


#on définit 3 vitesses possible, enclenché par la touche a "avancer"
#Avancer = ""
#Commande = input("On avance ? ")
#Avancer += str(Commande)
#vitesse = Avancer.count("a")

#La vitesse influe sur l'intensité du signal

vitesse = 10

start_scope()

## [Equation]

eqs = '''
dv/dt = (I-v)/tau : 1
td : second
I : 1
Icircuit : 1
tau : second
th : 1
'''


## [Groupe de neurone, declencheur et oscillateur]

Declencheur = NeuronGroup(1, eqs, threshold= 't == 5*ms', reset='v = 0', method='euler')
G = NeuronGroup(2, eqs, threshold= 'v >= th', reset= 'v = 0', method='exact')

Declencheur.I = [2]
Icircuit = 2

Declencheur.tau = [10]*ms

G.I = [0,0]
G.tau = [10, 50]*ms

G.th = 0.8
td = -1*10*ms*log(1-0.8/(Icircuit*vitesse))   #-1*G.tau*log(1-G.th/(2*vitesse))


## [Connexion neurone déclencheur oscillation]

#Déclenchement de la marche
S_declenche = Synapses(Declencheur, G, on_pre='I_post = 2*vitesse; I_pre = 0')
S_declenche.connect(i=0, j=0)

## [Oscillation]

S_oscil =Synapses(G, G,'w:1', on_pre='v_post += w ',on_post='I_pre = 0; I_post = 2*vitesse; tau_pre = 50*ms ; tau_post = 10*ms') 

S_oscil.connect(i = 0, j=1)
S_oscil.connect(i = 1, j=0)

S_oscil.w = 0.8/(1+exp(-td/(50*ms))+exp(-2*td/(50*ms))+exp(-3*td/(50*ms)))#G.th[0]/(1+exp(-td/G.tau)+exp(-2*td/G.tau)+exp(-3*td/G.tau))




## [Run et affichage]
M = StateMonitor(G, 'v', record=True)
Dec = StateMonitor(Declencheur, 'v', record=True)

run(30*ms)

plot(M.t/ms, Dec.v[0], label='Neuron dec')
plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend()
show()

