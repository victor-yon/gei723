## [OSCILLATEUR IetD avec courant de fuite]

## Import
from brian2 import *
import matplotlib as plt

## [Neurone avec fuite]
#on définit 3 vitesses possible, enclenché par la touche a "avancer"
#Avancer = ""
#Commande = input("On avance ? ")
#Avancer += str(Commande)
#vitesse = Avancer.count("a")

#C'est ce paramétre de vitesse qu'il faut faire varier
#ça fonctionne pour les vitesses 1,2,6,7,8 et 10
#A partir de la vitesse 6, les sauts synaptiques ne sont plus totalement verticaux

vitesse = 1

start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
td : ms
w : 1
I : 1
tau : ms
th : 1
vitesse : 1

'''
vitesse = 1

Declencheur = NeuronGroup(1, eqs, threshold= 't == 5*ms', reset='v = 0', method='euler')
G = NeuronGroup(2, eqs, threshold= 'v >= th', reset= 'v = 0', method='exact')

Declencheur.I = [2]
Declencheur.tau = [10]*ms


G.th = [0.8,0.8]
G.I = [0, 0]
G.tau = [10, 50]*ms
td = -1*G.tau*log(1-G.th/2) #/G.I) a définir peut être aprés la synapse pour que I(2) soit diff de 0
w = G.th/(1+exp(-td/(G.tau[1]))+exp(-2*td/(G.tau[1]))+exp(-3*td/(G.tau[1])))
poid0 = w[0]
# print(w)
# print(1+exp(-td/(G.tau[1]))+exp(-2*td/(G.tau[1]))+exp(-3*td/(G.tau[1])))
# print(td)


S_declenche = Synapses(Declencheur, G, on_pre='I_post =2*vitesse; I_pre = 0') #Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
S_declenche.connect(i=0, j=0)

#S1 = Synapses(G, G, on_pre='v_post += poid0', on_post = 'I_pre = 0; I_post = 2') #; tau_pre = (10/vitesse)*second ; tau_post = (10/vitesse)*second')
#S1.connect(i=0, j=1)
#S1.connect(i = 1, j=0)

#S1.w = 'th/(1+exp(-td/(50*second))+exp(-2*td/(50*second))+exp(-3*td/(50*second)))'
#attention, ici je n'ai pas mis tau_post mais j'ai rentré 50/vitesse

#S1.delay = 10*ms




M = StateMonitor(G, 'v', record=True)
Dec = StateMonitor(Declencheur, 'v', record=True)

run(50*ms)

plot(M.t/ms, Dec.v[0], label='Neuron dec')
plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend()
show()
