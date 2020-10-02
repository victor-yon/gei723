## [OSCILLATEUR]

## Import
from brian2 import *

## [Neurone sans fuite]
#on définit 3 vitesses possible, enclenché par la touche a "avancer"
#Avancer = ""
#Commande = input("On avance ? ")
#Avancer += str(Commande)
#vitesse = Avancer.count("a")

#C'est ce paramétre de vitesse qu'il faut faire varier
#ça fonctionne pour les vitesses 1,2,6,7,8 et 10
#A partir de la vitesse 6, les sauts synaptiques ne sont plus totalement verticaux

vitesse = 2

#equation utilisée
eqs = '''
dv/dt = I/tau : 1 
I : 1
tau : second
th : 1
'''

#3 Neurones : 1 Pour déclencher le signal, 2 qui oscillent
#On définit le temps que mets le neurone déclencheur à transmettre l'info à 5ms

Declencheur = NeuronGroup(1, eqs, threshold= 't == 5*ms', reset='v = 0', method='euler')
G =  NeuronGroup(2, eqs, threshold= 'v>= th', reset='v = 0', method='euler')

G.th=0.8
Declencheur.I = [2]
G.I = [0,0]
Declencheur.tau = [10]*ms

#La variable tau défini la fréquence de déclenchement, elle doit varier avec la vitesse
G.tau = 10/vitesse *ms

#Déclenchement de la marche
S_declenche = Synapses(Declencheur, G, on_pre='I_post =2; I_pre = 0') #Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
S_declenche.connect(i=0, j=0)

#Oscillation
#a = 0
S_oscil =Synapses(G, G, on_pre='v_post += 0.2', on_post = '''I_pre = 0; I_post = 2''')
S_oscil.connect(i = 0, j=1)
S_oscil.connect(i = 1, j=0)
S_oscil.delay = G.th*G.tau /2

Dec = StateMonitor(Declencheur, 'v', record=True) 
M = StateMonitor(G, 'v', record=True)

spikemon = SpikeMonitor(G)
nombre = spikemon.count

run(50*ms)
pic_n1 = nombre[0]
pic_n2 = spikemon.count[1]
print(nombre, pic_n1, pic_n2 )

plot(M.t/ms, Dec.v[0], label='Neuron 0')
plot(M.t/ms, M.v[0], label='Neuron 1')
plot(M.t/ms, M.v[1], label='Neuron 2')
xlabel('Time (ms)')
ylabel('v')
legend()
title("Vitesse : {0}".format(vitesse))
plt.show()

#Faire une commande, quand on presse le fleche, I du neurone déclencheur est initialisé
#a 2, donc il charge, et déclenche au bout de 5ms


