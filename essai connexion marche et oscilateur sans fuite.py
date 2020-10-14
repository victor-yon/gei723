## [Essai de connection]

## Import
from brian2 import *

## [Neurone sans fuite]

## [Commande]


#on définit 3 vitesses possible, enclenché par la touche a "avancer"
#Avancer = ""
#Commande = input("On avance ? ")
#Avancer += str(Commande)
#vitesse = Avancer.count("a")

#C'est ce paramétre de vitesse qu'il faut faire varier
#ça fonctionne pour les vitesses 5 compris
#Quand on repart de 0, il ne reste pas à zero jusqu'au saut synaptique ..


vitesse = 10
droite = 1
gauche = 2

## [Equation]


#equation utilisée
eqs = '''
dv/dt = I/tau : 1 
I : 1
tau : second
th : 1
'''
## [Oscilateur neurone]


#3 Neurones : 1 Pour déclencher le signal, 2 qui oscillent
#On définit le temps que mets le neurone déclencheur à transmettre l'info à 5ms

Declencheur = NeuronGroup(1, eqs, threshold= 't == 5*ms', reset='v = 0', method='euler')
G =  NeuronGroup(2, eqs, threshold= 'v>= th', reset='v = 0', method='euler')

G.th=0.8
Declencheur.I = [0.5]
G.I = [0,0]
Declencheur.tau = [2.5]*ms

#La variable tau défini la fréquence de déclenchement, elle doit varier avec la vitesse
G.tau = 31.25*ms  #a la base, 10/vitesse*ms

## [Marche neurones]

#Nombre de paires de pattes

N = 3

Npaire = N//2
Nimpaire = (N+1)//2

Droit_paire = NeuronGroup(Npaire,eqs, threshold= 'v>= th', reset='v = 0', method='euler')
Droit_impaire = NeuronGroup(Nimpaire,eqs, threshold= 'v>= th', reset='v = 0', method='euler')
Gauche_paire = NeuronGroup(Npaire,eqs, threshold= 'v>= th', reset='v = 0', method='euler')
Gauche_impaire = NeuronGroup(Nimpaire,eqs, threshold= 'v>= th', reset='v = 0', method='euler')

#Seuil
Droit_paire.th = 0.8
Droit_impaire.th = 0.8
Gauche_paire.th = 0.8
Gauche_impaire.th = 0.8

#Constante de temps
Droit_paire.tau = 31.25/vitesse*ms
Droit_impaire.tau = 31.25/vitesse*ms
Gauche_paire.tau = 31.25/vitesse*ms
Gauche_impaire.tau = 31.25/vitesse*ms

#intensité dans les neurones
Droit_paire.I = 0
Droit_impaire.I = 0
Gauche_paire.I = 0
Gauche_impaire.I = 0

## [Connexion neurone déclencheur oscillation]


#Déclenchement de la marche
S_declenche = Synapses(Declencheur, G, on_pre='I_post =2*vitesse; I_pre = 0') #Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
S_declenche.connect(i=0, j=0)

## [Oscilation]


#Oscillation
#a = 0
S_oscil =Synapses(G, G, on_pre='v_post += 0.2', on_post='I_pre = 0; I_post = 2*vitesse') # on_post = '''I_pre = 0; I_post = 2'''
S_oscil.connect(i = 0, j=1)
S_oscil.connect(i = 1, j=0)
S_oscil.delay = (G.th*G.tau*1/(2*vitesse))*(39/32) #/2


Dec = StateMonitor(Declencheur, 'v', record=True) 
M = StateMonitor(G, 'v', record=True)
Dimp = StateMonitor(Droit_impaire, 'v', record=0)
Dpair = StateMonitor(Droit_paire, 'v', record=0)
Gpair = StateMonitor(Gauche_paire, 'v', record=0)
Gimp = StateMonitor(Gauche_impaire, 'v', record=0)

spikemon = SpikeMonitor(G)
nombre = spikemon.count

## [Connexion à la marche][Neurone qui sont reliés à la premiére partie de l'oscillateur]

#Relier l'oscilateur aux pattes. "nombre" compte le nombre de fois que les 2 neurones
#De l'oscilateur déclenchent


S_gauche_paire = Synapses(G, Gauche_paire, on_pre = 'v_post += th/droite')
S_droite_impaire =  Synapses(G, Droit_impaire, on_pre = 'v_post += th/gauche')

S_gauche_paire.connect(i = 0, j = [k for k in range(Npaire)])
#S_gauche_paire.delay = 2*ms
S_droite_impaire.connect(i = 0, j=[k for k in range(Nimpaire)])
#S_droite_impaire.delay = 2*ms

## [Connexion à la marche][Neurone qui sont reliés à la deuxiéme partie de l'oscillateur]

S_gauche_impaire = Synapses(G, Gauche_impaire, on_pre = 'v_post += th/droite')
S_droite_paire =  Synapses(G, Droit_paire, on_pre = 'v_post += th/gauche')
S_gauche_impaire.connect(i = 1, j=[k for k in range(Nimpaire)])
#S_gauche_paire.delay = 2*ms
S_droite_paire.connect(i = 1, j=[k for k in range(Npaire)])
#S_droite_impaire.delay = 2*ms


## [Run et affichage]

run(300*ms)

pic_n1 = nombre[0]
pic_n2 = spikemon.count[1]
print(nombre, pic_n1, pic_n2 )

plot(M.t/ms, Dec.v[0], label='Neuron D')
plot(M.t/ms, M.v[0], "r-" ,label='Neuron 1')
plot(M.t/ms, M.v[1], "b-" ,label='Neuron 2')
#plot(M.t/ms, Dimp.v[0], label = 'Neuron droit impaire')
#plot(M.t/ms, Dpair.v[0], label = 'Neuron droit paire')
#plot(M.t/ms, Gimp.v[0], label = 'Neuron gauche impaire')
#plot(M.t/ms, Gpair.v[0], label = 'Neuron gauche paire')


xlabel('Time (ms)')
ylabel('v')
legend()
title("Oscilation du CPG pour la vitesse : {0}".format(vitesse))
show()


