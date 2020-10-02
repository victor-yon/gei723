from brian2 import *
import matplotlib
import numpy as np

start_scope()
# Neurones qui génèrent les mouvements
# -------------------------------------------
# Constantes

N = 3
eqs = '''
dv/dt = I/tau : 1
I : 1
tau : second
th : 1
'''
TAU = 10
duration = 200*ms

# Besoin en entrée des spikes venant des neurones B_GD et A ainsi que le nombre de paire de pattes
#
def mouvement_pattes(N, eqs):
    # up. Doit lancer une décharge qui met
    up = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method='exact')

    # Avant lance une décharge à chaque spike de B
    avant = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method='exact')

    # Arrière lance des décharges sur l'autre couleur par rapport à avant
    arriere = NeuronGroup(2 * N, eqs, threshold='v >= th', reset='v = 0', method='exact')
    commandes = NeuronGroup(4 * N, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    compteur = NeuronGroup(2 * N, eqs, threshold='v >= th', reset='v = 0', method='exact')
    # Indiquer comment faire les connexions avec A et B
    # Logique: up = 1, avant = 2, arriere = 3
    # Vert sera pour le mouvement avant des pattes paires et le mouvement arriere des pattes impaires.
    # Bleu inverse
    # ordre_synapses = ('Comment connecter les synapses :' \
    #                  'pattes impaires A1 --> (0, 2, 4):'\n
    #                  'pattes paires A2 --> (1, 3, 5)  ')

    print()
    return up, avant, arriere, commandes, compteur

# Test pour simuler A et B
def neuron_test(N, eqs, up, avant, arriere):
    A = NeuronGroup(2, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    B = NeuronGroup(2, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact') # ou 4?

    return A, B

# Capteur qui change les poids synaptiques
def capteurs(N, A, 1, 2):
    capteurA1 = NeuronGroup(2, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    capteurA2 = NeuronGroup(2, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')

    return



up, avant, arriere, commandes, compteur = mouvement_pattes()
A, B = neuron_test()

# Synapses
S_up_impaires = Synapses(A1, up, 'w : 1')
S_up_impaires.connect(i = 0, j = [0:2*N-1:2], skip_if_invalid=True)

S_up_paires = Synapses(A2, up, 'w : 1')
S_up_paires.connect(i = 0, j = [1:2*N:2], skip_if_invalid=True)
# S_up_impaires.tau = TAU*np.eye(1,N/2)
# S_up_paires.tau = TAU*np.eye(1,N/2)

# Synapses B --> compteur. Si 2 ou 4B, on ajuste avec paires et impaire
S_compteur = Synapses(B, compteur, 'w : 1')
S_compteur.connect()
S_compteur.w = 1 # à adapter avec le nombre de connexions

# Synapses B --> commandes dépend du message de soline
S_B_commandes = Synapses(B, commandes, 'w : 1')
S_B_commandes.connect()

# Synapses compteur --> commandes à connecter si besoin d'un compteur
S_commandes = Synapses(compteur, commandes, 'w : 1', on_pre = 'S_B_commandes[1:N:2] = S_B_commandes[0:N-1:2], S_B_commandes[0:N-1:2] = S_B_commandes[1:N:2]')
S_commandes.connect()

# Synapses commandes --> moteurs
S_avant = Synapses(commandes, avant,)
S_arriere = Synapses(commmandes, arriere, )
# impaires
S_avant.connect(j='2*i', skip_if_invalid=True)
S_arriere.connect(j = '(2*i) + 1', skip_if_invalid=True)
# paires
S_avant.connect(j='2*i', skip_if_invalid=True)
S_arriere.connect(j = '(2*i) + 1', skip_if_invalid=True)

run(duration)

# Graphiques
M_up = StateMonitor(up)
spike_up = SpikeMonitor(up)

M_avant = StateMonitor(avant)
spike_avant = SpikeMonitor(avant)

M_arriere = StateMonitor(arriere)
spike_arriere = SpikeMonitor(arriere)

plot(spike_up.t/ms, spike_up.i, '.k')
# plot(spike_avant.t/ms, spike_avant.i, '.b')
# plot(spike_arriere.t/ms, spike_avant.i, '.g')

xlabel('instant de décharge')
ylabel('indice neurone')
legend()
show()





