from brian2 import *

import matplotlib
import numpy as np

start_scope()
# Neurones qui génèrent les mouvements
# -------------------------------------------
# Constantes

N = 3
vitesse = 3
eqs = '''
dv/dt = I/tau : 1
I : 1
tau : second
th : 1
'''
TAU = 10
duration = 200*ms

def oscillateur(N, eqs, vitesse):
    Declencheur = NeuronGroup(1, eqs, threshold='t == 5*ms', reset='v = 0', method='euler')
    G = NeuronGroup(2, eqs, threshold='v>= th', reset='v = 0', method='euler')

    G.th = 0.8
    Declencheur.I = [2]
    G.I = [0, 0]
    Declencheur.tau = [10] * ms
    G.tau = 10 / vitesse * ms

    # Déclenchement de la marche
    S_declenche = Synapses(Declencheur, G, on_pre='I_post =2; I_pre = 0')
    S_declenche.connect(i=0, j=0)

    # Oscillation
    # a = 0
    S_oscil = Synapses(G, G, on_pre='v_post += 0.2; I_pre = 0; I_post = 2')  # on_post = '''I_pre = 0; I_post = 2'''
    S_oscil.connect(i=0, j=1)
    S_oscil.connect(i=1, j=0)
    S_oscil.delay = G.th * G.tau * 1.5  # /2

    return Declencheur, G, S_declenche, S_oscil
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

    # Créer compteur Droite, gauche, paire, impaire
    # Droite impaire
    CDI = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='exact')
    # Droite paire
    CDP = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='exact')
    # Gauche impaire
    CGI = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='exact')
    # Gauche paire
    CGP = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='exact')

    return up, avant, arriere, commandes, CDI, CDP, CGI, CGP

# Test pour simuler A et B
def neuron_test(N, eqs):
    A1 = NeuronGroup(1, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    A2 = NeuronGroup(1, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    # B = NeuronGroup(2, eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact') # ou 4?
    Npaire = N//2
    Nimpaire = (N+1)//2
    Droit_paire = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method='euler')
    Droit_impaire = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method='euler')
    Gauche_paire = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method='euler')
    Gauche_impaire = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method='euler')


    return A1, A2, Droit_paire, Droit_impaire, Gauche_paire, Gauche_impaire

# Capteur qui change les poids synaptiques
def capteurs(G):
    capteurA1 = NeuronGroup(G[0], eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')
    capteurA2 = NeuronGroup(G[1], eqs, threshold = 'v >= th', reset = 'v = 0', method = 'exact')

    return capteurA1, capteurA2


def connexion_synaptiques(G, A1, A2, DI, DP, GI, GP, up, avant, arriere, CDI, CDP, CGI, CGP):
    # Synapses
    S_up_impaires = Synapses(A1, up, 'w : 1')
    S_up_impaires.connect(i=0, j='[1:2*N-1:2]', skip_if_invalid=True)

    S_up_paires = Synapses(A2, up, 'w : 1')
    S_up_paires.connect(i = 0, j = '[1:2*N:2]', skip_if_invalid=True)
    # S_up_impaires.tau = TAU*np.eye(1,N/2)
    # S_up_paires.tau = TAU*np.eye(1,N/2)

    # 16 groupes Synapses
    # Vers les neurones commande
    # GI --> CGI commandes
    Syn_GI_CGI = Synapses(GI, CGI, on_pre='v_post = th')
    Syn_GI_CGI.connect(i = 0, j = 0, skip_if_invalid=True)

    # GP --> CGP commandes
    Syn_GP_CGP = Synapses(GP, CGP, on_pre='v_post = th')
    Syn_GP_CGP.connect(i = 0, j = 0, skip_if_invalid=True)

    # DI --> CDI commandes
    Syn_DI_CDI = Synapses(DI, CDI, on_pre='v_post = th')
    Syn_DI_CDI.connect(i = 0, j = 0, skip_if_invalid=True)

    # DP --> CDP commandes
    Syn_DP_CDP = Synapses(DP, CDP, on_pre='v_post = th')
    Syn_DP_CDP.connect(i = 0, j = 0, skip_if_invalid=True)

    # 1 --> CGI changement poids
    Syn_1_CGI = Synapses(G[0], CGI, on_pre='S')
    Syn_1_CGI.connect(i = 0, j = 0, skip_if_invalid=True)

    # 1 --> CDP changement poids
    Syn_1_CDP = Synapses(G[0], CDP, on_pre='')
    Syn_1_CDP.connect(i = 0, j = 0, skip_if_invalid=True)

    # 2 --> CGP changement poids
    Syn_2_CGP = Synapses(G[1], CGP, on_pre='')
    Syn_2_CGP.connect(i = 0, j = 0, skip_if_invalid=True)

    # 2 --> CDI changement poids
    Syn_2_CDI = Synapses(G[1], CDI, on_pre='')
    Syn_2_CDI.connect(i = 0, j = 0, skip_if_invalid=True)

    # vers les neurones action
    # CGI --> avant
    Syn_CGI_avant = Synapses(CGI, avant, on_pre='v_post = th')
    Syn_CGI_avant.connect(i = 0, j = '[0:N//2:2]', skip_if_invalid=True)
    Syn_CGI_avant.w = np.eye[1, N+1//2]

    # CGI --> arriere
    Syn_CGI_arriere = Synapses(CGI, arriere, on_pre='v_post = th')
    Syn_CGI_arriere.connect(i = 0, j = '[0:N//2:2]', skip_if_invalid=True)
    Syn_CGI_arriere.w = np.eye[1, N+1//2]

    # CDI --> avant
    Syn_CDI_avant = Synapses(CDI, avant, on_pre='v_post = th')
    Syn_CDI_avant.connect(i = 0, j = '[0:N//2:2]', skip_if_invalid=True)
    Syn_CDI_avant.w = np.eye[1, N+1//2]

    # CDI --> arriere
    Syn_CDI_arriere = Synapses(CDI, arriere, on_pre='v_post = th')
    Syn_CDI_arriere.connect(i = 0, j = '[0:N//2:2]', skip_if_invalid=True)
    Syn_CDI_arriere.w = np.eye[1, N+1//2]

    # CGP --> avant
    Syn_CGP_avant = Synapses(CGP, avant, on_pre='v_post = th')
    Syn_CGP_avant.connect(i = 0, j = '[1:N//2:2]', skip_if_invalid=True)
    Syn_CGP_avant.w = np.eye[1, N//2]

    # CGP --> arriere
    Syn_CGP_arriere = Synapses(CGP, arriere, on_pre='v_post = th')
    Syn_CGP_arriere.connect(i = 0, j = '[1:N//2:2]', skip_if_invalid=True)
    Syn_CGP_avant.w = np.eye[1, N//2]

    # CDP --> avant
    Syn_CDP_avant = Synapses(CDP, avant, on_pre='v_post = th')
    Syn_CDP_avant.connect(i = 0, j = '[1:N//2:2]', skip_if_invalid=True)
    Syn_CDP_avant.w = np.eye[1, N//2]

    # CDP --> arriere
    Syn_CDP_arriere = Synapses(CDP, arriere, on_pre='v_post = th')
    Syn_CDP_arriere.connect(i = 0, j = '[1:N//2:2]', skip_if_invalid=True)
    Syn_CDP_arriere.w = np.eye[1, N//2]


Declencheur, G, S_declenche, S_oscil = oscillateur(N, eqs, vitesse)
up, avant, arriere, commandes, CDI, CDP, CGI, CGP = mouvement_pattes(N, eqs)
A1, A2, DP, DI, GP, GI= neuron_test(N, eqs)
connexion_synaptiques(G, A1, A2, DI, DP, GI, GP, up, avant, arriere, CDI, CDP, CGI, CGP)

run(duration)

# Graphiques
def Graphs(up, avant, arriere):
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

Graphs(up, avant, arriere)



