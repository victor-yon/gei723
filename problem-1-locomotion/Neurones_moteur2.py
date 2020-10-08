from brian2 import *
import numpy as np
from matplotlib import pyplot as plt

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
duration = 50*ms
methode = 'euler'
#def oscillateur(eqs, vitesse, methode):

Declencheur = NeuronGroup(1, eqs, threshold='t == 5*ms', reset='v = 0', method=methode)
G = NeuronGroup(2, eqs, threshold='v>=th', reset='v = 0', method=methode)
G.th = 0.8
Declencheur.I = [2]
G.I = [0, 0]
Declencheur.tau = 10 * ms
G.tau = 10/vitesse*ms

# Déclenchement de la marche
S_declenche = Synapses(Declencheur, G, on_pre='I_post =2; I_pre = 0')
S_declenche.connect(i=0, j=0)

# Oscillation
S_oscil = Synapses(G, G, on_pre='v_post += 0.2; I_pre = 0; I_post = 2')
S_oscil.connect(i=0, j=1)
S_oscil.connect(i=1, j=0)
S_oscil.delay = G.th * G.tau * 1.5

    #return Declencheur, G, S_declenche, S_oscil

#def mouvement_pattes(N, eqs, vitesse, methode):

# up
up = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method=methode)
up.th = 0.8
up.tau = 10*ms

# Avant lance une décharge à chaque spike de B
avant = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method=methode)
avant.th = 0.8
avant.tau = 10*ms

# Arrière lance des décharges sur l'autre couleur par rapport à avant
arriere = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method=methode)
arriere.th = 0.8
arriere.tau = 10*ms

# Créer compteur Droite, gauche, paire, impaire
# Droite impaire
CDI_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CDI_avant.th = 0.8
CDI_avant.tau = 10*ms

CDI_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CDI_arriere.th = 0.8
CDI_arriere.tau = 10*ms

# # Droite paire
CDP_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CDP_avant.th = 0.8
CDP_avant.tau = 10*ms

CDP_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CDP_arriere.th = 0.8
CDP_arriere.tau = 10*ms

# # Gauche impaire
CGI_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CGI_avant.th = 0.8
CGI_avant.tau = 10*ms

CGI_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CGI_arriere.th = 0.8
CGI_arriere.tau = 10*ms

# # Gauche paire
CGP_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CGP_avant.th = 0.8
CGP_avant.tau = 10*ms

CGP_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CGP_arriere.th = 0.8
CGP_arriere.tau = 10*ms

    #return up, avant, arriere, CDI_avant, CDI_arriere, CDP_avant, CDP_arriere, CGI_avant, CGI_arriere, CGP_avant, CGP_arriere

#def connexion_synaptiques(G, up, avant, arriere, CDI_avant, CDI_arriere, CDP_avant, CDP_arriere, CGI_avant, CGI_arriere, CGP_avant, CGP_arriere):

Syn_1_CGI_avant = Synapses(G, CGI_avant, on_pre = 'v_post = th')
Syn_1_CGI_avant.connect(i = 0, j = 0)

Syn_2_CGI_arriere = Synapses(G, CGI_arriere, on_pre = 'v_post = th')
Syn_2_CGI_arriere.connect(i = 1, j = 0)

# Vers les neurones commande
# GI_avant --> avant
Syn_CGI_avant = Synapses(CGI_avant, avant, on_pre='v_post = th')
Syn_CGI_avant.connect(i = 0, j = np.arange(0, (N+1), 2))
# CGI_arriere --> arriere
Syn_CGI_arriere = Synapses(CGI_arriere, arriere, on_pre = 'v_post += th')
Syn_CGI_arriere.connect(i = 0, j = np.arange(0, (N+1), 2))
#
# # GP_avant
# Syn_CGP_avant = Synapses(CGP_avant, avant, on_pre = 'v_post += th')
# Syn_CGP_avant.connect(i = 0, j = np.arange(0, (N+1)//2, 2))
#
# # GP_arriere
# Syn_CGP_arriere = Synapses(CGP_arriere, arriere, on_pre = 'v_post += th')
# Syn_CGP_arriere.connect(i = 0, j = np.arange(1, (N+1)//2, 2))
#
# # DI_avant
# Syn_CDI_avant = Synapses(CDI_avant, avant, on_pre = 'v_post += th')
# Syn_CDI_avant.connect(i = 0, j = np.arange(0, (N+1)//2, 2))
#
# # DI_arriere
# Syn_CDI_arriere = Synapses(CDI_arriere, arriere, on_pre = 'v_post += th')
# Syn_CDI_arriere.connect(i = 0, j = np.arange(0, (N+1)//2, 2))
#
# # DP_avant
# Syn_CDP_avant = Synapses(CDP_avant, avant, on_pre = 'v_post += th')
# Syn_CDP_avant.connect(i = 0, j = np.arange(1, (N+1)//2, 2))
#
# # DP_arriere
# Syn_CDP_arriere = Synapses(CDP_arriere, arriere, on_pre = 'v_post += th')
# Syn_CDP_arriere.connect(i = 0, j = np.arange(1, (N+1)//2, 2))
#
# Syn_1_CGI_avant = Synapses(G, CGI_avant, on_pre = 'v_post += th')
# Syn_1_CGI_avant.connect(i = 0, j = 0)
#
# Syn_2_CGI_arriere = Synapses(G, CGI_arriere, on_pre = 'v_post += th')
# Syn_2_CGI_arriere.connect(i = 1, j = 0)
#
# Syn_1_CGP_arriere = Synapses(G, arriere, on_pre = 'v_post += th')
# Syn_1_CGP_arriere.connect(i = 0, j = 0)
#
# Syn_2_CGP_avant = Synapses(G, arriere, on_pre = 'v_post += th')
# Syn_2_CGP_avant.connect(i = 1, j = 0)
#
# Syn_1_CDP_arriere = Synapses(G, on_pre = 'v_post += th')
# Syn_1_CDP_arriere.connect(i = 0, j = 0)
#
# Syn_2_CDP_avant = Synapses(G, arriere, on_pre = 'v_post += th')
# Syn_2_CDP_avant.connect(i = 1, j = 0)
#
# Syn_1_CDP_avant = Synapses(G, arriere, on_pre = 'v_post += th')
# Syn_1_CDP_avant.connect(i = 0, j = 0)
#
# Syn_2_CDP_arriere = Synapses(G, arriere, on_pre = 'v_post += th')
# Syn_2_CDP_arriere.connect(i = 1, j = 0)
    #return
M_G = StateMonitor(G, 'v', record = True)
Dec = StateMonitor(Declencheur, 'v', record = True)
M_avant = StateMonitor(avant, 'v', record=True)
spike_avant = SpikeMonitor(avant)
spike_arriere = SpikeMonitor(arriere)


run(duration)

# Graphiques
#def Graphs(up, avant, arriere, G):
# M_up = StateMonitor(up, 'v', record=True)
# spike_up = SpikeMonitor(up)



# M_arriere = StateMonitor(arriere, 'v', record=True)


# plot(spike_up.t/ms, spike_up.i, '.k')
#plot(M_avant.t/ms, M_avant.v[0])
#plt.hold(True)
plt.scatter(spike_avant.t/ms, spike_avant.i, c='r')
plt.scatter(spike_arriere.t/ms, (spike_arriere.i+1), c='g')
#plot(Dec.t/ms, Dec.v[0])
#plot(M_G.t/ms, M_G[1].v)

xlabel('instant de décharge')
ylabel('indice neurone')
#legend()
show()
#print(spike_avant.i)
# print(spike_avant.i)
#print(spike_avant.t)
#print(M_avant)

#Declencheur, G, S_declenche, S_oscil = oscillateur(eqs, vitesse, methode)
#up, avant, arriere, CDI_avant, CDI_arriere, CDP_avant, CDP_arriere, CGI_avant, CGI_arriere, CGP_avant, CGP_arriere = mouvement_pattes(N, eqs, vitesse, methode)

#connexion_synaptiques(G, up, avant, arriere, CDI_avant, CDI_arriere, CDP_avant, CDP_arriere, CGI_avant, CGI_arriere, CGP_avant, CGP_arriere)
# run(duration)

# M_G = StateMonitor(G, 'v', record=True)
# plot(M_G.t/ms, M_G.v[0])
# show()

#Graphs(up, avant, arriere, G)