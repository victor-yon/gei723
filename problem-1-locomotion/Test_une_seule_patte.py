from brian2 import *
import numpy as np
import matplotlib

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

# Oscillateur

Declencheur = NeuronGroup(1, eqs, threshold='t == 5*ms', reset='v = 0', method=methode)
G = NeuronGroup(2, eqs, threshold='v>=th', reset='v = 0', method=methode)
G.th = 0.8
Declencheur.I = [2]
G.I = [0, 0]
Declencheur.tau = [10] * ms
G.tau = 10/vitesse*ms

# Déclenchement de la marche
S_declenche = Synapses(Declencheur, G, on_pre='I_post =2; I_pre = 0')
S_declenche.connect(i=0, j=0)

# Oscillation
S_oscil = Synapses(G, G, on_pre='v_post += 0.2; I_pre = 0; I_post = 2')  # on_post = '''I_pre = 0; I_post = 2'''
S_oscil.connect(i=0, j=1)
S_oscil.connect(i=1, j=0)
S_oscil.delay = G.th * G.tau * 1.5  # /2

State_G = StateMonitor(G, 'v', record = True)

# up. Doit lancer une décharge qui met
#up = NeuronGroup(6, eqs, threshold='v >= th', reset='v = 0', method=methode)

# # Avant lance une décharge à chaque spike de B
avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
avant.th = 0.8
avant.tau = 10*ms
#
# # Arrière lance des décharges sur l'autre couleur par rapport à avant
# arriere = NeuronGroup(2 * N, eqs, threshold='v >= th', reset='v = 0', method='method')
#
# # Créer compteur Droite, gauche, paire, impaire
# # Droite impaire
# CDI_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# CDI_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# # Droite paire
# CDP_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# CDP_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# # Gauche impaire
CGI_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method=methode)
CGI_avant.th = 0.8
CGI_avant.tau = 10/vitesse*ms
# CGI_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# # Gauche paire
# CGP_avant = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')
# CGP_arriere = NeuronGroup(1, eqs, threshold='v >= th', reset='v = 0', method='euler')

# Vers les neurones commande
# GI_avant --> avant
S_G_CGI = Synapses(G, CGI_avant, on_pre = 'v_post = th')# ; I_post = 2')
S_G_CGI.connect(i = 0, j = 0)

Syn_CGI_avant = Synapses(CGI_avant, avant, on_pre='v_post = th')
Syn_CGI_avant.connect(i=0, j=0)
# CGI_arriere --> arriere
# Syn_CGI_arriere = Synapses(CGI_arriere, arriere, on_pre = 'v_post += th')
# Syn_CGI_arriere.connect(i = 0, j = np.arange(0, (N+1)//2, 2))
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

# M_up = StateMonitor(up, 'v', record=True)
# spike_up = SpikeMonitor(up)
#
M_avant = StateMonitor(avant, 'v', record=True)
spike_avant = SpikeMonitor(avant)
M_CGI = StateMonitor(CGI_avant, 'v', record=True)
spike_CGI = SpikeMonitor(CGI_avant)
#
# M_arriere = StateMonitor(arriere, 'v', record=True)
# spike_arriere = SpikeMonitor(arriere)

# plot(spike_up.t / ms, spike_up.i, '.k')
# plot(spike_avant.t/ms, spike_avant.i, '.b')
# plot(spike_arriere.t/ms, spike_avant.i, '.g')

xlabel('instant de décharge')
ylabel('indice neurone')
# legend()
# show()



run(duration)
print(spike_avant.t)

#plot(State_G.t/ms, State_G[1].v)
# show()
# plot(M_CGI.t/ms, M_CGI.v[0])
# plot(M_avant.t/ms, M_avant.v[0])
scatter(spike_avant.t/ms, spike_avant.i)
show()