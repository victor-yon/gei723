# commit

from brian2 import *

from pygame import keys
from pygame.locals import K_i # i pour input
import time
# from brian2 import *
# import pygame

%matplotlib inline
# import matplotlib

start_scope()
time_step = 1*ms
sortie_D = []
sortie_G = []


eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second 
'''
temps_total = 0*ms
D = NeuronGroup(1, eqs, threshold ='v>0.8', reset = 'v = 0', method = 'exact')
G = NeuronGroup(2, eqs, threshold = 'v')
S_D = Synapses(D, G, on_pre='v = I')
S_D.connect(i = 0, j = 0)
S_D.w = 1
S_D_delay = 0

S = Synapses(G, G, on_pre='v += w')
S.connect(condition='i!=j')
S.w = [0.5, 0]
S.delay = [0, 0]

M_D = StateMonitor(D, variables=True, record=True)
M_G = StateMonitor(G, variables=True, record=True)
store()
while True:
    restore()
    if keys.get_pressed() == K_i:
        D.I = [1]
    else:
        D.I = [0]
    print(G.I)
    run(time_step)
    temps_total += time_step
    store()
    sortie_D.append(M_D.v)
    sortie_G.append(M_G.v)

    time.sleep(0.01)
    if temps_total >= 100*ms:
        break

# print(sortie_G)
# Graph
plot(M_G.t/ms, sortie_D)
plot(M_G.t/ms, sortie_G[0])
plot(M_G.t/ms, sortie_G[1])
xlabel(r'$\tau$ (ms)')
ylabel('potentiel v')


# Le commentaire en bas du fichier!