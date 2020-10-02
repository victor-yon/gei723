# import pygame
# from pygame import key
# from pygame.locals import K_i # i pour input
# import time

from brian2 import *
import matplotlib

start_scope()
# time_step = 1*ms
# sortie_D = []
# sortie_G = []

eqs = '''
dv/dt = (I)/tau : 1
I : 1
tau : second
th : 1
'''
# temps_total = 0*ms
duration = 100*ms

# D = NeuronGroup(1, eqs, threshold='v>=0.8', reset='v = 0', method='euler')
G = NeuronGroup(2, eqs, threshold='v>=0.8', reset='v = 0', method='euler')
# D.tau = 20*ms
# D.I = 2
G.tau = [20, 20]*ms
G.I = [2, 0]

# S_D = Synapses(D, G, 'w : 1', on_pre='v_post += w')
# S_D.connect(i=0, j=0)
# S_D.w = 1
# S_D.delay = 0*ms

S = Synapses(G, G, 'w : 1',  on_pre='v_post += w')
S.connect(condition='i!=j')
S.w = [0.5, 0]
S.delay = [0, 0]*ms

# M_D = StateMonitor(D, 'v', record=True)
M_G = StateMonitor(G, 'v', record=True)

# store()
# pygame.init()
# while True:
#     restore()
#     if key.get_pressed() == K_i:
#         D.I = [1]
#     else:
#         D.I = [0]
#     print(G.I)
#     run(time_step)
#     temps_total += time_step
#     store()
#     sortie_D.append(M_D.v)
#     sortie_G.append(M_G.v)
#
#     time.sleep(0.01)
#     if temps_total >= 100*ms:
#         break

run(duration)

# print(sortie_G)
# Graph
# plot(M_G.t/ms, sortie_D)
# plot(M_G.t/ms, sortie_G[1])
# plot(M_G.t/ms, sortie_G[0])
# print(M_D.t/ms)
# print(M_D.v)

plot(M_G.t / ms, M_G.v[0])
plot(M_G.t/ms, M_G.v[0])
plot(M_G.t/ms, M_G.v[1])


xlabel(r'$\tau$ (ms)')
ylabel('potential v')
legend()
show()