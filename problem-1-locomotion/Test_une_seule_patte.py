from brian2 import *
import numpy as np
import matplotlib

N = 3
vitesse = 3
eqs = '''
dv/dt = I/tau : 1
I : 1
tau : second
th : 1
'''
duration = 50*ms
UP = 0
DOWN = 1
AVANT = 2
ARRIERE = 3

methode = 'euler'

def oscillateur(eqs, vitesse, methode):

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

    return Declencheur, G, S_declenche, S_oscil

def patte(CPG, eqs, methode):

    motors = NeuronGroup(4, eqs, threshold='v >= th', reset='v = 0', method=methode)
    motors.th = 0.8
    motors.tau = 10*ms

    S = Synapses(CPG, motors, on_pre='v_post = th')
    S.connect(i = 0, j = [UP, AVANT])
    S.connect(i = 1, j = [DOWN, ARRIERE])

    state = StateMonitor(motors, 'v', record=True)
    spike = SpikeMonitor(motors)

    return state, spike, S, motors

def graph(spike):
    spike_up_t = [x for x, y in zip(spike.t / ms, spike.i) if y == UP]
    spike_up_i = [y for x, y in zip(spike.t / ms, spike.i) if y == UP]

    spike_down_t = [x for x, y in zip(spike.t / ms, spike.i) if y == DOWN]
    spike_down_i = [y for x, y in zip(spike.t / ms, spike.i) if y == DOWN]

    spike_avant_t = [x for x, y in zip(spike.t / ms, spike.i) if y == AVANT]
    spike_avant_i = [y for x, y in zip(spike.t / ms, spike.i) if y == AVANT]

    spike_arriere_t = [x for x, y in zip(spike.t / ms, spike.i) if y == ARRIERE]
    spike_arriere_i = [y for x, y in zip(spike.t / ms, spike.i) if y == ARRIERE]

    scatter(spike_up_t, spike_up_i, label='UP')
    scatter(spike_down_t, spike_down_i, label='DOWN')
    scatter(spike_avant_t, spike_avant_i, label='FRONT')
    scatter(spike_arriere_t, spike_arriere_i, label='BACK')

    xlabel('instant de décharge')
    ylabel('indice du neurone')
    legend()
    show()
    return


if __name__ == '__main__':
    start_scope()

    Declencheur, CPG, S_declenche, S_oscil = oscillateur(eqs, vitesse, methode)

    state, spike, S, motors = patte(CPG, eqs, methode)
    run(duration)

    graph(spike)