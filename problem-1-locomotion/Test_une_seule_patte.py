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
eqs_motor = '''
dv/dt = -v/tau : 1
tau : second
th : 1
'''

duration = 50*ms
UP = 0
DOWN = 1
AVANT = 2
ARRIERE = 3

SENSOR_INPUT = -1.5
BACKWARD_THRESHOLD = 0.5

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

def build_ground_contact_nn():
    # ======== Core Network ========
    # 2 LIF neuron
    eqs_core = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    th : 1
    '''

    core = NeuronGroup(4, eqs_core, threshold='v >= th', reset='v = 0', method='exact')
    core.I = [-SENSOR_INPUT, SENSOR_INPUT, -SENSOR_INPUT, SENSOR_INPUT]
    core.tau = 5*ms
    core.th = [BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD, BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD]

    # ======== Ground Motor ========
    # 1 LIF neuron
    eqs_motor = '''
    dv/dt = I/tau : 1
    tau : second
    th : 1
    '''

    # motor = NeuronGroup(1, eqs_motor, threshold='v >= th', reset='v = 0', method='exact')
    # motor.tau = [50] * ms

    # ======= Core to Motor ========
    # syn_core_motor = Synapses(core, motors, on_pre='v_post = th')
    # syn_core_motor.connect(i=[0, 1], j=0)
    return core #, syn_core_motor


def patte(CPG, eqs, methode, groupe):

    motors = NeuronGroup(4, eqs, threshold='v >= th', reset='v = 0', method=methode)
    motors.th = 0.8
    motors.tau = 10*ms


    S = Synapses(CPG, motors, on_pre='v_post = th')
    S.connect(i = groupe, j = [UP, AVANT])
    S.connect(i = 1-groupe, j = [DOWN, ARRIERE])

    state = StateMonitor(motors, 'v', record=True)
    spike = SpikeMonitor(motors)

    return state, spike, S, motors

def patte_avec_capteur(CPG, eqs, eqs_motor, methode, groupe):

    motors = NeuronGroup(2, eqs, threshold='v >= th', reset='v = 0', method=methode)
    motors.th = 0.8
    motors.tau = 10*ms
    up_down = NeuronGroup(2, eqs_motor, threshold='v > 0.1', reset='v = 0', method='exact')
    up_down.tau = 50*ms

    S = Synapses(CPG, motors, on_pre='v_post = th')
    S.connect(i = groupe, j = 0) # avant
    S.connect(i = 1-groupe, j = 1) # arriere

    state = StateMonitor(motors, 'v', record=True)
    spike = SpikeMonitor(motors)
    spike_up_down = SpikeMonitor(up_down)

    return state, spike, S, motors, up_down, spike_up_down

def graph(spike, spike_up_down, label):
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
    title(label)
    legend()
    show()
    return

def graph_up_down(spike, spike_up_down, label):
    spike_up_t = np.array([x for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == UP])
    spike_up_i = np.array([y for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == UP])+2

    print(spike_up_i)
    spike_down_t = np.array([x for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == DOWN])
    spike_down_i = np.array([y for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == DOWN]) + 2

    spike_avant_t = [x for x, y in zip(spike.t / ms, spike.i) if y == 0]
    spike_avant_i = [y for x, y in zip(spike.t / ms, spike.i) if y == 0]

    spike_arriere_t = [x for x, y in zip(spike.t / ms, spike.i) if y == 1]
    spike_arriere_i = [y for x, y in zip(spike.t / ms, spike.i) if y == 1]

    scatter(spike_up_t, spike_up_i, label='UP')
    scatter(spike_down_t, spike_down_i, label='DOWN')
    scatter(spike_avant_t, spike_avant_i, label='FRONT')
    scatter(spike_arriere_t, spike_arriere_i, label='BACK')

    xlabel('instant de décharge')
    ylabel('indice du neurone')
    title(label)
    legend()
    show()


if __name__ == '__main__':
    start_scope()

    Declencheur, CPG, S_declenche, S_oscil = oscillateur(eqs, vitesse, methode)

    core = build_ground_contact_nn()

    #state, spike, S, motors = patte(CPG, eqs, methode, 0)
    # state1, spike1, S1, motors1 = patte(CPG, eqs, methode, 1)
    state, spike, S, motors, up_down, spike_up_down = patte_avec_capteur(CPG, eqs, eqs_motor, methode, 0)

    syn_core_up_down = Synapses(core, up_down, on_pre = 'v_post += 1')
    syn_core_up_down.connect(i = [0, 1], j = 0)
    syn_core_up_down.connect(i = [2, 3], j = 1)

    syn_cpg_core = Synapses(CPG, core, on_pre = 'v_post += 1')
    syn_cpg_core.connect(i = 0, j = [0, 3])
    syn_cpg_core.connect(i = 1, j = [1, 2])

    syn_CPG_motor = Synapses(CPG, motors, on_pre = 'v_post = th')
    syn_CPG_motor.connect(condition='i==j')

    # syn_core_motor = Synapses(core, motors, on_pre = 'v_post = th')
    # syn_core_motor.connect(i = [0, 1], j = UP)
    # syn_core_motor.connect(i = [2, 3], j = DOWN)

    # syn_core_motor1 = Synapses(core, motors1, on_pre = 'v_post = th')
    # syn_core_motor1.connect(i = [0, 2], j = UP)
    # syn_core_motor1.connect(i = [1, 3], j = DOWN)


    run(duration)

    graph_up_down(spike, spike_up_down, 'patte impaire')
    #graph(spike1, 'patte paire')