import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

N = 3

vitesse = 1
droite = 1
gauche = 1

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
eqs_core = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
th : 1
'''

duration = 200*ms
UP = 0
DOWN = 1
AVANT = 2
ARRIERE = 3

SENSOR_INPUT = 0
BACKWARD_THRESHOLD = 0.4

methode = 'euler'

def oscillateur(eqs, vitesse, methode):

    Declencheur = NeuronGroup(1, eqs, threshold='t == 5*ms', reset='v = 0', method=methode)
    G = NeuronGroup(2, eqs, threshold='v>=th', reset='v = 0', method=methode)
    G.th = 0.8
    Declencheur.I = [2]
    G.I = [0, 0]
    Declencheur.tau = 200 * ms
    G.tau = 31.25/vitesse*ms

    # Déclenchement de la marche
    S_declenche = Synapses(Declencheur, G, on_pre='I_post =2*vitesse; I_pre = 0')  # Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
    S_declenche.connect(i=0, j=0)

    # Oscillation
    S_oscil = Synapses(G, G, on_pre='v_post += 0.2', on_post='I_pre = 0; I_post = 2*vitesse')  # on_post = '''I_pre = 0; I_post = 2'''
    S_oscil.connect(i=0, j=1)
    S_oscil.connect(i=1, j=0)
    S_oscil.delay = (G.th * G.tau * 1 / (2 * vitesse)) * (36 / 32)  # /2

    return Declencheur, G, S_declenche, S_oscil

# le core contient 4 éléments, 2 pour up et 2 pour down.
def build_ground_contact_nn():
    # ======== Core Network ========
    # 2 LIF neuron
    core = NeuronGroup(4, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core.I = [-SENSOR_INPUT, SENSOR_INPUT, -SENSOR_INPUT, SENSOR_INPUT]
    core.tau = 5*ms
    core.th = [BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD, BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD]

    return core

# Le groupe direction créé les groupes qui modulent le nombre de spike qui se rendent au neurones moteurs
def groupe_direction(CPG, eqs, vitesse, motors, up_down,methode):
    GI = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    GI.th = 0.8
    GI.tau = 20/vitesse*ms
    GP = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    GP.th = 0.8
    GP.tau = 20/vitesse*ms
    DI = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    DI.th = 0.8
    DI.tau = 20/vitesse*ms
    DP = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    DP.th = 0.8
    DP.tau = 20/vitesse*ms

    # Pour une patte, besoin de seulement deux groupes
    S_GI = Synapses(CPG, GI, on_pre = 'v_post += th/gauche')
    S_GI.connect(i = 0, j = 0)
    S_GP = Synapses(CPG, GP, on_pre = 'v_post += th/gauche')
    S_GP.connect(i = 1, j = 0)
    # S_DI = Synapses(CPG, DI, on_pre = 'v_post = th/droite')
    # S_DI.connect(i = 1, j = 0)
    # S_DP = Synapses(CPG, GI, on_pre = 'v_post = th/droite')
    # S_DP.connect(i = 0, j = 0)

    # C'est pour généraliser à N > 1 qu'il faudra connecter davantage les groupes.
    S_GI_motors = Synapses(GI, motors, on_pre = 'v_post = th')
    S_GI_motors.connect(i = 0, j = 0)
    S_GP_motors = Synapses(GP, motors, on_pre = 'v_post = th')
    S_GP_motors.connect(i = 0, j = 1)
    # S_DI_motors = Synapses(DI, motors, on_pre = 'v_post = th')
    # S_DI_motors.connect()
    # S_DP_motors = Synapses(DP, motors, on_pre = 'v_post = th')
    # S_DP_motors.connect()
    State_GI = StateMonitor(GI, 'v', record = True)
    State_GP = StateMonitor(GP, 'v', record = True)
    State_motors = StateMonitor(motors, 'v', record=True)
    spike_motors = SpikeMonitor(motors)
    # virage = NeuronGroup(4, eqs, threshold='v>= th', reset='v = 0', method=methode)
    # virage.th = 0.8
    # virage.tau = 10/vitesse*ms
    #
    # Syn_CPG_virage = Synapses(CPG, virage, on_pre = 'v_post = th') # ordre: GI GP DI DP
    # Syn_CPG_virage.connect(i = 0, j = [GI, DP])
    # Syn_CPG_virage.connect(i = 1, j = [GP, DI])


    return GI, GP, DI, DP, S_GI, S_GP, S_GI_motors, S_GP_motors, State_motors, spike_motors, State_GI, State_GP

# Créé une seule patte par appel de la fonction
def patte_avec_capteur(CPG, eqs, eqs_motor, core, methode, groupe):

    motors = NeuronGroup(2, eqs, threshold='v >= th', reset='v = 0', method=methode)
    motors.th = 0.8
    motors.tau = 10*ms
    up_down = NeuronGroup(2, eqs_motor, threshold='v > 0.1', reset='v = 0', method='exact')
    up_down.tau = 50*ms

    syn_core_up_down = Synapses(core, up_down, on_pre = 'v_post += 1')
    syn_core_up_down.connect(i = [0, 1], j = 0)
    syn_core_up_down.connect(i = [2, 3], j = 1)
    syn_cpg_core = Synapses(CPG, core, on_pre = 'v_post += 1')
    syn_cpg_core.connect(i = 0, j = [0, 3])
    syn_cpg_core.connect(i = 1, j = [1, 2])

    state = StateMonitor(motors, 'v', record=True)
    spike = SpikeMonitor(motors)
    spike_up_down = SpikeMonitor(up_down)

    return state, spike, motors, up_down, spike_up_down, syn_cpg_core, syn_core_up_down

def pattes_completes(N, eqs, eqs_motor, eqs_core, methode, vitesse):
    # Forward et backward
    motors = NeuronGroup(2*N, eqs, threshold='v >= th', reset='v = 0', method=methode)
    motors.th = 0.8
    motors.tau = 10*ms
    up_down = NeuronGroup(2*N, eqs_motor, threshold='v > 0.1', reset='v = 0', method='exact')
    up_down.tau = 50*ms

    # groupe inverse sens de marche
    core = NeuronGroup(4, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core.I = [-SENSOR_INPUT, SENSOR_INPUT, -SENSOR_INPUT, SENSOR_INPUT]
    core.tau = 5*ms
    core.th = [BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD, BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD]

    # groupe direction
    GI = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    GI.th = 0.8
    GI.tau = 20/vitesse*ms
    GP = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    GP.th = 0.8
    GP.tau = 20/vitesse*ms
    DI = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    DI.th = 0.8
    DI.tau = 20/vitesse*ms
    DP = NeuronGroup(1, eqs, threshold='v>= th', reset='v = 0', method=methode)
    DP.th = 0.8
    DP.tau = 20/vitesse*ms

    # connexions entre les up down
    syn_core_up_down = Synapses(core, up_down, on_pre = 'v_post += 1')
    syn_core_up_down.connect(i = [0, 1], j = np.arange(0, 2*N, 2))
    syn_core_up_down.connect(i = [2, 3], j = np.arange(1, 2*N, 2))
    syn_cpg_core = Synapses(CPG, core, on_pre = 'v_post += 1')
    syn_cpg_core.connect(i = 0, j = [0, 3])
    syn_cpg_core.connect(i = 1, j = [1, 2])

    # connexions entre le CPG et les groupes de direction
    S_GI = Synapses(CPG, GI, on_pre = 'v_post += th/gauche')
    S_GI.connect(i = 0, j = 0)
    S_GP = Synapses(CPG, GP, on_pre = 'v_post += th/gauche')
    S_GP.connect(i = 1, j = 0)
    S_DI = Synapses(CPG, DI, on_pre = 'v_post = th/droite')
    S_DI.connect(i = 1, j = 0)
    S_DP = Synapses(CPG, GI, on_pre = 'v_post = th/droite')
    S_DP.connect(i = 0, j = 0)

    # C'est pour généraliser à N > 1 qu'il faudra connecter davantage les groupes.
    S_GI_motors = Synapses(GI, motors, on_pre = 'v_post = th')
    S_GI_motors.connect(i = 0, j = np.arange(0, N, 2))
    S_GP_motors = Synapses(GP, motors, on_pre = 'v_post = th')
    S_GP_motors.connect(i = 0, j = np.arange(1, N-1, 2))
    S_DI_motors = Synapses(DI, motors, on_pre = 'v_post = th')
    S_DI_motors.connect(i = 0, j = np.arange(N, 2*N, 2))
    S_DP_motors = Synapses(DP, motors, on_pre = 'v_post = th')
    S_DP_motors.connect(i = 0, j = np.arange(N+1, 2*N, 2))
    # State_GI = StateMonitor(GI, 'v', record = True)
    # State_GP = StateMonitor(GP, 'v', record = True)
    state_motors = StateMonitor(motors, 'v', record=True)
    spike_motors = SpikeMonitor(motors)

    return spike_motors, state_motors,


def graph(spike, spike_up_down, label):
    spike_up_t = np.array([x for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == UP])
    spike_up_i = np.array([y for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == UP]) + 2

    # print(spike_up_i)
    spike_down_t = np.array([x for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == DOWN])
    spike_down_i = np.array([y for x, y in zip(spike_up_down.t / ms, spike_up_down.i) if y == DOWN]) + 2

    spike_avant_t = [x for x, y in zip(spike.t / ms, spike.i) if y == 0]
    spike_avant_i = [y for x, y in zip(spike.t / ms, spike.i) if y == 0]

    spike_arriere_t = [x for x, y in zip(spike.t / ms, spike.i) if y == 1]
    spike_arriere_i = [y for x, y in zip(spike.t / ms, spike.i) if y == 1]

    fig1 = plt.figure(1)
    scatter(spike_up_t, spike_up_i, label='UP')
    scatter(spike_down_t, spike_down_i, label='DOWN')
    scatter(spike_avant_t, spike_avant_i, label='FRONT')
    scatter(spike_arriere_t, spike_arriere_i, label='BACK')
    #axis.Axis.set_major_formatter(FormatStrFormatter('%.s'))

    xlabel('instant de décharge (ms)')
    ylabel('indice du neurone')
    title(label)
    legend()
    fig1.show()

    # plt.savefig('figure.png')

if __name__ == '__main__':
    start_scope()
    Declencheur, CPG, S_declenche, S_oscil = oscillateur(eqs, vitesse, methode)
    core = build_ground_contact_nn()

    state, spike, motors, up_down, spike_up_down, syn_cpg_core, syn_core_up_down = patte_avec_capteur(CPG, eqs, eqs_motor, core, methode, 0)
    GI, GP, DI, DP, S_GI, S_GP, S_GI_motors, S_GP_motors, State_motors, spike_motors, State_GI, State_GP = groupe_direction(CPG, eqs, vitesse, motors, up_down, methode)
    run(duration)

    graph(spike, spike_up_down, 'patte droite impaire à plus basse vitesse en tournant à gauche')
