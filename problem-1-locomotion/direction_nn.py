from brian2 import *

from cpg_nn import SPIKE_TRAIN_SIZE

LEFT = 0
RIGHT = 1


def build_direction_nn(cpg_core, group, side_sensor_value):
    # ==================== Core Main ==================
    # 1 IF neuron
    eqs_core = '''
    dv/dt = -v/tau : 1
    tau : second
    th : 1
    '''

    core_main = NeuronGroup(1, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core_main.tau = 10 * ms
    core_main.th = 0.8

    # =================== Core Inhib ==================
    # 1 LIF neuron
    eqs_core_inhib = '''
    dv/dt = (I-v)/tau : 1
    v_bis : 1
    tau : second
    I : 1
    th : 1
    step : 1
    '''

    core_inhib = NeuronGroup(1, eqs_core_inhib, threshold='(v + v_bis) > th', reset='v_bis -= step', method='exact')
    core_inhib.tau = 10 * ms
    core_inhib.th = 1
    core_inhib.I = side_sensor_value
    core_inhib.step = 1 / SPIKE_TRAIN_SIZE  # This is the v_bis increase step (use for reset)

    # ================= Main to Inhib =================
    syn_main_inhib = Synapses(core_inhib, core_main, on_pre='v_post -= 1')
    syn_main_inhib.connect(i=0, j=0)

    # ================ CPG to Direction ===============
    syn_cpg_core = Synapses(cpg_core, core_main, on_pre='v_post += 1')
    syn_cpg_core.connect(i=group, j=0)
    # Delay required because if the 2 pulse (-1 and +1) arrived at the same time only one is take into account
    syn_cpg_core.delay = 2 * ms

    # ================== CPG to Inhib =================
    syn_cpg_inhib = Synapses(cpg_core, core_inhib, 'w : 1', on_pre='v_bis_post += w')
    syn_cpg_inhib.connect(i=group, j=0)
    syn_cpg_inhib.w = 1 / SPIKE_TRAIN_SIZE

    # =============== CPG to Inhib Rest ===============
    syn_cpg_inhib_rest = Synapses(cpg_core, core_inhib, on_pre='v_bis_post = 0')
    syn_cpg_inhib_rest.connect(i=1 - group, j=0)

    return core_main, core_inhib, syn_main_inhib, syn_cpg_core, syn_cpg_inhib, syn_cpg_inhib_rest


def monitor_direction(direction_nn):
    core_main, core_inhib, _, _, _, _ = direction_nn

    state_mon_main = StateMonitor(core_main, 'v', record=True)
    state_mon_inhib = StateMonitor(core_inhib, ('v', 'v_bis'), record=True)
    return state_mon_main, state_mon_inhib


def plot_monitor_direction(m_direction, side, side_sensor_value):
    state_mon_main, state_mon_inhib = m_direction

    plot(state_mon_main.t / ms, state_mon_main.v[0], color='blue', label='Sortie')
    plot(state_mon_inhib.t / ms, state_mon_inhib.v[0] + state_mon_inhib.v_bis[0], color='red', label='Inhibition')
    xlabel('Time (ms)')
    ylabel('v')
    title(f"Module de rotation {'gauche' if side == LEFT else 'droit'} - Capteur latéral à {side_sensor_value}A")
    legend()
    show()
