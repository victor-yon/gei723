from brian2 import *

BACKWARD_THRESHOLD = 0.5


def build_ground_contact_nn(cpg, front_sensor_input, inverted):
    # ====================== Core =====================
    # 2 LIF neuron
    eqs_core = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    th : 1
    '''

    core_main = NeuronGroup(2, eqs_core, threshold='v > th', reset='v = 0', method='exact')
    core_main.I = [-front_sensor_input, front_sensor_input]
    core_main.tau = [1, 1] * ms  # Very fast leakage is required
    core_main.th = [BACKWARD_THRESHOLD, 1 + BACKWARD_THRESHOLD]

    # ===================== Output ====================
    # 1 LIF neuron
    eqs_output = '''
    dv/dt = -v/tau : 1
    tau : second
    '''

    output = NeuronGroup(1, eqs_output, threshold='v > 0.1', reset='v = 0', method='exact')
    output.tau = [50] * ms

    # ================= Core to Output ================
    syn_core_motor = Synapses(core_main, output, on_pre='v_post += 1')
    syn_core_motor.connect(i=[0, 1], j=0)

    # =================== CPG to Core =================
    syn_cpg_core = Synapses(cpg, core_main, on_pre='v_post += 1')
    idx = 1 if inverted else 0
    syn_cpg_core.connect(i=idx, j=0)
    syn_cpg_core.connect(i=1 - idx, j=1)

    return output, core_main, syn_core_motor, syn_cpg_core


def monitor_ground_contact(direction_nn):
    output, core_main, _, _ = direction_nn

    state_mon_core = StateMonitor(core_main, 'v', record=True)
    state_mon_output = StateMonitor(output, 'v', record=True)
    return state_mon_core, state_mon_output


def plot_monitor_ground_contact(m_ground_contact, front_sensor_input):
    state_mon_core, state_mon_output = m_ground_contact

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(state_mon_core.t / ms, state_mon_core.v[0], color='blue', label='Core 1')
    ax1.axhline(y=BACKWARD_THRESHOLD, color='blue', linestyle=':', label='Seuil 1')
    ax1.plot(state_mon_core.t / ms, state_mon_core.v[1], color='green', label='Core 2')
    ax1.axhline(y=1 + BACKWARD_THRESHOLD, color='green', linestyle=':', label='Seuil 2')
    ax2.plot(state_mon_output.t / ms, state_mon_output.v[0], color='red', label='Sortie')
    xlabel('Time (ms)')
    ylabel('v')
    ax1.legend()
    ax2.legend()
    suptitle(f"Module du sens de la marche - Capteur frontal à {front_sensor_input}A")
    show()
