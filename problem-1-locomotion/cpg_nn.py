from brian2 import *

SPIKE_TRAIN_SIZE = 5
SPEED_CONSTANT = 30


def build_cpg_nn(back_sensor_value):
    # 3 Neurones : 1 Pour déclencher le signal, 2 qui oscillent
    # Convert the sensor input [0,1] into speed [0,5]
    # Add 0.1 to avoid 0
    speed = back_sensor_value * 10 + 0.1

    eqs = '''
    dv/dt = I/tau : 1 
    I : 1
    tau : second
    th : 1
    speed : 1
    '''

    # ==================== Trigger ====================
    # On définit le temps que met le neurone déclencheur à transmettre l'info à 5ms
    trigger = NeuronGroup(1, eqs, threshold='t == 5*ms', reset='v = 0', method='euler')
    trigger.I = 2
    trigger.tau = 10 * ms

    # ====================== Core =====================
    core_nn = NeuronGroup(2, eqs, threshold='v >= th', reset='v = 0', method='euler')
    core_nn.th = 0.8
    core_nn.I = [0, 0]
    core_nn.speed = speed
    # La variable tau défini la fréquence de déclenchement, elle doit varier avec la vitesse
    core_nn.tau = SPEED_CONSTANT * ms

    # ================ Trigger to Core ================
    # Déclenchement de l'oscillateur
    # Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
    syn_trigger_core = Synapses(trigger, core_nn, on_pre='I_post = speed_post; I_pre = 0')
    syn_trigger_core.connect(i=0, j=0)

    # ================== Core to Core =================
    syn_core = Synapses(core_nn, core_nn, on_pre='v_post += 0.2', on_post='I_post = speed_post; I_pre = 0')
    syn_core.connect(i=0, j=1)
    syn_core.connect(i=1, j=0)
    syn_core.delay = core_nn.th * core_nn.tau * (1 / speed) * (79 / 64)

    return core_nn, trigger, syn_trigger_core, syn_core


def monitor_cpg(cpg_nn):
    core_nn, trigger, _, _ = cpg_nn
    m_core = StateMonitor(core_nn, 'v', record=True)
    m_trigger = StateMonitor(trigger, 'v', record=True)
    return m_core, m_trigger


def plot_monitor_cpg(m_cpg, back_sensor_input):
    m_core, m_trigger = m_cpg
    plot(m_trigger.t / ms, m_trigger.v[0], label='Déclencheur', color='black')
    plot(m_core.t / ms, m_core.v[0], label='Neuron 1', color='tab:red')
    plot(m_core.t / ms, m_core.v[1], label='Neuron 2', color='tab:green')
    xlabel('Time (ms)')
    ylabel('v')
    title(f"CPG - Capteur bas à {back_sensor_input}A")
    legend()
    show()
