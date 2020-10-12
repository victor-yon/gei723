from brian2 import *


def cpg_nn(back_sensor_value):
    # 3 Neurones : 1 Pour déclencher le signal, 2 qui oscillent
    # Convert the sensor input [0,1] into speed [1,5]
    speed = 1 + (back_sensor_value * 4)

    eqs = '''
    dv/dt = I/tau : 1 
    I : 1
    tau : second
    th : 1
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
    # La variable tau défini la fréquence de déclenchement, elle doit varier avec la vitesse
    core_nn.tau = 10 / speed * ms

    # ================ Trigger to Core ================
    # Déclenchement de l'oscillateur
    # Le fait de mettre I_pre a 0 ici empechera peut être de relancer la simultation a la suite
    syn_trigger_core = Synapses(trigger, core_nn, on_pre='I_post =2; I_pre = 0')
    syn_trigger_core.connect(i=0, j=0)

    # ================== Core to Core =================
    syn_core = Synapses(core_nn, core_nn, on_pre='v_post += 0.2; I_pre = 0; I_post = 2')
    syn_core.connect(i=0, j=1)
    syn_core.connect(i=1, j=0)
    syn_core.delay = core_nn.th * core_nn.tau * 1.5

    return trigger, core_nn, syn_trigger_core, syn_core


def monitor_cpg(core_nn):
    return StateMonitor(core_nn, 'v', record=True)


def plot_monitor_cpg(monitor, back_sensor_input):
    plot(monitor.t / ms, monitor.v[0], label='Neuron 1')
    plot(monitor.t / ms, monitor.v[1], label='Neuron 2')
    xlabel('Time (ms)')
    ylabel('v')
    title(f"CPG - Capteur bas à {back_sensor_input}A")
    legend()
    show()
