import matplotlib.pyplot as plt
from brian2 import NeuronGroup, Synapses, SpikeMonitor

FORWARD = 0
BACKWARD = 1
UP = 2
DOWN = 3


def leg_nn(cpg_core_nn, group):
    motors = NeuronGroup(4, 'v : 1', threshold='v > 0.8', reset='v = 0', method='exact')

    # ================= CPG to Motors =================
    syn_cpg_motor = Synapses(cpg_core_nn, motors, on_pre='v_post += 1')
    syn_cpg_motor.connect(i=group, j=FORWARD)
    syn_cpg_motor.connect(i=1 - group, j=BACKWARD)
    syn_cpg_motor.connect(i=group, j=UP)
    syn_cpg_motor.connect(i=1 - group, j=DOWN)

    return motors, syn_cpg_motor


def monitor_leg(motors):
    return SpikeMonitor(motors)


def plot_monitor_leg(monitor):
    print(list(zip(monitor.i, monitor.t)))

    spike_time_forward = [t for i, t in zip(monitor.i, monitor.t) if i == FORWARD]
    spike_time_backward = [t for i, t in zip(monitor.i, monitor.t) if i == BACKWARD]
    spike_time_up = [t for i, t in zip(monitor.i, monitor.t) if i == UP]
    spike_time_down = [t for i, t in zip(monitor.i, monitor.t) if i == DOWN]

    plt.scatter(spike_time_forward, [FORWARD] * len(spike_time_forward), label='FORWARD')
    plt.scatter(spike_time_backward, [BACKWARD] * len(spike_time_backward), label='BACKWARD')
    plt.scatter(spike_time_up, [UP] * len(spike_time_up), label='UP')
    plt.scatter(spike_time_down, [DOWN] * len(spike_time_down), label='DOWN')

    plt.xlabel('Instant de décharge')
    plt.ylabel('Indice du neurone')
    plt.title('Neurones moteurs d\'une patte')
    plt.legend()
    plt.show()
