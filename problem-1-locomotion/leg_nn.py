from typing import List

import matplotlib.pyplot as plt
from brian2 import NeuronGroup, Synapses, SpikeMonitor, ms

FORWARD = 0
BACKWARD = 1
UP = 2
DOWN = 3


def leg_nn(direction_a, direction_b, input_up, input_down):
    motors = NeuronGroup(4, 'v : 1', threshold='v > 0.8', reset='v = 0', method='exact')

    # ============== Direction to Motors ==============
    syn_cpg_motor_a = Synapses(direction_a, motors, on_pre='v_post += 1')
    syn_cpg_motor_a.connect(i=0, j=FORWARD)
    syn_cpg_motor_b = Synapses(direction_b, motors, on_pre='v_post += 1')
    syn_cpg_motor_b.connect(i=0, j=BACKWARD)

    # ============ Ground Contact to Motors ===========
    syn_up_motor = Synapses(input_up, motors, on_pre='v_post += 1')
    syn_up_motor.connect(i=0, j=UP)
    syn_down_motor = Synapses(input_down, motors, on_pre='v_post += 1')
    syn_down_motor.connect(i=0, j=DOWN)

    return motors, syn_cpg_motor_a, syn_cpg_motor_b, syn_up_motor, syn_down_motor


def monitor_leg(motors):
    return SpikeMonitor(motors)


def plot_monitor_legs(monitors: list, leg_names: List[str], time_offest: float = 0):
    nb = len(monitors)
    fig, subplots = plt.subplots(nb, figsize=(8, nb * 1.5 + 2))

    for m, s, n in zip(monitors, subplots, leg_names):
        spike_time_forward = [t for i, t in zip(m.i, m.t / ms) if i == FORWARD and t >= time_offest]
        spike_time_backward = [t for i, t in zip(m.i, m.t / ms) if i == BACKWARD and t >= time_offest]
        spike_time_up = [t for i, t in zip(m.i, m.t / ms) if i == UP and t >= time_offest]
        spike_time_down = [t for i, t in zip(m.i, m.t / ms) if i == DOWN and t >= time_offest]

        s.scatter(spike_time_forward, [1] * len(spike_time_forward), label='Avance patte', marker='>', color='tab:blue')
        s.scatter(spike_time_backward, [1] * len(spike_time_backward), label='Recule patte', marker='<',
                  color='tab:orange')
        s.scatter(spike_time_up, [0] * len(spike_time_up), label='Lève patte', marker='^', color='tab:green')
        s.scatter(spike_time_down, [0] * len(spike_time_down), label='Baisse patte', marker='v', color='tab:red')

        s.set_title(n)

    plt.setp(subplots, yticks=[0, 1], yticklabels=['Lever\nBaisser', 'Avancer\nReculer'], ylim=[-0.5, 1.5])
    plt.suptitle(f'Neurones moteurs des {len(monitors)} premières pattes', fontsize=14)
    plt.xlabel('Instant de décharge (ms)')
    handles, labels = subplots[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()
