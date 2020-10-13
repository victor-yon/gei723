from typing import List

import matplotlib.pyplot as plt
from brian2 import NeuronGroup, Synapses, SpikeMonitor, ms

FORWARD = 0
BACKWARD = 1
UP = 2
DOWN = 3


def legs_nn(nb, input_forward, input_backward, input_up, input_down):
    """
    Create 4 motor neurons for each leg of this category.

    :param nb: The number of leg to create
    :param input_forward: The neurone that will output the forward signal.
    :param input_backward: The neurone that will output the backward signal.
    :param input_up: The neurone that will output the up signal.
    :param input_down: The neurone that will output the down signal.
    :return: The different part of the network
    """
    motors = NeuronGroup(4 * nb, 'v : 1', threshold='v > 0.8', reset='v = 0', method='exact')

    # ============== Direction to Motors ==============
    syn_cpg_motor_a = Synapses(input_forward, motors, on_pre='v_post += 1')
    syn_cpg_motor_a.connect(i=0, j=[x for x in range(4 * nb) if x % 4 == FORWARD])
    syn_cpg_motor_b = Synapses(input_backward, motors, on_pre='v_post += 1')
    syn_cpg_motor_b.connect(i=0, j=[x for x in range(4 * nb) if x % 4 == BACKWARD])

    # ============ Ground Contact to Motors ===========
    syn_up_motor = Synapses(input_up, motors, on_pre='v_post += 1')
    syn_up_motor.connect(i=0, j=[x for x in range(4 * nb) if x % 4 == UP])
    syn_down_motor = Synapses(input_down, motors, on_pre='v_post += 1')
    syn_down_motor.connect(i=0, j=[x for x in range(4 * nb) if x % 4 == DOWN])

    return motors, syn_cpg_motor_a, syn_cpg_motor_b, syn_up_motor, syn_down_motor


def monitor_legs(legs_nn):
    motors, _, _, _, _ = legs_nn
    # Record only the 4 first neurons which is one leg
    return SpikeMonitor(motors, record=range(4))


def plot_monitor_legs(monitors: list, leg_names: List[str], time_offset: float = 0):
    nb = len(monitors)
    fig, subplots = plt.subplots(nb, figsize=(8, nb * 1.5 + 2))

    for m, s, n in zip(monitors, subplots, leg_names):
        spike_time_forward = [t for i, t in zip(m.i, m.t / ms) if i == FORWARD and t >= time_offset]
        spike_time_backward = [t for i, t in zip(m.i, m.t / ms) if i == BACKWARD and t >= time_offset]
        spike_time_up = [t for i, t in zip(m.i, m.t / ms) if i == UP and t >= time_offset]
        spike_time_down = [t for i, t in zip(m.i, m.t / ms) if i == DOWN and t >= time_offset]

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
