from math import sqrt, ceil, floor
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, Synapses, SpikeGeneratorGroup, units, Network


def stdp_shape(eqs_stdp: str, on_pre: str, on_post: str, neuron_threshold: int = 1, neuron_variable: str = '',
               min_delta: int = -50, max_delta: int = 50, nb_measurement: int = 50, plot_title: str = 'STDP shape',
               verbose: bool = False) -> None:
    """
    Draw the shape of a STDP equation (evolution of Δ weights over time).

    :param eqs_stdp: The STDP equation
    :param on_pre: The action before the spike
    :param on_post: The action after the spike
    :param neuron_threshold: The neuron threshold (should be compatible with the on_pre equation)
    :param neuron_variable: 
    :param min_delta: The minimal time difference between post and pre synaptic spike
    :param max_delta: The maximal time difference between post and pre synaptic spike
    :param nb_measurement: The number of measurement point between the delta min and max
    :param plot_title: The title of the plot
    :param verbose: If true output text value for each measurement
    """

    # Most simple neurons equation for simulation
    neurons = NeuronGroup(2, model=f'v : 1\n{neuron_variable}', threshold=f'v>={neuron_threshold}', reset='v=0',
                          method='euler')
    neurons.v = 0

    # One synapse between the neurons, using the equations to evaluate
    synapses = Synapses(neurons, neurons, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')
    synapses.connect(i=0, j=1)

    # Spike generator as input for the 2 neurons
    input_generator = SpikeGeneratorGroup(2, [], [] * units.ms)
    # The generator force the neurons to spike
    input_generator_synapses = Synapses(input_generator, neurons, on_pre=f'v_post += {neuron_threshold}')
    input_generator_synapses.connect(i=[0, 1], j=[0, 1])

    # Store the time and weights difference for each step in vectors
    delta_t = np.linspace(min_delta, max_delta, num=nb_measurement)
    delta_w = np.zeros(delta_t.size)

    # Create the network
    nn = Network(neurons, synapses, input_generator, input_generator_synapses)
    # Save the network
    nn.store()

    for i in range(delta_t.size):
        dt = delta_t[i]

        # Load the initial network
        nn.restore()

        # We force the neurons to spike with the target delta t
        # Depending of the time one will trigger before the other
        if dt < 0:
            input_generator.set_spikes([0, 1], [-dt, 0] * units.ms)
        else:
            input_generator.set_spikes([0, 1], [0, dt] * units.ms)

        # Run the simulation for this delta t
        nn.run((np.abs(dt) + 1) * units.ms)
        # Since the weight was 0 at the beginning of the run, Δ weight = current weight
        delta_w[i] = synapses.w[0]

        if verbose:
            print(f'Δt {dt:5.3} - Δw {synapses.w[0]:5.3}')

    # Print the plot
    plt.figure(figsize=(8, 5))
    plt.axhline(y=0, color='grey')
    plt.axvline(x=0, color='grey')
    plt.plot(delta_t, delta_w, linestyle='-', marker='o')
    plt.title(plot_title)
    plt.xlabel('Δ time (ms)')
    plt.ylabel('Δ weights')
    plt.show()


def img_show(images, title: str = '', ground_truth: List[str] = None, prediction_success: List[bool] = None) -> None:
    """
    Show a set of image.

    :param images: A list of torch tensor images
    :param title: The title of the figure
    :param ground_truth: The list of ground truth, if defined is show for each images
    :param prediction_success: The list of prediction success, if defined the ground truth is printed in green for good
    prediction or red for wrong classification
    """
    fig = plt.figure()
    fig.suptitle(title)

    # Compute the size of the image array
    x = sqrt(len(images))
    column = ceil(x)
    line = floor(x)

    # Show images
    for i, image in enumerate(images):
        sub = fig.add_subplot(column, line, i + 1)
        sub.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        sub.axis('off')

        if ground_truth is not None:
            sub.title.set_text(ground_truth[i])

            if prediction_success is not None:
                if prediction_success[i]:
                    sub.title.set_color('g')
                else:
                    sub.title.set_color('r')
    plt.show()
