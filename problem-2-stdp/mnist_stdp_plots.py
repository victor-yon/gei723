"""
author: Antoine Marion et Victor Yon
date: 19/11/2020
version history: See Github
description: Définition des méthodes permettant de créer des figures avec matplotlib et de les enregistrer.
"""

import logging
from math import sqrt, ceil, floor
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from brian2 import units, NeuronGroup, Synapses, SpikeGeneratorGroup, Network
from sklearn.metrics import confusion_matrix

from mnist_stdp_out import OUT_DIR
from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_stdp')

COURBES = 10  # Nombre de courbes d'activation que l'on souhaite
WEIGHTS_TO_RECORD = [4000, 60000]


def plot_post_testing(y_pred, y_true, parameters):
    # Matrice Confusion

    save_dir = Path(OUT_DIR, parameters.run_name)

    figure, ax = plt.subplots()
    max_val = 10
    cf = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize='true')
    plt.ylabel('étiquettes')
    plt.xlabel('prédictions')
    plt.title('Matrice de confusion')
    ax.matshow(cf, cmap=plt.cm.Blues)
    ax.xaxis.tick_bottom()
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(max_val):
        for j in range(max_val):
            c = cf[j, i]
            ax.text(i, j, f'{c:.2f}', va='center', ha='center')

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_mat.png')
    plt.show()


def plot_post_training(net, train_stats, parameters):
    Stopwatch.starting('plotting')
    LOGGER.info(f'Start plotting...')

    save_dir = Path(OUT_DIR, parameters.run_name)

    # Unpack stats returned from the train function
    spike_per_label_e, spike_per_label_input, average_spike_evolution_e, average_spike_evolution_i, \
    count_activation_map, average_weights_evolution = train_stats

    plt.figure()
    plt.plot(average_spike_evolution_e, linestyle='--', linewidth=3, label="exitateur")
    plt.plot(average_spike_evolution_i, label="inhibiteur")
    plt.title("Moyenne des décharge des neurones excitateurs avec les échantillons")
    plt.xlabel('Nombre d\'échantillons')
    plt.ylabel('Moyenne des décharges')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'spike_mean.png')
    plt.show()

    plt.figure()
    for i in range(2):
        plt.plot(net['volt_mon_e'].t[:10000] / units.second, net['volt_mon_e'].v[i, :10000], label=f'excitator {i}')
    plt.title(f'Potentiel {COURBES} premiers neurones excitateurs')
    plt.xlabel('Temps (s)')
    plt.ylabel('voltage des neurones en V')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'excitatory_v.png')
    plt.show()

    # Activation map graph
    plt.figure()
    for i in range(1, np.size(count_activation_map, axis=0)):
        plt.plot(range(parameters.nb_excitator_neurons), count_activation_map[i, :], label=f'exemple {i}')
    # plot chaque ligne de la matrice spike pour la carte d'activation
    plt.xlabel('indice du neurone de la couche excitatrice')
    plt.ylabel('Nombre de décharge par neurones')
    plt.title('Carte d\'activation de 9 exemples (couche excitatrice)')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.ylim([0, 10])
    plt.tight_layout()
    plt.savefig(save_dir / 'activity_map.png')
    plt.show()

    # Courbes d'accord
    plt.figure()
    for i in range(COURBES):
        plt.plot(range(10), spike_per_label_e[:, int(i * parameters.nb_excitator_neurons / 10)],
                 label=f'neurone {int(i * parameters.nb_excitator_neurons / 10)}')
    plt.title(f"Échantillon des courbes d\'accord de {COURBES} neurones")
    plt.xlabel('valeur de l\'étiquette')
    plt.ylabel('nombre de déchange')
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=[1.25, 1])
    plt.savefig(save_dir / 'accord.png')
    plt.show()

    # Histogramme des courbes d'accord
    plt.figure()
    plt.hist(net['synapses_input_e'].w, bins=20, edgecolor='black', label='')
    plt.title(f"répartition des poids selon leur valeur après {parameters.nb_train_samples} itérations")
    plt.xlabel('Valeur des poids')
    plt.ylabel('Quantité dans chaque catégorie')
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_dir / 'weights_repartition.png')
    plt.show()

    plt.subplot(211)
    plt.title('Synapses échantillonées entrée vers excitateur')
    plt.plot(net['synapses_input_e'].w[0:-1:200], '.k', label='poids des synapses')
    plt.ylabel('Poids')
    plt.xlabel('Indices des synapses')
    plt.subplot(212)
    plt.title('évolution des poids dans le temps')
    for i in range(len(WEIGHTS_TO_RECORD)):
        plt.plot(net['mon_input'].t / units.second, net['mon_input'].w[i], label=f'synapse {WEIGHTS_TO_RECORD[i]}')
    ax_average_weights = net['mon_input'].t[0:-1:5000]
    plt.plot(ax_average_weights, average_weights_evolution, label=f'moyenne des poids')
    plt.xlabel('Temps (s)')
    plt.ylabel('Poids')
    plt.ylim([0, 0.20])
    plt.tight_layout()
    plt.legend(loc='upper center', bbox_to_anchor=[0.85, 0.90], prop={'size': 6})
    plt.savefig(save_dir / 'weights_evolution.png')
    plt.show()

    # Activity map
    activity_map(spike_per_label_input, parameters)

    time_msg = Stopwatch.stopping('plotting')
    LOGGER.info(f'Plotting completed. {time_msg}.')


def activity_map(spike_per_label_input, parameters):
    # Normalize the matrix
    spikes_activity = spike_per_label_input - np.min(spike_per_label_input)
    spikes_activity = spikes_activity / np.max(spikes_activity)

    # Turn into pixels value [0-255]
    spikes_activity = np.rint(spikes_activity * 255)

    # Reshape in picture size
    spikes_activity = spikes_activity.reshape((10, 28, 28))

    fig = plt.figure()
    fig.suptitle(f'Carte d\'activité des neurones d\'entrée après {parameters.nb_train_samples} itérations')

    # Show images
    for i, image in enumerate(spikes_activity):
        sub = fig.add_subplot(2, 5, i + 1)
        sub.imshow(image, cmap=plt.cm.hot, interpolation='nearest')
        sub.axis('off')
        sub.title.set_text(i)

    plt.savefig(Path(OUT_DIR, parameters.run_name, 'activity_map.png'))
    plt.show()


def img_show(images, title: str = '', ground_truth: List[str] = None, prediction_success: List[bool] = None) -> None:
    """
    Show a set of images.

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
