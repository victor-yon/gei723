import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brian2 import units

from mnist_stdp_out import OUT_DIR
from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_stdp')

COURBES = 10  # Nombre de courbes d'activation que l'on souhaite
WEIGHTS_TO_RECORD = [4000, 60000]


def plot_post_training(net, train_stats, parameters):
    Stopwatch.starting('plotting')
    LOGGER.info(f'Start plotting...')

    save_dir = Path(OUT_DIR, parameters.run_name)

    # Unpack stats returned from the train function
    spike_per_label, average_spike_evolution_e, average_spike_evolution_i, count_activation_map, \
    average_weights_evolution = train_stats

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
    plt.title('Carte d\'activation de 9 exemples d\'entraînement')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.ylim([0, 10])
    plt.tight_layout()
    plt.savefig(save_dir / 'activity_map.png')
    plt.show()

    # Courbes d'accord
    plt.figure()
    for i in range(COURBES):
        plt.plot(range(10), spike_per_label[:, int(i *parameters.nb_excitator_neurons/10)], label=f'neurone {i + 1}')
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

    time_msg = Stopwatch.stopping('plotting')
    LOGGER.info(f'Plotting completed. {time_msg}.')
