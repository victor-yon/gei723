import logging

import matplotlib.pyplot as plt
from brian2 import units

from stopwatch import Stopwatch

LOGGER = logging.getLogger('mnist_stdp')


def plot_post_training(net, train_stats, parameters):
    Stopwatch.starting('plotting')
    LOGGER.info(f'Start plotting...')

    # Unpack stats returned from the train function
    spike_per_label, average_spike_evolution_e, average_spike_evolution_i = train_stats

    plt.figure()
    plt.plot(average_spike_evolution_e, label="exitateur")
    plt.plot(average_spike_evolution_i, label="inhibiteur")
    plt.title("Evolution de la moyenne des décharge exitateur")
    plt.legend()
    # plt.subplot(212)
    # plt.plot(evolution_moyenne_matrice_poids)
    # plt.title("Evolution de la moyenne des poids")
    plt.show()

    plt.figure()
    plt.plot(net['volt_mon_e'].t / units.second, net['volt_mon_e'].v[0], label='excitator')
    plt.plot(net['volt_mon_i'].t / units.second, net['volt_mon_i'].v[0], label='inhibitor')
    plt.title('Potentiel firtst neuron exc et inhib')
    plt.legend()
    plt.show()

    # Activation map graph
    # plt.figure()
    # for i in range(len(count_activation_map)):
    #     plt.plot(range(len(current_spike_count_e)), count_activation_map[i])
    # # plot chaque ligne de la matrice spike pour la carte d'activation
    # plt.title('Carte d'' activation')
    # plt.show()

    # Courbes d'accord
    # plt.figure()
    # plt.plot(cou)
    # # plot les colonnes de spikes pour avoir les courbes d'accord. Pas toutes les faire car beaucoup trop
    # plt.title(f"Échantillon des courbes d''accord des {COURBES} premiers neurones")
    # plt.show()

    # Histogramme des courbes d'accord
    fig = plt.figure()
    plt.hist(net['synapses_input_e'].w, bins=10, edgecolor='black')
    bins = range(1, 11)
    plt.title(f"répartition des poids selon leur valeur après {parameters.nb_train_samples} itérations")
    plt.show()

    # Average of weights with the number of samples
    # plt.figure()
    # plt.title(f"Moyenne des poids après {nb_train_samples} itération")
    # plt.plot(evolution_moyenne_matrice_poids)
    # plt.show()

    plt.subplot(211)
    plt.title('Synapses input to exc')
    plt.plot(net['synapses_input_e'].w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(net['mon_input'].t / units.second, net['mon_input'].w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.title('Synapses exc to inh')
    plt.plot(net['synapses_e_i'].w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(net['mon_e_i'].t / units.second, net['mon_e_i'].w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.title('Synapses inh to exc')
    plt.plot(net['synapses_i_e'].w, '.k')
    plt.ylabel('Weights')
    plt.xlabel('Synapses index')
    plt.subplot(212)
    plt.plot(net['mon_i_e'].t / units.second, net['mon_i_e'].w.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

    time_msg = Stopwatch.stopping('plotting')
    LOGGER.info(f'Plotting completed. {time_msg}.')
