import matplotlib.pyplot as plt

from parameters import Parameters
from results_output import save_plot
from sklearn.metrics import confusion_matrix

COURBES = 10

def plot_losses(losses_evolution, parameters: Parameters):
    plt.plot(losses_evolution)
    plt.title("Evolution de la fonction de cout pendant l'entrainement.\n"
              f"Taille des batches : {parameters.batch_size} - "
              f"Nombre d'époques : {parameters.nb_epoch}")
    plt.xlabel('Nombre de batches')
    plt.ylabel('Erreur (MSE)')
    save_plot('losses', parameters)
    plt.show()




# Matrice de confusion
def plot_post_test(y_pred, y_true, parameters):
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
    save_plot('confusion_matrix', parameters)
    plt.show()

# Carte d'activation
def plot_activation_map():
    plt.figure()
    for i in range(1, np.size(count_activation_map, axis=0)):
        plt.plot(range(parameters.nb_excitator_neurons), count_activation_map[i, :], label=f'exemple {i}')
    # plot chaque ligne de la matrice spike pour la carte d'activation
    plt.xlabel('indice du neurone de la couche excitatrice')
    plt.ylabel('Nombre de décharge par neurones')
    plt.title('Carte d\'activation de 9 exemples (couche excitatrice)')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    #plt.ylim([0, 10])
    plt.tight_layout()
    save_plot('activation_map_hidden_layer',parameters)
    plt.show()


# Courbes d'accord




# Histogramme des courbes des poids

# Evolution des poids

# Fonction de perte

# Fonction d'approximation du gradient