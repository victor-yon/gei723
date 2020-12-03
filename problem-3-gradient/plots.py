import matplotlib.pyplot as plt

from parameters import Parameters
from results_output import save_plot
from sklearn.metrics import confusion_matrix
import numpy as np

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
def plot_post_test(y_pred, y_true, parameters: Parameters):
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
# On veut la somme des spikes à travers le temps des 128 neurones cachées à 
def plot_activation_map(activation_map_data, parameters: Parameters):
    
    plt.figure()
    
    for i in range(len(activation_map_data)):
        activation_map_data_to_plot = activation_map_data[i].detach().numpy()
        plt.plot(range(parameters.size_hidden_layers[0]), activation_map_data_to_plot, label=f'exemple {i}')
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

# Potentiel de noeuds de la couche cachée


# Histogramme des courbes des poids

# Evolution des poids

# Fonction d'approximation du gradient
def gradient_surrogates():
    plt.figure()
    # Set un axe des x avec un linspace
    x_values = np.linspace(-1, 1, 101)
    # Tracer les trois fonctions que l'on a utilisées soit la fast sigmoid, absolue, 
    fast_sigmoid = x_values / (1 + np.absolute(x_values))
    step_function = 
    piecewise_linear
    
    plt.plot()