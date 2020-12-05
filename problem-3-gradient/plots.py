import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from parameters import Parameters
from results_output import save_plot

COURBES = 10


def plot_losses(losses_evolution, parameters: Parameters):
    plt.plot(losses_evolution)
    plt.title("Evolution de la fonction de cout pendant l'entrainement.\n"
              f"Taille des batches : {parameters.batch_size} - "
              f"Nombre d'époques : {parameters.nb_epoch} ({parameters.run_name})")
    plt.xlabel('Nombre de batches')
    plt.ylabel('Erreur (MSE)')
    save_plot(f'losses', parameters)
    plt.show()


# Matrice de confusion
def plot_post_test(y_pred, y_true, parameters: Parameters):
    figure, ax = plt.subplots()
    max_val = 10
    cf = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize='true')
    plt.ylabel('étiquettes')
    plt.xlabel('prédictions')
    plt.title(f'Matrice de confusion ({parameters.run_name})')
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
    plt.title(
        f'Carte d\'activation de {len(activation_map_data)} exemples (couche excitatrice) ({parameters.run_name})')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.tight_layout()
    save_plot('activation_map_hidden_layer', parameters)
    plt.show()


# Courbes d'accord couche sortie
def plot_output_one_hot(data ,parameters: Parameters): # dernier next_layer_input
    data = data.detach().numpy()
    
    plt.figure()
    for i in range(10):
        data_to_plot = d
        plt.plot()
    plt.xlabel()
    plt.ylabel()
    plt.title()
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    save_plot('courbes accord sortie', parameters)
    plt.show()

# Potentiel de noeuds de la couche cachée
def plot_potential_input_layer(parameters: Parameters):
    # Prend en entrée le ne
    plt.figure()
    plt.xlabel('indice du neurone de la couche excitatrice')
    plt.ylabel('Nombre de décharge par neurones')
    plt.title(f'Carte d\'activation de 9 exemples (couche excitatrice) ({parameters.run_name})')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.tight_layout()
    save_plot('potential', parameters)
    plt.show()


# Histogramme des courbes des poids
# Doit montrer la différence entre l'extreme learning et l'apprentissage standard
def plot_weight_hist(params, parameters: Parameters):
    nb_of_histograms = len(parameters.size_hidden_layers) + 1
    # params = params.detach().numpy()
    for i in range(nb_of_histograms):
        params[i] = params[i].detach().numpy()
        plt.subplot(nb_of_histograms, 1, i + 1)
        # params
        plt.hist(params[i], bins=20, edgecolor='black', label=f'{i}e weight tensor')
        plt.xlabel('indice du neurone de la couche excitatrice')
        plt.ylabel('Nombre de décharge par neurones')
        if i == 0:
            plt.title(f'Histogrammes des poids depuis la couche d\'entrée vers la sortie ({parameters.run_name})')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.tight_layout()
    save_plot('histograms', parameters)
    plt.show()


# Relu with different alpha
def plot_relu_alpha(parameters: Parameters):
    plt.figure()
    x_values = np.linspace(-2, 2, 301)
    relu_fn = np.zeros(len(x_values))
    relu_fn[np.where(x_values <= 0)] = -parameters.surrogate_alpha * x_values[np.where(x_values <= 0)]
    relu_fn[np.where(x_values > 0)] = x_values[np.where(x_values > 0)]
    plt.plot(x_values, relu_fn)
    plt.title(f'Relu avec alpha = {parameters.surrogate_alpha} ({parameters.run_name})')
    plt.tight_layout()
    save_plot(f'relu with alpha {parameters.surrogate_alpha}', parameters)
    plt.show()


# Fonction d'approximation du gradient
def plot_gradient_surrogates(parameters: Parameters):
    # Set un axe des x avec un linspace
    x_values = np.linspace(-1, 1, 101)

    plt.figure()

    # Tracer les trois fonctions que l'on a utilisées soit la fast sigmoid, absolue, 
    fast_sigmoid = x_values / (1 + np.absolute(x_values))
    step_function = np.zeros(len(x_values))
    # step_function[np.where(x_values < 0)] = 0
    step_function[np.where(x_values >= 0)] = 1
    piecewise_linear = np.zeros(len(x_values))
    piecewise_linear[np.where(x_values < -0.5)] = 1
    piecewise_linear[np.where((x_values >= -0.5) & (x_values < 0))] = 2

    plt.plot(x_values, fast_sigmoid, label='fast_sigmoid')
    plt.plot(x_values, step_function, label='step_function')
    plt.plot(x_values, piecewise_linear, label='piecewise_linear')
    plt.xlabel('Nombre d\'échantillons')
    plt.ylabel('Moyenne des décharges')
    plt.title(f'Fonction d\'approximation du gradient ({parameters.run_name})')
    plt.legend()
    plt.tight_layout()
    save_plot('gradient_surrogates', parameters)
    plt.show()
