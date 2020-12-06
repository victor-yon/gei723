import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from parameters import Parameters
from results_output import save_plot



def plot_losses(losses_evolution, parameters: Parameters):
    plt.plot(losses_evolution)
    plt.title("Evolution de la fonction de coût pendant l'entraînement.\n"
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
def plot_activation_map(activation_map_data, parameters: Parameters):
    plt.figure()

    for i in range(len(activation_map_data)):
        activation_map_data_to_plot = activation_map_data[i].detach().numpy()
        plt.plot(range(parameters.size_hidden_layers[0]), activation_map_data_to_plot, label=f'exemple {i}')
    # plot chaque ligne de la matrice spike pour la carte d'activation
    plt.xlabel('indice du neurone de la couche excitatrice')
    plt.ylabel('Nombre de décharge par neurones')
    plt.title(
        f'Carte d\'activation de {len(activation_map_data)} exemples (1ère couche cachée) ({parameters.run_name})')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.tight_layout()
    save_plot('activation_map_hidden_layer', parameters)
    plt.show()

# Courbes d'accord couche sortie
def plot_output_one_hot(data, labels, parameters: Parameters): # dernier next_layer_input
    data = data.detach().numpy()
    data_to_plot = np.zeros([10, 10])
    plt.figure()
    for i in range(10):
        data_to_plot[i,:] = np.sum(data[np.where([labels == 0])[0],:],axis=0)
        plt.plot(range(10), data[i,:], label=f'label {i}')
    plt.xlabel('étiquettes')
    plt.xlim([-1, 10])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.ylabel('Décharge par étiquette selon le label')
    plt.title(f'Courbes d\'accord ({parameters.run_name})')
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1)
    plt.tight_layout()
    save_plot('courbes accord sortie', parameters)
    plt.show()

# Histogramme des poids
def plot_weight_hist(params, parameters: Parameters):
    mat1 = params[0].detach().numpy()
    mat2 = params[-1].detach().numpy()
    
    fig, axs = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle(f'Histogrammes des poids synaptiques ({parameters.run_name})')
    axs[0].hist(mat1, bins=20, edgecolor='black')
    axs[0].set_title('entrée -> première couche cachée')
    axs[0].set_ylabel('Quantités par valeur')
    
    axs[1].hist(mat2, bins=20, edgecolor='black')
    axs[1].set_title('dernière couche cachée -> sortie')
    axs[1].set_xlabel('valeur des poids')
    axs[1].set_ylabel('Quantités par valeur')
    save_plot('histograms', parameters)
    plt.show()

# Évolution de certains poids
def plot_weight_evo(weight_evo, parameters: Parameters):
    x_val = range(len(weight_evo[0]))
    fig, axs = plt.subplots(2,1,constrained_layout=True)
    fig.suptitle(f'Évolution de poids sélectionnés au hasard ({parameters.run_name})')
    axs[0].set_xlabel('temps')
    axs[0].plot(x_val, weight_evo[0], label='Poids 1')
    axs[0].plot(x_val, weight_evo[1], label='Poids 2')
    axs[0].set_title('entrée->cachée')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('valeur')
    axs[1].plot(x_val,weight_evo[2], label='Poids 3')
    axs[1].plot(x_val, weight_evo[3], label='Poids 4')
    axs[1].set_xlabel('temps')
    axs[1].set_ylabel('valeur')
    axs[1].set_title('cachée->sortie')
    axs[1].legend(loc='upper right')
    save_plot('weight_evolution', parameters)
    plt.show()


# Relu with different alpha
def plot_relu_alpha(parameters: Parameters):
    plt.figure()
    x_values = np.linspace(-2, 2, 301)
    relu_fn = np.zeros(len(x_values))
    relu_fn[np.where(x_values <= 0)] = parameters.surrogate_alpha * x_values[np.where(x_values <= 0)]
    relu_fn[np.where(x_values > 0)] = x_values[np.where(x_values > 0)]
    plt.plot(x_values, relu_fn)
    plt.title(f'Relu avec alpha = {parameters.surrogate_alpha} ({parameters.run_name})')
    plt.axhline(0, color='black', linestyle=':')
    plt.axvline(0, color='black', linestyle=':')
    plt.xlabel('valeurs de x')
    plt.ylabel('Relu(x)')
    plt.tight_layout()
    save_plot(f'Relu with alpha {parameters.surrogate_alpha}', parameters)
    plt.show()


# Fonction d'approximation du gradient
def plot_gradient_surrogates(parameters: Parameters):
    # Set un axe des x avec un linspace
    x_values = np.linspace(-1, 1, 101)

    plt.figure()
    alpha = parameters.surrogate_alpha
    # Tracer les trois fonctions que l'on a utilisées soit la fast sigmoid, absolue,
    if parameters.surrogate_gradient == 'fast_sigmoid':
        fast_sigmoid = x_values / (1 + np.absolute(x_values))
        plt.plot(x_values, fast_sigmoid, label='fast_sigmoid')
        
    elif parameters.surrogate_gradient == 'relu':
        step_function = np.zeros(len(x_values))
        # step_function[np.where(x_values < 0)] = 0
        step_function[np.where(x_values >= 0)] = 1
        plt.plot(x_values, step_function, label='step_function')
        
    elif parameters.surrogate_gradient == 'piecewise':
        piecewise_linear = np.zeros(len(x_values))
        piecewise_linear[x_values >= alpha] = (1 / (1 - alpha)) * x_values[
                        x_values >= alpha] - alpha / (1 - alpha)
        plt.plot(x_values, piecewise_linear, label='piecewise_linear')

        
    elif parameters.surrogate_gradient == 'piecewise_sym':
        piecewise_sym = np.ones(len(x_values))*x_values
        piecewise_sym[x_values <= -alpha] = 0
        piecewise_sym[x_values > alpha] = 0
        plt.plot(x_values, piecewise_sym, label='piecewise_symetric')

    plt.xlabel('Potentiel membranaire')
    plt.ylabel('Approximation du gradient')
    plt.title(f'Fonction d\'approximation du gradient ({parameters.run_name})')
    plt.legend()
    plt.tight_layout()
    save_plot('gradient_surrogates', parameters)
    plt.show()
