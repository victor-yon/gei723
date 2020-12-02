import matplotlib.pyplot as plt

from parameters import Parameters
from results_output import save_plot


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
    # plt.savefig(save_dir / 'confusion_mat.png')
    plt.show()
# Courbes d'accord

# Histogramme des courbes des poids

# Evolution des poids

# Fonction de perte

# Fonction d'approximation du gradient