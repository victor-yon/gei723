from brian2 import *
from sklearn import datasets, model_selection

# from Initialisation import

DATA_LIMIT = 20
TEST_SIZE = 5
TRAIN_SIZE = DATA_LIMIT - TEST_SIZE
NUMBER_NODES_PER_LAYER = 40
FILE_NAME = 'net_test'
save_to_disk = False
time_per_sample = 0.35 * units.second
resting_time = 0.15 * units.second

X_all, y_all = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data')

X_all.shape, y_all.shape

X = X_all[:DATA_LIMIT]
y = y_all[:DATA_LIMIT]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SIZE)

# Créer un net
input_rate = np.ones([1, 784])  # NUMBER_NODES_PER_LAYER])
layer_1 = PoissonGroup(784, input_rate * Hz)

eqs_neuron = '''
dv/dt = -v/tau : 1
tau : second
'''
layer_2 = NeuronGroup(NUMBER_NODES_PER_LAYER, eqs_neuron, threshold='v >= 1', method='euler')
layer_2.tau = 100 * ms

syn = Synapses(layer_1, layer_2, 'w : 1', on_pre='v_post += 0.5')
syn.connect(True, p=0.5)
syn.w = 'rand()*10'

e_monitor = SpikeMonitor(layer_2, record=False)

net = Network(layer_1, layer_2, syn, e_monitor)
# le store
net.store(filename=FILE_NAME)
net.restore(filename=f"{FILE_NAME} after {TRAIN_SIZE}samples - with {NUMBER_NODES_PER_LAYER}nodes in layer_1")
print(net)
# # faire une fonction train
# # insérer restore dedans avec net.restore
# spikes = np.zeros((10, NUMBER_NODES_PER_LAYER))
# old_spike_counts = np.zeros(len(layer_2))
# def train(spikes, old_spike_counts):
#     net.restore(filename=FILE_NAME)
#     for j, (sample, label) in enumerate(zip(X_train, y_train)):
#         # Afficher régulièrement l'état d'avancement
#         if (j % 1) == 0:
#             print("Running sample %i out of %i" % (j, len(X_train)))
#
#         # Configurer le taux d'entrée
#         layer_1.rates = sample / 4 * units.Hz
#         print(old_spike_counts)
#
#         # Simuler le réseau
#         net.run(time_per_sample)
#
#         # Enregistrer les décharges
#         spikes[int(label)] += e_monitor.count - old_spike_counts
#         # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
#         old_spike_counts = np.copy(e_monitor.count)
#
#         # Arrêter l'entrée
#         layer_1.rates = 0 * units.Hz
#
#         # Laisser les variables retourner à leurs valeurs de repos
#         net.run(resting_time)
#
#         # Normaliser les poids
#         #weight_matrix = np.zeros([784, 784])
#         #weight_matrix[input_synapse.i, input_synapse.j] = input_synapse.w
#         #if save_to_disk:
#         net.store(filename=f'{FILE_NAME} after {TRAIN_SIZE}samples - with {NUMBER_NODES_PER_LAYER}nodes in layer_1')
#     return
# train(spikes, old_spike_counts)

# def test():
