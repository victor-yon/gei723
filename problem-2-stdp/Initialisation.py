import numpy as np
from sklearn import datasets, model_selection
from brian2 import *
import matplotlib.pyplot as plt
import logging
import time


DATA_LIMIT = 200
TEST_SIZE = 10
TRAIN_SIZE = DATA_LIMIT - TEST_SIZE
# Répartition pour quand les tests seront finis
# DATALIMIT = # Multiplie de 20
# VAL_LIMIT = 0.15*DATALIMIT
# TEST_LIMIT = 0.15*DATALIMIT
# TRAIN_LIMIT = DATALIMIT-(VAL_LIMIT+TEST_LIMIT)
NUMBER_NODES_PER_LAYER = 1600
EPOCHS = 1
CONNECTIVITY = [1, 0.0025, 0.9]
FILE_NAME = None #'net_file'
WMAX = 6
MU = 0.8
# INDEX = 1 # Valeur entre 0 et 9 pour choisir le learning rate
NU_EE_PRE = 0.01
NU_EE_POST = 0.1
COURBES = 30
WEIGHT_INITIALIZATION = 'rand()'
accuracy = []

# spécifie le niveau de logging:
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s:%(name)s:%(message)s')
logging.info('--------beginning of logfile---------')

X_all, y_all = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data')

X_all.shape, y_all.shape

X = X_all[:DATA_LIMIT]
y = y_all[:DATA_LIMIT]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SIZE)

index = np.random.randint(0, len(X_train)-1)
# Fixons le seed aléatoire afin de pouvoir reproduire les résultats
np.random.seed(0)

# Horloge de Brian2
defaultclock.dt = 0.5 * units.ms

# Cible de génération de code pour Brian2
prefs.codegen.target = 'cython'

time_per_sample =   0.35 * units.second
resting_time = 0.15 * units.second

v_rest_e = -65. * units.mV
v_rest_i = -60. * units.mV

v_reset_e = -65. * units.mV
v_reset_i = -45. * units.mV

v_thresh_e = -52. * units.mV
v_thresh_i = -40. * units.mV

refrac_e = 3. * units.ms
refrac_i = 1. * units.ms

tc_theta = 1e7 * units.ms
theta_plus_e = 0.05 * units.mV

tc_pre_ee = 20 * units.ms
tc_post_1_ee = 20 * units.ms
tc_post_2_ee = 40 * units.ms

# Taux d'apprentissage
nu_ee_pre =  NU_EE_PRE
nu_ee_post =  NU_EE_POST

input_rates = np.zeros([1,784])
input_group = PoissonGroup(784, rates = input_rates*Hz) # Groupe de Poisson

neuron_model = '''
    dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / tau  : volt (unless refractory)

    I_synE =  ge * nS * -v           : amp

    I_synI =  gi * nS * (d_I_synI-v) : amp

    dge/dt = -ge/(1.0*ms)            : 1

    dgi/dt = -gi/(2.0*ms)            : 1

    tau                              : second (constant, shared)

    d_I_synI                         : volt (constant, shared)

    dtheta/dt = -theta / (tc_theta)  : volt
'''

excitatory_group = NeuronGroup(
    N=NUMBER_NODES_PER_LAYER, model=neuron_model, refractory='refrac_e',
    threshold='v > v_thresh_e + theta', reset='v = v_reset_e; theta += theta_plus_e', method='euler')
excitatory_group.tau = 100 * units.ms
excitatory_group.d_I_synI = -100. * units.mV

inhibitory_group = NeuronGroup(
    N=NUMBER_NODES_PER_LAYER, model=neuron_model, refractory='refrac_i',
    threshold='v > v_thresh_i', reset='v = v_reset_i', method='euler')
inhibitory_group.tau = 10 * units.ms
inhibitory_group.d_I_synI = -85. * mV
voltage_exc_neuron = StateMonitor(excitatory_group[0], 'v', record=True)
synapse_model = "w : 1"

synapse_on_pre_i = "ge_post -= w"

synapse_on_pre_e = "gi_post += w"

stdp_synapse_model = '''
    w : 1

    plastic : boolean (shared) # Activer/désactiver la plasticité

    post2before : 1

    dpre/dt   =   -pre/(tc_pre_ee) : 1 (event-driven)

    dpost1/dt  = -post1/(tc_post_1_ee) : 1 (event-driven)

    dpost2/dt  = -post2/(tc_post_2_ee) : 1 (event-driven)

    wmax = WMAX : 1

    mu = MU : 1
'''

stdp_pre = '''
    ge_post += w

    pre = 1.

    w = clip(w + (nu_ee_pre * post1), 0, wmax)
'''
# w = clip(w + nu_ee_post*(pre-post2before)*(wmax - w)**mu, 0, wmax)
stdp_post = '''
    post2before = post2

    w = clip(w + (nu_ee_post*pre)*(wmax-w), 0, wmax) # TODO Check equation

    post1 = 1

    post2 = 1
'''

input_synapse = Synapses(input_group, excitatory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post,
                         method='euler')
input_synapse.connect(True, p=CONNECTIVITY[0])  # Fully connected
input_synapse.delay = '10 * ms'
input_synapse.plastic = True
input_synapse.w = WEIGHT_INITIALIZATION

e_i_synapse = Synapses(excitatory_group, inhibitory_group, model=synapse_model, method='euler')
e_i_synapse.connect(j='i')
e_i_synapse.w = 'rand()*10.4'

i_e_synapse = Synapses(inhibitory_group, excitatory_group, model=synapse_model, method='euler')
i_e_synapse.connect(True, p=CONNECTIVITY[2])
i_e_synapse.w = 'rand()*17.0'

total_number_of_synapses = len(input_synapse) + len(e_i_synapse) + len(i_e_synapse)
logging.info(f'Nombre de synapses dans le réseau: {total_number_of_synapses}')

e_monitor = SpikeMonitor(excitatory_group, record=False)

# Créons le réseau.
net = Network(input_group, excitatory_group, inhibitory_group,
              input_synapse, e_i_synapse, i_e_synapse, e_monitor)
# net.store(filename = f"{FILE_NAME}")
# net.restore(filename=f"{FILE_NAME} entraînement")
#print(net)

if FILE_NAME is not None:
    net.store(filename = FILE_NAME)

def pause(pause_time=None):
    if pause_time == None:
        pause = input('Press to continue')
    time.sleep(pause_time)

## Entrainement

spikes = np.zeros((10, len(excitatory_group)))
old_spike_counts = np.zeros(len(excitatory_group))
first_run = 1
def training(spikes, old_spike_counts):
    # Entrainement
    #net.restore(filename = FILE_NAME)
    number_of_epochs = EPOCHS
    for i in range(number_of_epochs):
        print('Starting epoch %i' % i)
        for j, (sample, label) in enumerate(zip(X_train, y_train)):
            # Afficher régulièrement l'état d'avancement
            if (j % 1) == 0:
                print("Running sample %i out of %i" % (j, len(X_train)))

            # Configurer le taux d'entrée
            input_group.rates = sample / 4 * units.Hz

            # Simuler le réseau
            net.run(time_per_sample)
            # print('voltage neurone 1')
            # print(voltage_exc_neuron)
            # Enregistrer les décharges
            spikes[int(label)] += e_monitor.count - old_spike_counts
            # print('e_count')
            # print(e_monitor.count)
            #
            # print('old_spike_count')
            # print(old_spike_counts)

            # print('matrix spikes')
            # print(spikes[:,:])
            # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
            old_spike_counts = np.copy(e_monitor.count)

            # Arrêter l'entrée
            input_group.rates = 0 * units.Hz

            # Laisser les variables retourner à leurs valeurs de repos
            net.run(resting_time)

            # Normaliser les poids
            weight_matrix = np.zeros([784, NUMBER_NODES_PER_LAYER])
            weight_matrix[input_synapse.i, input_synapse.j] = input_synapse.w
            # print('Matrice poids avant normalisation')
            # print(weight_matrix[362:422][15:25])
            # weight_matrix = weight_matrix/input_synapse.wmax
            # col_sums = np.sum(weight_matrix, axis=0)
            # colFactors = weight_matrix[0] / col_sums
            super_script = (np.log10(np.ones(np.shape(weight_matrix))*np.sqrt((np.max(weight_matrix))/np.min(weight_matrix)))-np.log10(weight_matrix))/3.5
            # print('super_script')
            # print(super_script)
            L = np.max(weight_matrix)
            weight_matrix = weight_matrix/L
            weight_matrix = np.multiply(weight_matrix, np.power(np.ones(np.shape(weight_matrix))*10, super_script))

            # for k in range(len(excitatory_group)):
            #     weight_matrix[:, k] *= colFactors[k]
            input_synapse.w = weight_matrix[input_synapse.i, input_synapse.j]
            # print('matrice poids après normalisation')
            # print(weight_matrix[:10][:10])
            #if not sample % 100:
                # input_synapse.plastic = False
                # num_correct_output = 0

                # for i, (sample, label) in enumerate(zip(X_test, y_test)):
                #     # Afficher régulièrement l'état d'avancement
                #     if (i % 1) == 0:
                #         print("Running sample %i out of %i" % (i, len(X_test)))
                #
                #     # Configurer le taux d'entrée
                #     # ATTENTION, vous pouvez utiliser un autre type d'encodage
                #     input_group.rates = sample / 4 * units.Hz
                #
                #     # Simuler le réseau
                #     net.run(time_per_sample)
                #
                #     # Calculer le nombre de décharges pour l'échantillon
                #     current_spike_count = e_monitor.count - old_spike_counts
                #     # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
                #     old_spike_counts = np.copy(e_monitor.count)
                #
                #     # Prédire la classe de l'échantillon
                #     output_label = np.argmax(spikes, axis=1)[0]
                #     # Si la prédiction est correcte
                #     if output_label == int(label):
                #         num_correct_output += 1
                #
                #     # Laisser les variables retourner à leurs valeurs de repos
                #     net.run(resting_time)
                #
                # logging.info("The model accuracy is : %.3f" % (num_correct_output / len(X_test)))
                # accuracy.append(num_correct_output / len(X_test))
                # logging.info(
                #     f"parameters: {DATA_LIMIT}samples, including{TEST_SIZE}for testing, {NUMBER_NODES_PER_LAYER}neurons in learning layers,"
                #     f"{NU_EE_POST[INDEX]} as learning rate, {EPOCHS}iterations, {CONNECTIVITY[0]}connectivity in excitatory layer,"
                #     f" {CONNECTIVITY[1]}connectivity in inhibitory layer")
    #print(net)
    #store(filename =f"{FILE_NAME} after {TRAIN_SIZE} samples") #samples, {NUMBER_NODES_PER_LAYER}neurons in learning layers,"
                  # f"{NU_EE_POST[INDEX]} as learning rate, {EPOCHS}iterations, {CONNECTIVITY[0]}connectivity in excitatory layer,"
                  # f" {CONNECTIVITY[1]}connectivity in inhibitory layer")
    return weight_matrix

def test(spikes, old_spike_counts):
    labeled_neurons = np.argmax(spikes, axis=1)

    # Déasctiver la plasticité STDP
    input_synapse.plastic = False
    num_correct_output = 0
    for i, (sample, label) in enumerate(zip(X_test, y_test)):
        # Afficher régulièrement l'état d'avancement
        if (i % 1) == 0:
            print("Running sample %i out of %i" % (i, len(X_test)))

        # Configurer le taux d'entrée
        # ATTENTION, vous pouvez utiliser un autre type d'encodage
        input_group.rates = sample / 4 * units.Hz

        # Simuler le réseau
        net.run(time_per_sample)

        # Calculer le nombre de décharges pour l'échantillon
        current_spike_count = e_monitor.count - old_spike_counts
        # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
        old_spike_counts = np.copy(e_monitor.count)

        # Prédire la classe de l'échantillon
        output_label = np.argmax(spikes, axis=1)[0]
        # Si la prédiction est correcte
        if output_label == int(label):
            num_correct_output += 1

        # Laisser les variables retourner à leurs valeurs de repos
        net.run(resting_time)

    logging.info("The model accuracy is : %.3f" % (num_correct_output / len(X_test)))
    #accuracy.append(num_correct_output / len(X_test))
    logging.info(f"parameters: {DATA_LIMIT}samples, including{TEST_SIZE}for testing, {NUMBER_NODES_PER_LAYER}neurons in learning layers,"
                  f"{NU_EE_POST} as learning rate, {EPOCHS}iterations, {CONNECTIVITY[0]}connectivity in excitatory layer,"
                  f" {CONNECTIVITY[1]}connectivity in inhibitory layer")


beginning = time.time()
if FILE_NAME is not None:
    #print('net')
    net.restore(filename = FILE_NAME)
weight_matrix = training(spikes, old_spike_counts)
training_time = time.time()
logging.info('training took {} seconds' .format(training_time - beginning))
#store(filename = FILE_NAME)
plt.figure()
for i in range(10):
    plt.plot(np.arange(0, np.size(spikes, axis=1)), spikes[i, :]) # plot chaque lignes de la matrice spike pour la carte d'activation
plt.title('Carte d''activation')
plt.show()

plt.figure()
for i in range(COURBES):
    plt.plot(np.arange(0, np.size(spikes, axis=0)), spikes[:, i]) # plot les colonnes de spikes pour avoir les courbes d'accord. Pas toutes les faire car beaucoup trop
plt.title(f"Échantillon des courbes d''accord des {COURBES} premiers neurones")
plt.show()

# Histogramme des courbes d'accord
fig = plt.figure()
#plt.hist(sum_mat_test[:50], bins=10, edgecolor='black')
plt.hist(weight_matrix, bins=10, edgecolor='black')
bins = range(1,11)
#bins_labels(bins, fontsize=15)
plt.title(f"répartition des poids selon leur valeur après {TRAIN_SIZE} itérations")
plt.show()

# plt.figure()
# plt.plot(voltage_exc_neuron.t/ms, voltage_exc_neuron.v[0])
# plt.show()


#restore
if FILE_NAME is not None:
    print('net')
    net.restore(filename = f"{FILE_NAME} after {TRAIN_SIZE} samples")
test_time_begin = time.time()
test(spikes, old_spike_counts)
#print(net)
logging.info("Test took {} seconds" .format(time.time()-test_time_begin))
logging.info('======End of logfile=======')
