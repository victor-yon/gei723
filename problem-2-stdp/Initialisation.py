import numpy as np
from sklearn import datasets, model_selection
from brian2 import *
import matplotlib.pyplot as plt
import logging
import time


DATA_LIMIT = 15                         # Total number of samples used
TEST_SIZE = 5                           # Part of the data used for test
TRAIN_SIZE = DATA_LIMIT - TEST_SIZE     # samples that remains when test samples are substracted
NUMBER_NODES_PER_LAYER = 100            # Number of nodes in Excitatory and Inhibitory Layers
EPOCHS = 1                              # Number of epochs to run
CONNECTIVITY = [1, 0.25, 0.9]           # Respective connectivity value for input to exc, exc to inh, and inh to exc
WMAX = 6                                # Upper limit to weight values
NU_EE_PRE = 0.01                        # Learning rate for pre synaptic spikes
NU_EE_POST = 0.1                        # Learning rate for post synaptic spikes
COURBES = 30                            # Amount of neurons considered for the activation map
WEIGHT_INITIALIZATION = 'rand()'        # Initialization of the weights for synapses from input to exc layer
accuracy = []                           # Classifier accuracy with test dataset
weight_average = []                     # Average value of all weight at each sample iteration

# spécifie le niveau de logging. Infos are in the logfile.
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

refrac_e = 5. * units.ms
refrac_i = 2. * units.ms

tc_theta = 1e2 * units.ms
theta_plus_e = 0.15 * units.mV

tc_pre_ee = 20 * units.ms
tc_post_1_ee = 20 * units.ms
tc_post_2_ee = 40 * units.ms

# Taux d'apprentissage
nu_ee_pre =  NU_EE_PRE
nu_ee_post =  NU_EE_POST

input_rates = np.zeros([1,784])
input_group = PoissonGroup(784, rates = input_rates*Hz) # Groupe de Poisson

# Définition des groupes de neurones
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
inhibitory_group.d_I_synI = -8.5 * mV


# Définition des groupes de synapses
synapse_model = "w : 1"

synapse_on_pre_i = "gi_post -= w"

synapse_on_pre_e = "ge_post += w"

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
stdp_post = '''
    post2before = post2

    w = clip(w + (nu_ee_post*pre), 0, wmax) # TODO Check equation

    post1 = 1

    post2 = 1
'''

input_synapse = Synapses(input_group, excitatory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post,
                         method='euler')
input_synapse.connect(True, p=CONNECTIVITY[0])  # Fully connected
input_synapse.delay = '10 * ms'
input_synapse.plastic = True
input_synapse.w = WEIGHT_INITIALIZATION

e_i_synapse = Synapses(excitatory_group, inhibitory_group, model=synapse_model, on_pre=synapse_on_pre_e, on_post=synapse_on_pre_i, method='euler')
e_i_synapse.connect(True, p=CONNECTIVITY[1])
e_i_synapse.w = 'rand()*10.4'

i_e_synapse = Synapses(inhibitory_group, excitatory_group, model=synapse_model, on_pre=synapse_on_pre_e, on_post=synapse_on_pre_i, method='euler')
i_e_synapse.connect(True, p=CONNECTIVITY[2])
i_e_synapse.w = 'rand()*17.0'

total_number_of_synapses = len(input_synapse) + len(e_i_synapse) + len(i_e_synapse)
logging.info(f'Nombre de synapses dans le réseau: {total_number_of_synapses}')
e_monitor = SpikeMonitor(excitatory_group, record=False)
voltage_inh_neuron = SpikeMonitor(inhibitory_group, record=True)

# Créons le réseau.
net = Network(input_group, excitatory_group, inhibitory_group,
              input_synapse, e_i_synapse, i_e_synapse, e_monitor)

# Fonction pour mettre une pause dans le code. Uniquement pour débug
def pause(pause_time=None):
    if pause_time == None:
        pause = input('Press to continue')
    time.sleep(pause_time)

## Training

spikes = np.zeros((10, len(excitatory_group)))
old_spike_counts = np.zeros(len(excitatory_group))

# Training contains the EPOCH loop, returns weight_matrix and the average of weights for graphic purposes
def training(spikes, old_spike_counts, weight_average, voltage_inh_neuron):
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
            print(voltage_inh_neuron.v[0])
            # Enregistrer les décharges
            spikes[int(label)] += e_monitor.count - old_spike_counts

            # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
            old_spike_counts = np.copy(e_monitor.count)

            # Arrêter l'entrée
            input_group.rates = 0 * units.Hz

            # Laisser les variables retourner à leurs valeurs de repos
            net.run(resting_time)

            # Normaliser les poids
            weight_matrix = np.zeros([784, NUMBER_NODES_PER_LAYER])
            weight_matrix[input_synapse.i, input_synapse.j] = input_synapse.w

            super_script = (np.log10(np.ones(np.shape(weight_matrix))*np.sqrt((np.max(weight_matrix))/np.min(weight_matrix)))-np.log10(weight_matrix))/3.2

            L = np.max(weight_matrix)
            weight_matrix = weight_matrix
            weight_matrix = np.multiply(weight_matrix, np.power(np.ones(np.shape(weight_matrix))*10, super_script))
            weight_average.append(np.average(weight_matrix))
            input_synapse.w = weight_matrix[input_synapse.i, input_synapse.j]


    return weight_matrix, weight_average

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
        # current_spike_count = e_monitor.count - old_spike_counts
        # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
        old_spike_counts = np.copy(e_monitor.count)

        # Prédire la classe de l'échantillon
        # See which value is higher in lines of Spikes. If several, takes the first
        output_label = np.argmax(spikes, axis=1)[0]
        # Si la prédiction est correcte
        if output_label == int(label):
            num_correct_output += 1

        # Laisser les variables retourner à leurs valeurs de repos
        net.run(resting_time)
    # Add infos of the simulation in the log files with the parameters of the
    logging.info("The model accuracy is : %.3f" % (num_correct_output / len(X_test)))
    logging.info(f"parameters: {DATA_LIMIT}samples, including{TEST_SIZE}for testing, {NUMBER_NODES_PER_LAYER}neurons in learning layers,"
                  f"{NU_EE_POST} as learning rate, {EPOCHS}iterations, {CONNECTIVITY}connectivity")

# Training
beginning = time.time()
weight_matrix, weight_average = training(spikes, old_spike_counts, weight_average, voltage_inh_neuron)
training_time = time.time()
logging.info('training took {} seconds' .format(training_time - beginning))

# Activation map graph
plt.figure()
for i in range(10):
    plt.plot(np.arange(0, np.size(spikes, axis=1)), spikes[i, :]) # plot chaque lignes de la matrice spike pour la carte d'activation
plt.title('Carte d''activation')
plt.show()

# Courbes d'accord
plt.figure()
for i in range(COURBES):
    plt.plot(np.arange(0, np.size(spikes, axis=0)), spikes[:, i]) # plot les colonnes de spikes pour avoir les courbes d'accord. Pas toutes les faire car beaucoup trop
plt.title(f"Échantillon des courbes d''accord des {COURBES} premiers neurones")
plt.show()

# Histogramme des courbes d'accord
fig = plt.figure()
plt.hist(weight_matrix, bins=10, edgecolor='black')
bins = range(1,11)
plt.title(f"répartition des poids selon leur valeur après {TRAIN_SIZE} itérations")
plt.show()

# Average of weights with the number of samples
plt.figure()
plt.title(f"Moyenne des poids après {TRAIN_SIZE} itération")
plt.plot(weight_average)
plt.show()

# Tests and adding testing time to logfile
test_time_begin = time.time()
test(spikes, old_spike_counts)
logging.info("Test took {} seconds" .format(time.time()-test_time_begin))
logging.info('======End of logfile=======')
