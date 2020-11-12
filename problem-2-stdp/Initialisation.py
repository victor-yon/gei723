import numpy as np
from sklearn import datasets, model_selection
from brian2 import *
import matplotlib.pyplot as plt
import logging
import time


DATA_LIMIT = 20
TEST_SIZE = 5
NUMBER_NODES_PER_LAYER = 784
EPOCHS = 1
CONNECTIVITY = [0.0025, 0.9]
FILE_NAME = None #'net_file'
WMAX = 1
MU = 1
INDEX = 1 # Valeur entre 0 et 9 pour choisir le learning rate
NU_EE_PRE = np.arange(0.1, 1.1, 0.1)
NU_EE_POST = np.arange(0.1, 1.1, 0.1)

# spécifie le niveau de logging:
logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format='%(asctime)s:%(name)s:%(message)s')
logging.debug('--------beginning of logfile---------')

X_all, y_all = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data')

X_all.shape, y_all.shape

X = X_all[:DATA_LIMIT]
y = y_all[:DATA_LIMIT]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=TEST_SIZE)

index = np.random.randint(0, len(X_train)-1)

# plt.figure()
# plt.axis('off')
# plt.imshow(X_train[index].reshape(28, 28), cmap=plt.cm.gray_r)
# plt.title("Échantillon MNIST avec étiquette %s" % y_train[index]);

# Fixons le seed aléatoire afin de pouvoir reproduire les résultats
np.random.seed(0)

# Horloge de Brian2
defaultclock.dt = 0.5 * units.ms

# Cible de génération de code pour Brian2
prefs.codegen.target = 'cython'
# for index in NU_EE_PRE?

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

tc_theta = 1e7 * units.ms
theta_plus_e = 0.05 * units.mV

tc_pre_ee = 20 * units.ms
tc_post_1_ee = 20 * units.ms
tc_post_2_ee = 40 * units.ms

# Taux d'apprentissage
nu_ee_pre =  NU_EE_PRE[INDEX] # [0, 1]
nu_ee_post =  NU_EE_POST[INDEX]# [0, 1]


input_rates = np.ones([1,784])
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
    threshold='v > v_thresh_e', reset='v = v_reset_e; theta += theta_plus_e', method='euler')
excitatory_group.tau = 100 * units.ms
excitatory_group.d_I_synI = -100. * units.mV

inhibitory_group = NeuronGroup(
    N=NUMBER_NODES_PER_LAYER, model=neuron_model, refractory='refrac_i',
    threshold='v > v_thresh_i', reset='v = v_reset_i', method='euler')
inhibitory_group.tau = 10 * units.ms
inhibitory_group.d_I_synI = -85. * mV

synapse_model = "w : 1"

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

    pre = 1

    w = clip(w + (nu_ee_pre * post1), 0, wmax)
'''

stdp_post = '''
    post2before = post2

    w = clip(w + nu_ee_post*(pre-post2before)*(wmax - w)**mu, 0, wmax) # TODO Check equation

    post1 = 1

    post2 = 1
'''

input_synapse = Synapses(input_group, excitatory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post,
                         method='euler')
input_synapse.connect(True)  # Fully connected
input_synapse.delay = '10 * ms'
input_synapse.plastic = True
input_synapse.w = '1'

e_i_synapse = Synapses(excitatory_group, inhibitory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post,
                       method='euler')
e_i_synapse.connect(True, p=CONNECTIVITY[0])
e_i_synapse.w = 'rand()*10.4'

i_e_synapse = Synapses(inhibitory_group, excitatory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post,
                       method='euler')
i_e_synapse.connect(True, p=CONNECTIVITY[1])
i_e_synapse.w = 'rand()*17.0'

total_number_of_synapses = len(input_synapse) + len(e_i_synapse) + len(i_e_synapse)
logging.debug(f'Nombre de synapses dans le réseau: {total_number_of_synapses}')

e_monitor = SpikeMonitor(excitatory_group, record=False)

# Créons le réseau.

net = Network(input_group, excitatory_group, inhibitory_group,
              input_synapse, e_i_synapse, i_e_synapse, e_monitor)
if FILE_NAME is not None:
    store(filename = FILE_NAME)

#logging.debug(net)
def pause(pause_time=None):
    if pause_time == None:
        pause = input('Press to continue')
    time.sleep(pause_time)

    ## Entrainement
def training():
    spikes = np.zeros((10, len(excitatory_group)))
    old_spike_counts = np.zeros(len(excitatory_group))

    # Entrainement
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

            # Enregistrer les décharges
            spikes[int(label)] += e_monitor.count - old_spike_counts
            # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
            old_spike_counts = np.copy(e_monitor.count)

            # Arrêter l'entrée
            input_group.rates = 0 * units.Hz

            # Laisser les variables retourner à leurs valeurs de repos
            net.run(resting_time)

            # Normaliser les poids
            weight_matrix = np.zeros([784, 784])
            weight_matrix[input_synapse.i, input_synapse.j] = input_synapse.w
            # weight_matrix = weight_matrix/input_synapse.wmax
            col_sums = np.sum(weight_matrix, axis=0)
            # colFactors = weight_matrix[0] / col_sums
            colFactors = 1 / col_sums

            for k in range(len(excitatory_group)):
                weight_matrix[:, k] *= colFactors[k]
            input_synapse.w = weight_matrix[input_synapse.i, input_synapse.j]
    #store(filename = FILE_NAME)
    return spikes, old_spike_counts

def test(spikes, old_spike_counts):
    ## Test

    labeled_neurons = np.argmax(spikes, axis=1)
    test_time_begin = time.time()
    # labeled_neurons


    # Déasctiver la plasticité STDP
    input_synapse.plastic = False

    num_correct_output = 0

    for i, (sample, label) in enumerate(zip(X_test, y_test)):
        # Afficher régulièrement l'état d'avancement
        if (i % 10) == 0:
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

    logging.debug("The model accuracy is : %.3f" % (num_correct_output / len(X_test)))
    logging.debug(f"parameters: {DATA_LIMIT}samples, including{TEST_SIZE}for testing, {NUMBER_NODES_PER_LAYER}neurons in learning layers,"
                  f"{NU_EE_POST[INDEX]} as learning rate, {EPOCHS}iterations, {CONNECTIVITY[0]}connectivity in excitatory layer,"
                  f" {CONNECTIVITY[1]}connectivity in inhibitory layer")


beginning = time.time()
#pause(2)
if FILE_NAME is not None:
    print('net')
    print(net)
    restore(filename = FILE_NAME)
    print(net)
spikes, old_spike_counts = training()
training_time = time.time()
logging.debug('training took {} seconds' .format(training_time - beginning))
store(filename = FILE_NAME)

#restore
test_time_begin = time.time()
#pause(2)
test(spikes, old_spike_counts)
logging.debug("Test took {} seconds" .format(time.time()-test_time_begin))
logging.debug('======End of logfile=======')
