import logging
import sys

from mnist_stdp import run
from simulation_parameters import SimulationParameters

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger('mnist_stdp').setLevel(logging.INFO)

    # Set simulation parameters
    # See file "simulation_parameters" for all possible parameters and default values
    parameters = SimulationParameters(run_name='tmp',
                                      nb_train_samples=50,
                                      nb_test_samples=50)

    # Run the beast
    run(parameters)
