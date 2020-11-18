import logging
import sys

from mnist_stdp import run
from simulation_parameters import SimulationParameters

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger('mnist_stdp').setLevel(logging.INFO)

    # Set simulation parameters
    parameters = SimulationParameters(run_name='run-13',
                                      nb_train_samples=3,
                                      nb_test_samples=10,
                                      nu_post=0.1)

    # Run the beast
    run(parameters)
