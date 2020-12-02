import logging
import sys

from parameters import Parameters
from run import run

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger('mnist_grad').setLevel(logging.INFO)

    # Set simulation parameters
    # See file "parameters" for all possible parameters and default values
    parameters = Parameters(run_name='tmp',
                            nb_train_samples=1000,
                            nb_test_samples=200,
                            batch_size=100,
                            nb_epoch=10)

    # Run the beast
    run(parameters)
