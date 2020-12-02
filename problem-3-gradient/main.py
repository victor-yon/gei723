import logging
import sys

from parameters import Parameters
from run import run

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger('mnist_grad').setLevel(logging.DEBUG)

    # Set simulation parameters
    # See file "parameters" for all possible parameters and default values
    parameters = Parameters(run_name='tmp',
                            nb_train_samples=50_000,
                            nb_test_samples=10_000,
                            batch_size=256,
                            nb_epoch=2)

    # Run the beast
    run(parameters)
