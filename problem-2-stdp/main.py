import logging
import sys

from mnist_stdp import run

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.getLogger('mnist_stdp').setLevel(logging.DEBUG)

    run(nb_train_samples=20, nb_test_samples=20)
