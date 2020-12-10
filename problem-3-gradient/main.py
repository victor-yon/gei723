"""
author: Victor Yon
date: 10/12/2020
version history: See Github
description: Fichier principal pour lancer une simulation.
"""

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
                            nb_epoch=20,
                            surrogate_gradient='piecewise_sym',
                            surrogate_alpha=0.5,
                            size_hidden_layers=(128, 64),
                            trainable_layers=(True, True))

    # Run the beast
    run(parameters)
