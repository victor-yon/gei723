"""
author: Victor Yon
date: 19/11/2020
version history: See Github
description: Fichier principal pour lancer une simulation.
"""

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
                                      classification_type='group',
                                      nb_train_samples=2500,
                                      nb_test_samples=500,
                                      normalization=False)

    # Run the beast
    run(parameters)
