"""
author: Victor Yon
date: 19/11/2020
version history: See Github
description: Défintion des fonctions permettant d'entregistrer les résultats dans des fichiers.
"""

import glob
import logging
from pathlib import Path

from simulation_parameters import SimulationParameters

OUT_DIR = './out'
LOGGER = logging.getLogger('mnist_stdp')


def init_out_directory(parameters: SimulationParameters):
    run_dir = Path(OUT_DIR, parameters.run_name)

    # If the keyword 'tmp' is used as run name, then remove the previous files
    if parameters.run_name == 'tmp':
        LOGGER.warning(f'Using temporary directory to save this run results.')
        if run_dir.exists():
            LOGGER.warning(f'Previous temporary files removed: {run_dir}')
            # Remove text files
            (run_dir / 'parameters.txt').unlink(missing_ok=True)
            (run_dir / 'results.txt').unlink(missing_ok=True)
            # Remove png images files
            for png_file in glob.glob(str(run_dir / '*.png')):
                Path(png_file).unlink()
            # Remove tmp directory
            run_dir.rmdir()

    run_dir.mkdir(parents=True)

    LOGGER.info(f'Output directory created: {run_dir}')

    parameter_file = run_dir / 'parameters.txt'
    with open(parameter_file, 'w+') as f:
        f.write(str(parameters))

    LOGGER.debug(f'Parameters saved in {parameter_file}')


def result_out(parameters: SimulationParameters, accuracy, time_msg):
    run_dir = Path(OUT_DIR, parameters.run_name)
    parameter_file = run_dir / 'results.txt'

    with open(parameter_file, 'w+') as f:
        f.write(
            '\n'.join([
                f'accuracy: {accuracy}',
                f'time: {time_msg}'
            ]))

    LOGGER.debug(f'Results saved in {parameter_file}')
