import logging
from pathlib import Path

from simulation_parameters import SimulationParameters

OUT_DIR = './out'
LOGGER = logging.getLogger('mnist_stdp')


def init_out_directory(parameters: SimulationParameters):
    run_dir = Path(OUT_DIR, parameters.run_name)
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

    LOGGER.debug(f'Parameters saved in {parameter_file}')
