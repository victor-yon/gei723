"""
author: Antoine Marion et Victor Yon
date: 19/11/2020
version history: See Github
description: Définition d'un objet de type `dataclass` permettant de stocker les paramètres d'entrainement et leurs valeurs par défaut.
"""

from dataclasses import dataclass

from brian2 import units, Quantity


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class SimulationParameters:
    """
    Storing all parameters of a simulation.
    """
    run_name: str = ''

    nb_train_samples: int = 60000
    nb_test_samples: int = 10000
    nb_epoch: int = 1

    nb_excitator_neurons: int = 400
    nb_inhibitor_neurons: int = 400

    exposition_time: Quantity = 0.35 * units.second
    resting_time: Quantity = 0.15 * units.second

    input_intensity: int = 2

    normalization: bool = True
    classification_type: str = 'single'

    offset: Quantity = 20.0 * units.mV
    v_rest_e: Quantity = -65 * units.mV
    v_rest_i: Quantity = -60 * units.mV
    v_reset_e: Quantity = -65 * units.mV
    v_reset_i: Quantity = -45 * units.mV
    v_thresh_e: Quantity = -52 * units.mV  # Will also include theta, offset and timer
    v_thresh_i: Quantity = -40 * units.mV
    refrac_e: Quantity = 5 * units.ms
    refrac_i: Quantity = 2 * units.ms

    tc_pre: Quantity = 20 * units.ms
    tc_post_1: Quantity = 20 * units.ms
    tc_post_2: Quantity = 40 * units.ms

    nu_pre: float = 0.0001  # learning rate
    nu_post: float = 0.01  # learning rate

    wmax: float = 1.0

    tc_theta: Quantity = 1e7 * units.ms
    theta_plus: Quantity = 0.05 * units.mV

    def __post_init__(self):
        """
        Validate parameters.
        """
        if self.nb_train_samples > 60000:
            raise ValueError('The number of train sample can\'t be more than 60000')
        if self.nb_test_samples > 10000:
            raise ValueError('The number of test sample can\'t be more than 10000')
        if self.classification_type not in ['single', 'group']:
            raise ValueError('The classification type should be "single" or "group".')

    def __str__(self):
        return '\n'.join([f'{name}: {str(value)}' for name, value in self.get_namespace().items()])

    def get_namespace(self):
        """
        :return: A dictionary object that can be use by brian as namespace
        """
        return self.__dict__
