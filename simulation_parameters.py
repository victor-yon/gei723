from dataclasses import dataclass

from brian2 import units, Quantity


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class SimulationParameters:
    """
    Storing all parameters of a simulation.
    """
    nb_train_samples: int = 60000
    nb_test_samples: int = 10000

    nb_excitator_neurons: int = 400  # a faire varier
    nb_inhibitor_neurons: int = 400  # a faire varier

    exposition_time: Quantity = 0.35 * units.second
    resting_time: Quantity = 0.15 * units.second

    input_intensity: int = 2

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

    nu_pre: float = 0.0001  # learning rate à faire varier
    nu_post: float = 0.01  # learning rate à faire varier

    wmax: float = 1.0  # à faire varier

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

    def get_namespace(self):
        """
        :return: A dictionary object that can be use by brian as namespace
        """
        return self.__dict__
