from dataclasses import dataclass

import quantities as units


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Parameters:
    """
    Storing all parameters of a simulation.
    """
    run_name: str = ''

    nb_train_samples: int = 50000
    nb_test_samples: int = 10000
    nb_validation_samples: int = 10000
    nb_epoch: int = 20
    batch_size: int = 256
    use_validation: bool = False

    duration_per_image: units = 100 * units.ms
    delta_t: units = 1 * units.ms

    tau_v: units = 20 * units.ms
    tau_i: units = 5 * units.ms
    v_threshold: float = 1.0

    nb_hidden_neurons: int = 128
    learning_rate: int = 0.01

    @property
    def absolute_duration(self):
        return int(self.duration_per_image / self.delta_t)

    def __post_init__(self):
        """
        Validate parameters.
        """
        if self.nb_train_samples + self.nb_test_samples + self.nb_validation_samples > 70000:
            raise ValueError('The total number of sample can\'t be more than 70000')

    def __str__(self):
        return '\n'.join([f'{name}: {str(value)}' for name, value in self.get_namespace().items()])

    def get_namespace(self):
        """
        :return: Convert to a dictionary object
        """
        return self.__dict__