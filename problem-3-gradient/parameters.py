from dataclasses import dataclass
from typing import Tuple

import quantities as units


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Parameters:
    """
    Storing all parameters of a simulation.
    """
    run_name: str = ''

    nb_train_samples: int = 50_000
    nb_test_samples: int = 10_000
    nb_validation_samples: int = 10_000
    nb_epoch: int = 20
    batch_size: int = 256
    use_validation: bool = False

    duration_per_image: units = 100 * units.ms
    delta_t: units = 1 * units.ms

    tau_v: units = 20 * units.ms
    tau_i: units = 5 * units.ms
    v_threshold: float = 1.0

    size_hidden_layers: Tuple[int, ...] = (128,)  # Number of neuron for each hidden layers
    learning_rate: int = 0.01

    surrogate_gradient: str = 'relu'  # "relu" or "fast_sigmoid" or "piecewise" or "sigmoid" or "piecewise_sym"
    surrogate_alpha: float = None

    extreme_learning: bool = False  # If true only the parameters of the last layer will trained

    @property
    def absolute_duration(self):
        return int(self.duration_per_image / self.delta_t)

    def __post_init__(self):
        """
        Validate parameters.
        """

        if len(self.size_hidden_layers) < 1:
            raise ValueError('At least one hidden layer is required.')

        # Keep 10 000 sample for validation only
        if self.nb_train_samples + self.nb_test_samples > 60_000:
            raise ValueError('The number of train + test samples can\'t be more than 60 000 '
                             '(keep 10 000 for validation).')

        if self.nb_train_samples + self.nb_test_samples + self.nb_validation_samples > 70_000:
            raise ValueError('The total number of sample can\'t be more than 70 000')

        if self.duration_per_image.units != self.delta_t.units:
            raise ValueError(f'duration_per_image ({self.duration_per_image}) and delta_t ({self.delta_t})'
                             f' should have the same unit.')

        if self.tau_v.units != self.delta_t.units:
            raise ValueError(f'tau_v ({self.tau_v}) and delta_t ({self.delta_t}) should have the same unit.')

        if self.tau_i.units != self.delta_t.units:
            raise ValueError(f'tau_i ({self.tau_i}) and delta_t ({self.delta_t}) should have the same unit.')

        if self.surrogate_gradient not in ['relu', 'fast_sigmoid', 'piecewise', 'sigmoid', 'piecewise_sym']:
            raise ValueError(f'Unknown surrogate_gradient "{self.surrogate_gradient}".'
                             f'Should be: "relu" or "fast_sigmoid" or "piecewise" or "sigmoid" or "piecewise_sym".')

        if self.surrogate_alpha is None:
            raise ValueError(f'The surrogate alpha need to be explicitly set.')

    def __str__(self):
        return '\n'.join([f'{name}: {str(value)}' for name, value in self.get_namespace().items()])

    def get_namespace(self):
        """
        :return: Convert to a dictionary object
        """
        return self.__dict__
