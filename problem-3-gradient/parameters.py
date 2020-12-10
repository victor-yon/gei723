"""
author: Antoine Marion et Victor Yon
date: 10/12/2020
version history: See Github
description: Définition d'un objet de type `dataclass` permettant de stocker les paramètres d'entrainement et
leurs valeurs par défaut.
"""

from dataclasses import dataclass
from typing import Tuple

import quantities as units


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Parameters:
    """
    Storing all parameters of a simulation.
    """

    # Nom de l'exécution
    run_name: str = ''

    # Nombre d'images que l'on présente au réseau pendant la phase d'entraînement.
    nb_train_samples: int = 50_000
    # Nombre d'images que l'on présente au réseau pendant la phase de test.
    nb_test_samples: int = 10_000
    # Nombre d'images que l'on présente au réseau pendant la phase de validation.
    nb_validation_samples: int = 10_000
    # Nombre de fois où l'on présente le jeu de données (époques).
    nb_epoch: int = 20
    # Nombre d'images présentés entre chaque rétropropagation d'erreur.
    batch_size: int = 256
    # Si vrai utilise le jeu de validation à la place du jeu de test.
    use_validation: bool = False

    # Temps de présentation d'une image au réseau Temps total pendant lequel une image génère des décharges
    duration_per_image: units = 100 * units.ms
    # Discrétisation du temps, la mise à jour des calculs est effectuée tous les pas de temps delta t.
    delta_t: units = 1 * units.ms

    # Paramètres physique des neurones
    tau_v: units = 20 * units.ms
    tau_i: units = 5 * units.ms
    v_threshold: float = 1.0

    # Taux d'apprentissage permettant de réguler l'apprentissage pendant la rétropropagation de l'erreur.
    # Cela permet d'éviter d'atteindre des valeurs de poids trop élevées trop rapidement.
    learning_rate: float = 0.01

    # Type de fonction de non linéarité, cinq types de fonctions de non linéarité sont étudiées.
    surrogate_gradient: str = 'relu'  # "relu" or "fast_sigmoid" or "piecewise" or "sigmoid" or "piecewise_sym"
    # Valeur alpha de la fonction de subsitution
    surrogate_alpha: float = None

    # Nombre de neurones par couche cachée et nombre de couche cachée entre la couche d'entrée (présentation de l'image)
    # et la couche de sortie (prédiction du label)
    # Format: (size hidden 1, size hidden 2, ...)
    size_hidden_layers: Tuple[int, ...] = (128,)
    # If false the weights preceding this layer won't change during the training
    # Format: (connexions between input and hidden 1, connexions between hidden layers ...)
    trainable_layers: Tuple[bool, ...] = (True,)

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

        if len(self.size_hidden_layers) != len(self.trainable_layers):
            raise ValueError(f'The size_hidden_layers ({len(self.size_hidden_layers)})'
                             f' and trainable_layers ({len(self.trainable_layers)}) should have the same size.')

    def __str__(self):
        return '\n'.join([f'{name}: {str(value)}' for name, value in self.get_namespace().items()])

    def get_namespace(self):
        """
        :return: Convert to a dictionary object
        """
        return self.__dict__
