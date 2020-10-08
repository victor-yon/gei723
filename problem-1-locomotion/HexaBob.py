import logging
from math import sqrt, cos
from typing import List, Tuple


class BobLeg:
    _index: Tuple[int, int]
    _is_on_ground: bool = True
    _current_position: float
    _min_position: float
    _max_position: float
    _move_size: float
    _size: float

    def __init__(self, index: Tuple[int, int], min_position: int = -45, max_position: int = 45, move_size: float = 9,
                 size: int = 1):
        """
        Create one Bob's leg.

        :param index: The leg index in the body
        :param min_position: The maximal angular position of the leg. (degree°)
        :param max_position: The minimal angular position of the leg. (degree°)
        :param move_size: The angular standard move size. (degree°)
        :param size: The leg size. Used to compute global move. (arbitrary unit)
        """

        if min_position >= max_position:
            raise ValueError(
                f'The min leg position ({min_position}) should be lower than the max position ({max_position}).')

        self._index = index
        self._min_position = min_position
        self._max_position = max_position
        self._current_position = min_position
        self._move_size = move_size
        self._size = size

    def raise_from_ground(self):
        """
        Raise this leg from the ground.
        """
        if self._is_on_ground:
            self._is_on_ground = False
            logging.info(f'Leg {self._index} raised from the ground')
        else:
            logging.debug(f'Leg {self._index} raised from the ground (--no change--)')

    def down_to_ground(self):
        """
        Put the leg down to the ground.
        """
        if not self._is_on_ground:
            self._is_on_ground = True
            logging.info(f'Leg {self._index} down to the ground')
        else:
            logging.debug(f'Leg {self._index} down to the ground (--no change--)')

    def move_forward(self) -> float:
        """
        Move the leg forward (from the bottom to the top).
        If the result is more than the max angular position then reduce it to the max.

        :return The ground move distance. (arbitrary unit)
        """
        before = self._current_position

        self._current_position = min(self._current_position + self._move_size, self._max_position)

        diff = self._current_position - before

        if diff != 0:
            logging.info(f'Leg {self._index} moved forward {diff}° ' +
                         '[touching ground]' if self._is_on_ground else '[no ground]')
        else:
            logging.debug(f'Leg {self._index} moved forward 0° ' +
                          '[touching ground]' if self._is_on_ground else '[no ground]' +
                                                                         ' (--no change--)')

        if self._is_on_ground:
            return self._angular_move_to_distance(diff)
        return 0

    def move_backward(self) -> float:
        """
        Move the leg backward (from the top to the bottom).
        If the result is less than the min angular position then increase it to the min.

        :return The ground move distance. (arbitrary unit)
        """
        before = self._current_position

        self._current_position = max(self._current_position - self._move_size, self._min_position)

        diff = self._current_position - before

        if diff != 0:
            logging.info(f'Leg {self._index} moved backward {diff}° ' +
                         '[touching ground]' if self._is_on_ground else '[no ground]')
        else:
            logging.debug(f'Leg {self._index} moved backward 0° ' +
                          '[touching ground]' if self._is_on_ground else '[no ground]' +
                                                                         ' (--no change--)')

        if self._is_on_ground:
            return self._angular_move_to_distance(diff)
        return 0

    def _angular_move_to_distance(self, angular_delta: float):
        # a^2 = b^2 + c^2 − 2bc cos(A)
        # b = c = self._size
        distance = sqrt((2 * (self._size ** 2)) - (4 * self._size * cos(abs(angular_delta))))

        if angular_delta < 0:
            return -distance
        return distance


class HexaBob:
    _nb_legs_pair: int = 3
    _legs: List[Tuple[BobLeg, BobLeg]] = list()

    def __init__(self, nb_legs_pair: int = 3):
        """
        Create a new Bob with a fixed number of legs.

        :param nb_legs_pair: The number of pair of legs. Should be at least 1.
        """

        if nb_legs_pair <= 0:
            raise ValueError(f'Invalid number pair of legs : "{nb_legs_pair}". Should be at least 1.')

        self._nb_legs_pair = nb_legs_pair
        for i in range(nb_legs_pair):
            self._legs.append((BobLeg((0, i)), BobLeg((1, i))))

    def get_leg(self, index_side: int, index_leg: int) -> BobLeg:
        """
        Get a specific leg.

        :param index_side: The side (left: 0, right 1)
        :param index_leg: The leg (top: 0, bottom: nb_legs_pair - 1)
        :return: The leg object.
        """
        return self._legs[index_side][index_leg]

    def __str__(self) -> str:
        """
        Print a visual representation of Bob with legs indexes
        :return: The printable string
        """
        s = f'HexaBob with {2 * self._nb_legs_pair} legs\n\n'

        s += '          ☉^☉\n'
        for i in range(self._nb_legs_pair):
            s += f'(0,{i})   o─╼█╾─o   (1,{i})\n'
        s += '           ▼\n'

        return s
