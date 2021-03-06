"""
author: Victor Yon
date: 15/07/2019
version history: See Github
description: Défintion de fonctions utilitaires permettant de mesurer les temps d'exécution.
"""

from time import time


def duration_to_str(sec: float, precision: int = 2):
    """
    Transform a duration (in sec) into a human readable string.

    :param sec: The number of second of the duration. Decimals are milliseconds.
    :param precision: The number of unit we want. If 0 print all units.
    :return: A human readable representation of the duration.
    """

    if sec < 0:
        raise ValueError("Negative duration not supported")

    # Null duration
    if sec == 0:
        return "0ms"

    # Convert to ms
    mills = round(sec * 1_000)

    # Less than 1 millisecond
    if mills == 0:
        return "<1ms"

    periods = [
        ('d', 1_000 * 60 * 60 * 24),
        ('h', 1_000 * 60 * 60),
        ('m', 1_000 * 60),
        ('s', 1_000),
        ('ms', 1)
    ]

    strings = []
    for period_name, period_mills in periods:
        if abs(mills) >= period_mills:
            period_value, mills = divmod(mills, period_mills)
            strings.append("%s%s" % (period_value, period_name))

    if precision:
        strings = strings[:precision]

    return " ".join(strings)


class Stopwatch:
    _timers: dict = dict()

    def __init__(self, name: str):
        self._name = name
        self._start = None
        self._end = None

    def start(self) -> None:
        """ Start the stop watch """
        self._start = time()

    def stop(self) -> None:
        """ Stop the stop watch """
        self._end = time()

    def log(self, nb_item: int = 0) -> str:
        """
        Generate a human readable string which summary the current state of this stopwatch.
        :param nb_item: Number of item processed since the start.
        :return: A human readable summary of the current state.
        """

        if not self._start:
            return f"{self._name} timer not started"

        if not self._end:
            delta = time() - self._start
            msg = f"Duration so far: {duration_to_str(delta)}"

            if nb_item > 0:
                msg += f' ({duration_to_str(delta / nb_item)} / item)'

            return msg

        else:
            delta = self._end - self._start
            msg = f"Duration: {duration_to_str(delta)}"

            if nb_item > 0:
                msg += f' ({duration_to_str(delta / nb_item)} / item)'

            return msg

    @staticmethod
    def starting(timer_name: str) -> None:
        """
        Create and run a stopwatch with specified name.
        :param timer_name: The name of the stopwatch
        :return:
        """
        Stopwatch._timers[timer_name] = Stopwatch(timer_name)
        Stopwatch._timers[timer_name].start()

    @staticmethod
    def stopping(timer_name: str, nb_item: int = 0) -> str:
        """
        Stop and delete a stopwatch with specified name.
        :param nb_item: Number of item processed since the start.
        :param timer_name: The name of the stopwatch
        :return: A human readable summary of the final state of the stopped stopwatch.
        """
        if timer_name in Stopwatch._timers:
            Stopwatch._timers[timer_name].stop()
            log_msg = Stopwatch._timers[timer_name].log(nb_item)
            del Stopwatch._timers[timer_name]
            return log_msg
        else:
            raise ValueError(f"Stopwatch with name '{timer_name}' was never started or already stopped.")
