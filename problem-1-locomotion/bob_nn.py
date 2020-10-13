from brian2 import *

from cpg_nn import monitor_cpg, build_cpg_nn, plot_monitor_cpg
from direction_nn import build_direction_nn, monitor_direction, plot_monitor_direction, LEFT
from ground_contact_nn import build_ground_contact_nn, monitor_ground_contact, plot_monitor_ground_contact
from leg_nn import legs_nn, monitor_legs, plot_monitor_legs


def build_nn(nb_leg_pair: int = 3, sensor_back: float = 0.2, sensor_front: float = 0,
             sensor_left: float = 0, sensor_right: float = 0):
    """
    Build the Bob neural network and run it.

    :param nb_leg_pair: The number of pair of leg
    :param sensor_back: The back sensor [0,1]
    :param sensor_front: The front sensor [0,1]
    :param sensor_left: The left sensor [0,1]
    :param sensor_right: The right sensor [0,1]
    """

    nn = Network()

    # ======================= CPG =====================
    # The central pattern generator
    cpg_nn = build_cpg_nn(sensor_back)
    nn.add(cpg_nn)
    cpg_core = cpg_nn[0]  # Keep core group for modules link
    # Monitor
    m_cpg = monitor_cpg(cpg_nn)
    nn.add(m_cpg)

    # ==================== Direction ==================
    # For right and left rotation
    # 4 direction sub-network : 2 for each side

    # Left 0
    direction_nn_left_0 = build_direction_nn(cpg_core, 0, sensor_right)
    nn.add(direction_nn_left_0)
    direction_left_0 = direction_nn_left_0[0]  # Keep main neurone for modules link
    # Monitors
    m_direction_left_0 = monitor_direction(direction_nn_left_0)
    nn.add(m_direction_left_0)

    # Left 1
    direction_nn_left_1 = build_direction_nn(cpg_core, 1, sensor_right)
    direction_left_1 = direction_nn_left_1[0]  # Keep main neurone for modules link
    nn.add(direction_nn_left_1)

    # Right 0
    direction_nn_right_0 = build_direction_nn(cpg_core, 0, sensor_left)
    direction_right_0 = direction_nn_right_0[0]  # Keep main neurone for modules link
    nn.add(direction_nn_right_0)

    # Right 1
    direction_nn_right_1 = build_direction_nn(cpg_core, 1, sensor_left)
    direction_right_1 = direction_nn_right_1[0]  # Keep main neurone for modules link
    nn.add(direction_nn_right_1)

    # ================= Ground Contact ================
    # For up and down the leg
    # 2 ground contact module (up output & down output)

    ground_contact_nn_0 = build_ground_contact_nn(cpg_core, sensor_front, inverted=False)
    ground_output_0 = ground_contact_nn_0[0]  # Keep output neurone for modules link
    nn.add(ground_contact_nn_0)
    # Monitors
    m_ground_contact_nn_0 = monitor_ground_contact(ground_contact_nn_0)
    nn.add(m_ground_contact_nn_0)

    ground_contact_nn_1 = build_ground_contact_nn(cpg_core, sensor_front, inverted=True)
    ground_output_1 = ground_contact_nn_1[0]  # Keep output neurone for modules link
    nn.add(ground_contact_nn_1)

    # ====================== Legs =====================
    monitors_legs = list()

    # Left even legs
    legs_left_0 = legs_nn(nb_leg_pair // 2 + 1, direction_left_0, direction_left_1, ground_output_0, ground_output_1)
    m_legs_left_0 = monitor_legs(legs_left_0)
    monitors_legs.append(m_legs_left_0)
    nn.add(legs_left_0)
    nn.add(m_legs_left_0)

    # Left odd legs
    legs_left_1 = legs_nn(nb_leg_pair // 2, direction_left_1, direction_left_0, ground_output_1, ground_output_0)
    m_legs_left_1 = monitor_legs(legs_left_1)
    monitors_legs.append(m_legs_left_1)
    nn.add(legs_left_1)
    nn.add(m_legs_left_1)

    # Right even legs
    legs_right_0 = legs_nn(nb_leg_pair // 2, direction_right_1, direction_right_0, ground_output_1, ground_output_0)
    m_legs_right_0 = monitor_legs(legs_right_0)
    monitors_legs.append(m_legs_right_0)
    nn.add(legs_right_0)
    nn.add(m_legs_right_0)

    # Right odd legs
    legs_right_1 = legs_nn(nb_leg_pair // 2, direction_right_0, direction_right_1, ground_output_0, ground_output_1)
    m_legs_right_1 = monitor_legs(legs_right_1)
    monitors_legs.append(m_legs_right_1)
    nn.add(legs_right_1)
    nn.add(m_legs_right_1)

    def plot_results():
        plot_monitor_cpg(m_cpg, sensor_back)
        plot_monitor_direction(m_direction_left_0, LEFT, sensor_right)
        plot_monitor_ground_contact(m_ground_contact_nn_0, sensor_front)
        plot_monitor_legs(monitors_legs, ['Gauche 1', 'Gauche 2', 'Droit 1', 'Droit 2'], time_offset=150)

    return nn, plot_results
