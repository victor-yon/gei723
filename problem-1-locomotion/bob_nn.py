from brian2 import *

from cpg_nn import monitor_cpg, build_cpg_nn, plot_monitor_cpg
from direction_nn import build_direction_nn
from ground_contact_nn import build_ground_contact_nn
from leg_nn import leg_nn, monitor_leg, plot_monitor_legs


def build_and_run_nn(duration: int = 300, nb_leg_pair: int = 3, sensor_back: float = 0.2, sensor_front: float = 0,
                     sensor_left: float = 0, sensor_right: float = 0):
    """
    Build the Bob neural network and run it.

    :param duration: The duration of the run in ms
    :param nb_leg_pair: The number of pair of leg
    :param sensor_back: The back sensor [0,1]
    :param sensor_front: The front sensor [0,1]
    :param sensor_left: The left sensor [0,1]
    :param sensor_right: The right sensor [0,1]
    """

    nn = Network()

    # ======================= CPG =====================
    # The central pattern generator
    cpg_core, cpg_trigger, cpg_syn_trigger_core, cpg_syn_core = build_cpg_nn(sensor_back)
    nn.add(cpg_core, cpg_trigger, cpg_syn_trigger_core, cpg_syn_core)
    m_cpg = monitor_cpg(cpg_core)
    nn.add(m_cpg)

    # ==================== Direction ==================
    # 4 direction sub-network : 2 for each side
    # Left 0
    dir_main_l0, dir_inhib_l0, dir_syn_main_inhib_l0, dir_syn_cpg_core_l0, dir_syn_cpg_inhib_l0, \
    dir_syn_cpg_inhib_rest_l0 = build_direction_nn(cpg_core, 0, sensor_left)
    nn.add(dir_main_l0, dir_inhib_l0, dir_syn_main_inhib_l0, dir_syn_cpg_core_l0, dir_syn_cpg_inhib_l0,
           dir_syn_cpg_inhib_rest_l0)
    # m_dir_main_left_0, m_dir_inhib_left_0 = monitor_direction(dir_main_l0, dir_inhib_l0)

    # Left 1
    dir_main_l1, dir_inhib_l1, dir_syn_main_inhib_l1, dir_syn_cpg_core_l1, dir_syn_cpg_inhib_l1, \
    dir_syn_cpg_inhib_rest_l1 = build_direction_nn(cpg_core, 1, sensor_left)
    nn.add(dir_main_l1, dir_inhib_l1, dir_syn_main_inhib_l1, dir_syn_cpg_core_l1, dir_syn_cpg_inhib_l1,
           dir_syn_cpg_inhib_rest_l1)
    # m_dir_main_left_1, m_dir_inhib_left_1 = monitor_direction(dir_main_l1, dir_inhib_l1)

    # Right 0
    dir_main_r0, dir_inhib_r0, dir_syn_main_inhib_r0, dir_syn_cpg_core_r0, dir_syn_cpg_inhib_r0, \
    dir_syn_cpg_inhib_rest_r0 = build_direction_nn(cpg_core, 0, sensor_right)
    nn.add(dir_main_r0, dir_inhib_r0, dir_syn_main_inhib_r0, dir_syn_cpg_core_r0, dir_syn_cpg_inhib_r0,
           dir_syn_cpg_inhib_rest_r0)
    # m_dir_right_0 = monitor_direction(dir_main_r0, dir_inhib_r0)

    # Right 1
    dir_main_r1, dir_inhib_r1, dir_syn_main_inhib_r1, dir_syn_cpg_core_r1, dir_syn_cpg_inhib_r1, \
    dir_syn_cpg_inhib_rest_r1 = build_direction_nn(cpg_core, 1, sensor_right)
    nn.add(dir_main_r1, dir_inhib_r1, dir_syn_main_inhib_r1, dir_syn_cpg_core_r1, dir_syn_cpg_inhib_r1,
           dir_syn_cpg_inhib_rest_r1)
    # m_dir_right_1 = monitor_direction(dir_main_r1, dir_inhib_r1)

    # ================= Ground Contact ================
    # 2 ground contact module (up output & down output)
    g_output_0, g_core_0, g_syn_core_motor_0, g_syn_cpg_core_0 = \
        build_ground_contact_nn(cpg_core, sensor_front, inverted=False)
    nn.add(g_output_0, g_core_0, g_syn_core_motor_0, g_syn_cpg_core_0)
    # g_state_mon_core_0, g_state_mon_output_0 = monitor_ground_contact(g_output_0, g_core_0)
    g_output_1, g_core_1, g_syn_core_motor_1, g_syn_cpg_core_1 = \
        build_ground_contact_nn(cpg_core, sensor_front, inverted=True)
    nn.add(g_output_1, g_core_1, g_syn_core_motor_1, g_syn_cpg_core_1)

    # ====================== Legs =====================
    legs_left = list()
    monitor_legs_left = list()
    legs_right = list()
    monitor_legs_right = list()
    for i in range(nb_leg_pair):

        # ----------- Left -----------
        # Left even
        if i % 2 == 0:
            leg_motors_l, syn_cpg_motor_a_l, syn_cpg_motor_b_l, syn_up_motor_l, syn_down_motor_l = \
                leg_nn(dir_main_l0, dir_main_l1, g_output_0, g_output_1)
        # Left odd
        else:
            leg_motors_l, syn_cpg_motor_a_l, syn_cpg_motor_b_l, syn_up_motor_l, syn_down_motor_l = \
                leg_nn(dir_main_l1, dir_main_l0, g_output_1, g_output_0)

        # Save left
        nn.add(leg_motors_l, syn_cpg_motor_a_l, syn_cpg_motor_b_l, syn_up_motor_l, syn_down_motor_l)
        legs_left.append([leg_motors_l, syn_cpg_motor_a_l, syn_cpg_motor_b_l, syn_up_motor_l, syn_down_motor_l])
        # Monitor the 2 first legs of each side (all other are doing the same)
        if i < 2:
            m_leg_l = monitor_leg(leg_motors_l)
            monitor_legs_left.append(m_leg_l)
            nn.add(m_leg_l)

        # ---------- Right -----------
        # Right even
        if i % 2 == 0:
            leg_motors_r, syn_cpg_motor_a_r, syn_cpg_motor_b_r, syn_up_motor_r, syn_down_motor_r = \
                leg_nn(dir_main_r1, dir_main_r0, g_output_1, g_output_0)
        # Right odd
        else:
            leg_motors_r, syn_cpg_motor_a_r, syn_cpg_motor_b_r, syn_up_motor_r, syn_down_motor_r = \
                leg_nn(dir_main_r0, dir_main_r1, g_output_0, g_output_1)

        # Save right
        nn.add(leg_motors_r, syn_cpg_motor_a_r, syn_cpg_motor_b_r, syn_up_motor_r, syn_down_motor_r)
        legs_right.append([leg_motors_r, syn_cpg_motor_a_r, syn_cpg_motor_b_r, syn_up_motor_r, syn_down_motor_r])
        # Monitor the 2 first legs of each side (all other are doing the same)
        if i < 2:
            m_leg_r = monitor_leg(leg_motors_r)
            monitor_legs_right.append(m_leg_r)
            nn.add(m_leg_r)

    nn.run(duration * ms)

    # plot_monitor_direction(m_dir_main_left_0, m_dir_inhib_left_0, LEFT, sensor_left)
    plot_monitor_cpg(m_cpg, sensor_back)
    plot_monitor_legs(monitor_legs_left + monitor_legs_right, ['Gauche 1', 'Gauche 2', 'Droit 1', 'Droit 2'],
                      time_offest=150)
    # plot_monitor_ground_contact(g_state_mon_core_0, g_state_mon_output_0, sensor_front)


if __name__ == '__main__':
    build_and_run_nn(nb_leg_pair=2, duration=350)
