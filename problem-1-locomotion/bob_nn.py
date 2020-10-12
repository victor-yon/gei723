from brian2 import *

from cpg_nn import monitor_cpg, build_cpg_nn, plot_monitor_cpg
from direction_nn import build_direction_nn, monitor_direction, LEFT, plot_monitor_direction
from leg_nn import leg_nn, monitor_leg, plot_monitor_leg

DURATION = 400 * ms

# The value of Bob's sensor. Between 0 and 1 included.
SENSOR_BACK = 0.2
SENSOR_LEFT = 0.5
SENSOR_RIGHT = 0

if __name__ == '__main__':
    start_scope()

    # ======================= CPG =====================
    # The central pattern generator
    cpg_core, cpg_trigger, cpg_syn_trigger_core, cpg_syn_core = build_cpg_nn(SENSOR_BACK)
    m_cpg = monitor_cpg(cpg_core)

    # ==================== Direction ==================
    # 4 direction sub-network : 2 for each side
    # Left 0
    dir_main_l0, dir_inhib_l0, dir_syn_main_inhib_l0, dir_syn_cpg_core_l0, dir_syn_cpg_inhib_l0, \
    dir_syn_cpg_inhib_rest_l0 = build_direction_nn(cpg_core, 0, SENSOR_LEFT)
    m_dir_main_left_0, m_dir_inhib_left_0 = monitor_direction(dir_main_l0, dir_inhib_l0)

    # Left 1
    dir_main_l1, dir_inhib_l1, dir_syn_main_inhib_l1, dir_syn_cpg_core_l1, dir_syn_cpg_inhib_l1, \
    dir_syn_cpg_inhib_rest_l1 = build_direction_nn(cpg_core, 1, SENSOR_LEFT)
    # m_dir_main_left_1, m_dir_inhib_left_1 = monitor_direction(dir_main_l1, dir_inhib_l1)

    # Right 0
    dir_main_r0, dir_inhib_r0, dir_syn_main_inhib_r0, dir_syn_cpg_core_r0, dir_syn_cpg_inhib_r0, \
    dir_syn_cpg_inhib_rest_r0 = build_direction_nn(cpg_core, 0, SENSOR_RIGHT)
    # m_dir_right_0 = monitor_direction(dir_main_r0, dir_inhib_r0)

    # Right 1
    dir_main_r1, dir_inhib_r1, dir_syn_main_inhib_r1, dir_syn_cpg_core_r1, dir_syn_cpg_inhib_r1, \
    dir_syn_cpg_inhib_rest_r1 = build_direction_nn(cpg_core, 1, SENSOR_RIGHT)
    # m_dir_right_1 = monitor_direction(dir_main_r1, dir_inhib_r1)

    # ====================== Legs =====================
    leg_motors_l0, syn_cpg_motor_a_l0, syn_cpg_motor_b_l0 = leg_nn(dir_main_l0, dir_main_l1)
    m_leg_l0 = monitor_leg(leg_motors_l0)
    leg_motors_r0, syn_cpg_motor_a_r0, syn_cpg_motor_b_r0 = leg_nn(dir_main_r1, dir_main_r0)
    m_leg_r0 = monitor_leg(leg_motors_r0)

    run(DURATION)

    plot_monitor_direction(m_dir_main_left_0, m_dir_inhib_left_0, LEFT, SENSOR_LEFT)
    plot_monitor_cpg(m_cpg, SENSOR_BACK)
    plot_monitor_leg(m_leg_l0, 'gauche 0')
    plot_monitor_leg(m_leg_r0, 'droit 0')
