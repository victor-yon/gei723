from brian2 import *

from cpg_nn import monitor_cpg, build_cpg_nn, plot_monitor_cpg
from direction_nn import build_direction_nn, monitor_direction, LEFT, RIGHT, plot_monitor_direction
from leg_nn import leg_nn, monitor_leg, plot_monitor_leg

DURATION = 100 * ms

# The value of Bob's sensor. Between 0 and 1 included.
SENSOR_BACK = 0
SENSOR_LEFT = 0.5
SENSOR_RIGHT = 0

if __name__ == '__main__':
    start_scope()

    # ======================= CPG =====================
    cpg_core, cpg_trigger, cpg_syn_trigger_core, cpg_syn_core = build_cpg_nn(SENSOR_BACK)
    m_cpg = monitor_cpg(cpg_core)

    # ==================== Direction ==================
    # Left
    dir_main_l, dir_inhib_l, dir_syn_main_inhib_l, dir_syn_cpg_core_l, dir_syn_cpg_inhib_l, dir_syn_cpg_inhib_rest_l = \
        build_direction_nn(cpg_core, LEFT, SENSOR_LEFT)
    m_dir_main_left, m_dir_inhib_left = monitor_direction(dir_main_l, dir_inhib_l)
    # Right
    dir_main_r, dir_inhib_r, dir_syn_main_inhib_r, dir_syn_cpg_core_r, dir_syn_cpg_inhib_r, dir_syn_cpg_inhib_rest_r = \
        build_direction_nn(cpg_core, RIGHT, SENSOR_RIGHT)
    # m_dir_right = monitor_direction(dir_main_r, dir_inhib_r)

    # ====================== Legs =====================
    leg_motors_0, syn_cpg_motor_0 = leg_nn(cpg_core, 0)
    m_leg_0 = monitor_leg(leg_motors_0)
    leg_motors_1, syn_cpg_motor_1 = leg_nn(cpg_core, 1)
    m_leg_1 = monitor_leg(leg_motors_1)

    run(DURATION)

    plot_monitor_direction(m_dir_main_left, m_dir_inhib_left, LEFT, SENSOR_LEFT)
    plot_monitor_cpg(m_cpg, SENSOR_BACK)
    plot_monitor_leg(m_leg_0, 'gauche')
    plot_monitor_leg(m_leg_1, 'droit')
