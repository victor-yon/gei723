from brian2 import *

from cpg_nn import monitor_cpg, cpg_nn, plot_monitor_cpg
from leg_nn import leg_nn, monitor_leg, plot_monitor_leg

DURATION = 50 * ms

# The value of Bob's sensor. Between 0 and 1 included.
SENSOR_BACK = 0.5

if __name__ == '__main__':
    start_scope()

    # ======================= CPG =====================
    trigger, cpg_core_nn, syn_trigger_core, syn_core = cpg_nn(SENSOR_BACK)
    m_cpg = monitor_cpg(cpg_core_nn)

    # ====================== Legs =====================
    leg_motors_0, syn_cpg_motor_0 = leg_nn(cpg_core_nn, 0)
    leg_motors_1, syn_cpg_motor_1 = leg_nn(cpg_core_nn, 1)
    monitor = monitor_leg(leg_motors_0)

    run(DURATION)

    plot_monitor_cpg(m_cpg, SENSOR_BACK)
    plot_monitor_leg(monitor)
