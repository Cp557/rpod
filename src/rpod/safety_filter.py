"""Simplex safety filter for the learned approach policy, on onboard estimates.

At each control update the estimated port state is predicted one control period ahead under
the policy's LVLH acceleration, as a double integrator (the force law compensates the orbital
terms, which move the chaser well under a millimetre in 1 s). If the prediction leaves the safe
set, the reference law's hold-at-the-current-gap command is used instead. The safe set:
- closing speed at most `closing_limit(gap)`, a braking envelope that stops at half the
  chaser's thrust authority 0.25 m short of the hold point, capped at 0.3 m/s;
- lateral port offset at most 0.5 m + 0.1 × gap (the supervisor aborts at 5 m);
- gap at least 0.25 m short of the hold point.
"""

import numpy as np

HORIZON_S = 1.0
BRAKING_MPS2 = 0.01
MAXIMUM_CLOSING_MPS = 0.3
GAP_MARGIN_M = 0.25


def closing_limit(gap, goal):
    room = np.maximum(gap - goal + GAP_MARGIN_M, 0.0)
    return np.minimum(MAXIMUM_CLOSING_MPS, np.sqrt(2 * BRAKING_MPS2 * room) + 0.01)


def safe(port_r, port_v, goal):
    gap = -port_r[:, 1]
    lateral = np.sqrt(port_r[:, 0] ** 2 + port_r[:, 2] ** 2)
    return ((port_v[:, 1] <= closing_limit(gap, goal)) & (lateral <= 0.5 + 0.1 * gap)
            & (gap >= goal - GAP_MARGIN_M))


def needs_backup(m, acceleration, goal):
    """Where the policy's acceleration would leave the safe set within one control period."""
    port_r = m["port_r"] + m["port_v"] * HORIZON_S + 0.5 * acceleration * HORIZON_S**2
    port_v = m["port_v"] + acceleration * HORIZON_S
    return ~safe(port_r, port_v, goal)
