"""The M11 tumbling approach in the batched fleet, for Monte Carlo campaigns.

`TumbleFleetController` and `TumbleScoring` vectorize `tumble.TumbleController` and
`TumbleEnv`'s physics and scoring; a test holds the controller equal to the scalar one. The
fleet does not check hull clearance against the target mesh, which the rendered M11 suites
do; it keeps the center-of-mass and port-gap boundaries.
"""

import numpy as np

from .fleet import (FleetController, Scoring, apply, apply_transpose, cross, norm,
                    quaternion_products, split)
from .tumble import tumble_metrics


def batch_metrics(states, ports):
    """`tumble.tumble_metrics` for (n, 2, 13) states."""
    positions, velocities, quaternions, rates = split(states)
    return tumble_metrics(positions, velocities, quaternions, rates, ports)


def tumble_wrench_batch(states, mission, m, reference):
    """Vectorized `tumble.tumble_wrench`: chaser body force and torque."""
    positions, _, quaternions, rates = split(states)
    target_rotation, chaser_rotation = m["rotations"][:, 0], m["rotations"][:, 1]
    omega = rates[:, 0]
    inertia = np.asarray(mission.inertias_kg_m2[0])
    omega_dot = -cross(omega, inertia * omega) / inertia
    gap, gap_rate, gap_acc = reference
    offset = mission.docking_ports_body_m[1][1] - mission.docking_ports_body_m[0][1]
    zeros = np.zeros_like(gap)
    rho, rho_dot = m["rho"], m["rho_dot"]
    relative = (np.stack([zeros, -gap_acc, zeros], axis=-1)
                + mission.position_kp_s2 * (np.stack([zeros, -gap - offset, zeros], -1) - rho)
                + mission.velocity_kd_s * (np.stack([zeros, -gap_rate, zeros], -1) - rho_dot))
    body = (relative + 2 * cross(omega, rho_dot) + cross(omega_dot, rho)
            + cross(omega, cross(omega, rho)))
    gravity = -mission.mu_m3_s2 * positions / norm(positions)[..., None] ** 3
    force_n = mission.masses_kg[1] * (apply(target_rotation, body)
                                      + gravity[:, 0] - gravity[:, 1])
    error = quaternion_products(quaternions[:, 1] * [-1, -1, -1, 1], quaternions[:, 0])
    error = np.where(error[:, 3:] < 0, -error, error)
    length = norm(error[:, :3])
    small = length < 1e-12
    angle = 2 * np.arctan2(length, error[:, 3])[:, None] * error[:, :3]
    attitude = np.where(small[:, None], 2 * error[:, :3],
                        angle / np.where(small, 1, length)[:, None])
    to_chaser = np.einsum("nji,njk->nik", chaser_rotation, target_rotation)
    rate = rates[:, 1]
    desired_rate, desired_acc = apply(to_chaser, omega), apply(to_chaser, omega_dot)
    alpha = (mission.attitude_kp_s2 * attitude + mission.rate_kd_s * (desired_rate - rate)
             - cross(rate, desired_rate) + desired_acc)
    chaser_inertia = np.asarray(mission.inertias_kg_m2[1])
    return (apply_transpose(chaser_rotation, force_n),
            chaser_inertia * alpha + cross(rate, chaser_inertia * rate))


class TumbleFleetController(FleetController):
    """`TumbleController` for many episodes."""

    def _metrics(self, pair):
        return batch_metrics(pair, self.mission.docking_ports_body_m)

    def _wrench(self, pair, m, reference, rows, time_ns, estimates):
        return tumble_wrench_batch(pair[rows], self.mission, {k: v[rows] for k, v in m.items()},
                                   tuple(value[rows] for value in reference))


class TumbleScoring(Scoring):
    """`TumbleEnv` scoring: port-frame metrics and a free target."""

    def metrics(self, states):
        return batch_metrics(states, self.mission.docking_ports_body_m)

    def target_torque(self, states):
        return np.zeros((len(states), 3))


def tumbling_spec(seed, index):
    """A campaign episode against the tumbling target: the operational draw, a lidar noise
    stream, and a tumble from stream [seed, index, 5], as `m11_suite.m11_specs` draws it."""
    from .monte_carlo import campaign_spec

    spec = campaign_spec(seed, index)
    rng = np.random.default_rng([seed, index, 5])
    axis = rng.normal(size=3)
    return {**spec, "streams": {**spec["streams"], "lidar_noise": [seed, index, 4],
                                "tumble": [seed, index, 5]},
            "tumble": {"rate_deg_s": float(rng.uniform(0.2, 1.0)),
                       "axis_body": (axis / np.linalg.norm(axis)).tolist()}}


def tumbling_scenario():
    from .m11_suite import episode_initial_state
    from .monte_carlo import Scenario

    return Scenario("tumbling", tumbling_spec, episode_initial_state, TumbleFleetController,
                    TumbleScoring)
