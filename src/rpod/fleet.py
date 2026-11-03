"""Vectorized reference autonomy and scoring for many episodes stepped together.

Each piece mirrors a scalar reference, and tests hold them equal:
- `metrics` and `wrench` mirror `approach.state_metrics` and `commanded_wrench`;
- `FleetController` mirrors the `VisualController.update` supervisor and held wrench;
- `run_fleet` mirrors `VisualEnv` physics, guards and scoring, with the agent's timing.
States are (n, 2, 13) arrays: per body, position, velocity, XYZW quaternion and body rate.
"""

from dataclasses import dataclass

import numpy as np

from ._core import RigidBodyBatch, pair_metrics

OUTCOMES = ("", "success", "retreated", "navigation_lost", "acquisition_failed",
            "false_completion", "physical_violation", "time_limit", "numerical_failure")
ABORT_REASONS = ("", "estimated_closing_speed_limit", "estimated_corridor_limit",
                 "estimated_alignment_limit", "estimated_gap_abort_request")
SUCCESS, RETREATED, LOST, ACQUISITION_FAILED, FALSE_COMPLETION, VIOLATION, TIME_LIMIT, \
    NUMERICAL = range(1, 9)
# `Estimates.visual_update` codes for the agent's None / False / True.
NO_UPDATE, FAILED_UPDATE, ACCEPTED_UPDATE = -1, 0, 1


def split(states):
    """Positions, velocities, XYZW quaternions and body rates of (n, 2, 13) states."""
    return states[..., :3], states[..., 3:6], states[..., 6:10], states[..., 10:]


def manifest_state(manifest):
    """One (2, 13) state row from a manifest's initial state."""
    return np.concatenate([np.asarray(manifest[key], dtype=np.float64) for key in (
        "initial_positions_m", "initial_velocities_mps", "initial_quaternions_xyzw",
        "initial_angular_velocities_body_radps",
    )], axis=-1)


def cross(a, b):
    """Cross products of (..., 3) arrays, without `np.cross` axis bookkeeping."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def norm(vectors):
    """Euclidean norms over the last axis."""
    return np.sqrt(np.sum(vectors * vectors, axis=-1))


def apply(matrices, vectors):
    """`matrices @ vectors` over leading axes."""
    return np.sum(matrices * vectors[..., None, :], axis=-1)


def apply_transpose(matrices, vectors):
    """`matrices.T @ vectors` over leading axes."""
    return np.sum(matrices * vectors[..., :, None], axis=-2)


def rotation_matrices(quaternions):
    """`frames.quaternion_matrices` without its validation, for trusted (..., 4) arrays."""
    q = quaternions / norm(quaternions)[..., None]
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def lvlh(positions, velocities):
    """LVLH bases (radial, along-track, normal columns) and frame rates of (n, 3) orbits."""
    radius = norm(positions)
    momentum = cross(positions, velocities)
    momentum_norm = norm(momentum)
    radial = positions / radius[:, None]
    normal = momentum / momentum_norm[:, None]
    rate = np.zeros_like(positions)
    rate[:, 2] = momentum_norm / radius**2
    return np.stack((radial, cross(normal, radial), normal), axis=-1), rate


def quaternion_products(first, second):
    """Hamilton products of (..., 4) XYZW quaternions."""
    vector = (first[..., 3:] * second[..., :3] + second[..., 3:] * first[..., :3]
              + cross(first[..., :3], second[..., :3]))
    scalar = first[..., 3] * second[..., 3] - np.sum(first[..., :3] * second[..., :3], axis=-1)
    return np.concatenate([vector, scalar[..., None]], axis=-1)


def matrix_quaternions(rotations):
    """Vectorized `frames.matrix_quaternion`, with the same branch per matrix."""
    r = np.asarray(rotations, dtype=np.float64)
    quaternions = np.empty(r.shape[:-2] + (4,))
    trace = np.trace(r, axis1=-2, axis2=-1)
    positive = trace > 0
    scale = 2 * np.sqrt(np.where(positive, trace, 0) + 1)
    quaternions[positive] = np.stack([
        (r[..., 2, 1] - r[..., 1, 2]) / scale, (r[..., 0, 2] - r[..., 2, 0]) / scale,
        (r[..., 1, 0] - r[..., 0, 1]) / scale, scale / 4,
    ], axis=-1)[positive]
    index = np.argmax(np.diagonal(r, axis1=-2, axis2=-1), axis=-1)
    for first in range(3):
        rows = ~positive & (index == first)
        if not rows.any():
            continue
        second, third = (first + 1) % 3, (first + 2) % 3
        m = r[rows]
        scale = 2 * np.sqrt(1 + m[:, first, first] - m[:, second, second] - m[:, third, third])
        values = np.empty((len(m), 4))
        values[:, first] = scale / 4
        values[:, second] = (m[:, first, second] + m[:, second, first]) / scale
        values[:, third] = (m[:, first, third] + m[:, third, first]) / scale
        values[:, 3] = (m[:, third, second] - m[:, second, third]) / scale
        quaternions[rows] = values
    return quaternions / norm(quaternions)[..., None]


def metrics(states, ports):
    """Vectorized `approach.state_metrics` for (n, 2, 13) states, computed in Rust."""
    values = pair_metrics(np.ascontiguousarray(states, dtype=np.float64), ports)
    count = len(values)
    return {
        "rho": values[:, :3], "rho_dot": values[:, 3:6],
        "basis": values[:, 6:15].reshape(count, 3, 3),
        "rotations": values[:, 15:33].reshape(count, 2, 3, 3),
        "port_r": values[:, 33:36], "port_v": values[:, 36:39], "gap": -values[:, 34],
        "alignment_deg": values[:, 39], "relative_rate": values[:, 40],
    }


def gap_profile(start, end, elapsed_s, speed, acceleration):
    """Vectorized `approach.gap_profile`."""
    start, end, elapsed_s = np.broadcast_arrays(*map(np.asarray, (start, end, elapsed_s)))
    distance = np.abs(end - start)
    direction = np.copysign(1.0, end - start)
    ramp = np.minimum(speed / acceleration, np.sqrt(distance / acceleration))
    peak = acceleration * ramp
    cruise = np.maximum(0.0, (distance - acceleration * ramp**2) / np.where(peak > 0, peak, 1))
    remaining = 2 * ramp + cruise - elapsed_s
    rising, cruising = elapsed_s < ramp, elapsed_s < ramp + cruise
    falling = elapsed_s < 2 * ramp + cruise
    moved = np.select([rising, cruising, falling], [
        0.5 * acceleration * elapsed_s**2,
        0.5 * acceleration * ramp**2 + peak * (elapsed_s - ramp),
        distance - 0.5 * acceleration * remaining**2,
    ])
    velocity = np.select([rising, cruising, falling],
                         [acceleration * elapsed_s, peak, acceleration * remaining])
    acc = np.select([rising, cruising, falling], [acceleration, 0.0, -acceleration])
    moving = (distance > 0) & falling
    return (np.where(moving, start + direction * moved, end),
            np.where(moving, direction * velocity, 0.0), np.where(moving, direction * acc, 0.0))


def attitude_torque(quaternions, body_rates, desired_q, desired_rate_n, rotations, inertia,
                    kp, kd):
    """The attitude half of `commanded_wrench` for one body of each pair."""
    error = quaternion_products(quaternions * [-1, -1, -1, 1], desired_q)
    error = np.where(error[:, 3:] < 0, -error, error)
    length = norm(error[:, :3])
    small = length < 1e-12
    angle = 2 * np.arctan2(length, error[:, 3])[:, None] * error[:, :3]
    rotation_error = np.where(small[:, None], 2 * error[:, :3],
                              angle / np.where(small, 1, length)[:, None])
    desired_rate_b = apply_transpose(rotations, desired_rate_n)
    alpha = (kp * rotation_error + kd * (desired_rate_b - body_rates)
             - cross(body_rates, desired_rate_b))
    inertia = np.asarray(inertia)
    return inertia * alpha + cross(body_rates, inertia * body_rates)


def wrench(states, mission, m, reference):
    """Vectorized `commanded_wrench`: (chaser body force, target torque, chaser torque)."""
    positions, velocities, quaternions, rates = split(states)
    rotations, basis = m["rotations"], m["basis"]
    rate = norm(cross(positions[:, 0], velocities[:, 0])) / np.sum(positions[:, 0] ** 2, axis=-1)
    omega_l = np.zeros((len(states), 3))
    omega_l[:, 2] = rate
    gap, gap_rate, gap_acc = reference
    ports = mission.docking_ports_body_m
    zeros = np.zeros_like(gap)
    desired_r = np.stack([zeros, -gap - (ports[1][1] - ports[0][1]), zeros], axis=-1)
    desired_v = np.stack([zeros, -gap_rate, zeros], axis=-1)
    desired_a = np.stack([zeros, -gap_acc, zeros], axis=-1)
    local_acc = (
        desired_a
        + mission.position_kp_s2 * (desired_r - m["rho"])
        + mission.velocity_kd_s * (desired_v - m["rho_dot"])
        + 2 * cross(omega_l, m["rho_dot"])
        + cross(omega_l, cross(omega_l, m["rho"]))
    )
    gravity = -mission.mu_m3_s2 * positions / norm(positions)[..., None] ** 3
    force_n = mission.masses_kg[1] * (apply(basis, local_acc) + gravity[:, 0] - gravity[:, 1])
    force_b = apply_transpose(rotations[:, 1], force_n)
    desired_q = matrix_quaternions(basis)
    desired_rate_n = apply(basis, omega_l)
    torques = [attitude_torque(quaternions[:, body], rates[:, body], desired_q, desired_rate_n,
                               rotations[:, body], mission.inertias_kg_m2[body],
                               mission.attitude_kp_s2, mission.rate_kd_s) for body in (0, 1)]
    return force_b, torques[0], torques[1]


def own_attitude_torque(own_states, mission):
    """Vectorized `VisualController._own_attitude_torque` for (n, 13) chaser states."""
    position, velocity, quaternion, rate = (own_states[:, :3], own_states[:, 3:6],
                                            own_states[:, 6:10], own_states[:, 10:])
    basis, _ = lvlh(position, velocity)
    rate_n = cross(position, velocity) / np.sum(position * position, axis=-1)[:, None]
    return attitude_torque(quaternion, rate, matrix_quaternions(basis), rate_n,
                           rotation_matrices(quaternion), mission.inertias_kg_m2[1],
                           mission.attitude_kp_s2, mission.rate_kd_s)


def physical_valid(m, goal):
    """Vectorized `visual_env._physical_valid`."""
    target = np.zeros((len(m["gap"]), 3))
    target[:, 1] = -goal
    return ((norm(m["port_r"] - target) <= 0.2) & (norm(m["port_v"]) <= 0.02)
            & (m["alignment_deg"] <= 2) & (m["relative_rate"] <= 0.002))


def hard_guard(m):
    """Vectorized `visual_env._hard_guard`: True where the pair is physically unsafe."""
    return (norm(m["rho"]) <= 3) | (m["gap"] <= 2)


@dataclass(frozen=True)
class Estimates:
    """One tick of onboard navigation for every episode.

    `target` holds (n, 13) estimated target states laid out as `split` expects, so with a body
    angular velocity (`NavigationEstimate` holds an inertial one). Rows without an estimate are
    ignored.
    """

    target: np.ndarray
    available: np.ndarray
    capture_time_ns: np.ndarray
    fresh: np.ndarray
    visual_update: np.ndarray


class Dwells:
    """Vectorized `approach.Dwell`."""

    def __init__(self, count, duration_ns):
        self.duration_ns = duration_ns
        self.started_ns = np.full(count, -1, dtype=np.int64)

    def update(self, time_ns, valid, rows=None):
        """Update `rows` (all by default); return where the dwell is complete."""
        rows = np.ones(len(valid), bool) if rows is None else rows
        self.started_ns[rows & ~valid] = -1
        self.started_ns[rows & valid & (self.started_ns < 0)] = time_ns
        return rows & valid & (time_ns - self.started_ns >= self.duration_ns)

    def clear(self, rows):
        self.started_ns[rows] = -1


class FleetController:
    """`VisualController` for many episodes: the same supervisor and held wrench.

    `abort_gaps_m` holds each episode's `MissionSpec.abort_gap_m`, NaN for None.
    """

    def __init__(self, mission, count, abort_gaps_m=None):
        self.mission, self.count = mission, count
        self.abort_gaps_m = (np.full(count, np.nan) if abort_gaps_m is None
                             else np.asarray(abort_gaps_m, dtype=np.float64))
        self.phase = np.full(count, 4, dtype=np.int8)
        self.resume_phase = np.ones(count, dtype=np.int8)
        self.goal = np.full(count, mission.final_gap_m)
        self.profile_start = np.full(count, 30.0)
        self.profile_start_ns = np.zeros(count, dtype=np.int64)
        self.valid_count = np.zeros(count, dtype=np.int64)
        self.last_capture_ns = np.full(count, -1, dtype=np.int64)
        self.acquired = np.zeros(count, bool)
        self.abort_reason = np.zeros(count, dtype=np.int8)
        self.dwell = Dwells(count, mission.dwell_ns)
        self.force = np.zeros((count, 3))
        self.torque = np.zeros((count, 3))
        self.brakes = np.zeros(count, dtype=np.int64)

    def _wrench(self, pair, m, reference, rows):
        """Chaser force and torque where `rows` is set; the reference law by default."""
        force, _, torque = wrench(pair[rows], self.mission, {k: v[rows] for k, v in m.items()},
                                  tuple(value[rows] for value in reference))
        return force, torque

    def _reference_complete(self, reference):
        return (reference[0] == self.goal) & (reference[1] == 0)

    def update(self, time_ns, own, estimates, active):
        """Advance every active episode one tick; return each episode's claim code (0 none)."""
        mission = self.mission
        has = estimates.available & active
        # `visual_control.estimated_pair`; rows without an estimate get a harmless placeholder
        # pair whose metrics go unused.
        pair = np.stack([np.where(has[:, None], estimates.target, own), own], axis=1)
        m = metrics(pair, mission.docking_ports_body_m)
        gap = m["gap"]
        changed = np.zeros(self.count, bool)
        outcome = np.zeros(self.count, dtype=np.int8)

        def transition(rows, phase):
            self.phase[rows] = phase
            changed[rows] = True

        visual = estimates.visual_update
        newer = has & (visual == ACCEPTED_UPDATE) & (
            (self.last_capture_ns < 0) | (estimates.capture_time_ns > self.last_capture_ns))
        self.valid_count[newer] += 1
        self.last_capture_ns[newer] = estimates.capture_time_ns[newer]
        failed = active & (visual == FAILED_UPDATE)
        self.valid_count[failed] = 0
        self.dwell.clear(failed)
        brake = failed & np.isin(self.phase, (1, 2))
        self.resume_phase[brake] = self.phase[brake]
        self.profile_start[brake & has] = gap[brake & has]
        self.brakes[brake] += 1
        transition(brake, 5)

        lost = active & self.acquired & (
            (self.last_capture_ns < 0) | (time_ns - self.last_capture_ns > mission.maximum_age_ns))
        transition(lost, 6)
        outcome[lost] = LOST
        acquiring = active & ~lost & (self.phase == 4)
        acquire = acquiring & (self.valid_count >= 3) & has
        self.acquired[acquire] = True
        self.profile_start[acquire], self.profile_start_ns[acquire] = gap[acquire], time_ns
        transition(acquire, 1)
        deadline = acquiring & ~acquire & (time_ns >= mission.acquisition_deadline_ns)
        transition(deadline, 6)
        outcome[deadline] = ACQUISITION_FAILED
        reacquire = active & ~lost & ~acquiring & (self.phase == 5) & (self.valid_count >= 2) & has
        self.profile_start[reacquire], self.profile_start_ns[reacquire] = gap[reacquire], time_ns
        self.dwell.clear(reacquire)
        transition(reacquire, self.resume_phase[reacquire])

        approaching = active & has & (self.phase == 1)
        lateral = norm(m["port_r"][:, [0, 2]])
        reasons = np.select([
            m["port_v"][:, 1] > 0.4, lateral > 5, m["alignment_deg"] > 15,
            ~np.isnan(self.abort_gaps_m) & (gap <= np.nan_to_num(self.abort_gaps_m)),
        ], [1, 2, 3, 4], 0)
        abort = approaching & (reasons > 0)
        self.abort_reason[abort & (self.abort_reason == 0)] = reasons[abort & (self.abort_reason == 0)]
        self.goal[abort] = mission.retreat_gap_m
        self.profile_start[abort], self.profile_start_ns[abort] = gap[abort], time_ns
        self.dwell.clear(abort)
        transition(abort, 2)

        moving = np.isin(self.phase, (1, 2))
        profile = gap_profile(self.profile_start, self.goal,
                              (time_ns - self.profile_start_ns) / 1e9,
                              mission.speed_mps, mission.acceleration_mps2)
        reference = tuple(np.where(moving, value, hold)
                          for value, hold in zip(profile, (self.profile_start, 0.0, 0.0)))
        target = np.zeros((self.count, 3))
        target[:, 1] = -self.goal
        at_goal = (
            moving & has & estimates.fresh
            & (time_ns - estimates.capture_time_ns <= 1_200_000_000)
            & self._reference_complete(reference)
            & (norm(m["port_r"] - target) <= 0.1) & (norm(m["port_v"]) <= 0.01)
            & (m["alignment_deg"] <= 1) & (m["relative_rate"] <= 0.001)
        )
        complete = self.dwell.update(time_ns, at_goal, rows=active & (outcome == 0))
        outcome[complete] = np.where(self.phase[complete] == 2, RETREATED, SUCCESS)
        transition(complete, 3)

        update = active & (outcome == 0) & (changed | ((time_ns // mission.dt_ns) % 10 == 2))
        own_rows = update & (~has | (self.phase == 4))
        self.force[own_rows] = 0
        if own_rows.any():
            self.torque[own_rows] = own_attitude_torque(own[own_rows], mission)
        law_rows = update & ~own_rows
        if law_rows.any():
            self.force[law_rows], self.torque[law_rows] = self._wrench(pair, m, reference,
                                                                       law_rows)
        return outcome


class TruthNavigation:
    """The paired-oracle estimate: current truth, as if captured at the latest delivery."""

    def __init__(self, count, dt_ns):
        self.dt_ns = dt_ns
        self.capture_time_ns = np.full(count, -1, dtype=np.int64)

    def update(self, tick, states, active):
        delivered = tick % 10 == 2
        if delivered:
            self.capture_time_ns[:] = (tick - 2) * self.dt_ns
        count = len(states)
        return Estimates(states[:, 0], self.capture_time_ns >= 0, self.capture_time_ns.copy(),
                         np.ones(count, bool),
                         np.full(count, ACCEPTED_UPDATE if delivered else NO_UPDATE))


def target_torque(states, mission):
    """Vectorized `visual_env._target_torque`: hold target LVLH (reference gap irrelevant)."""
    m = metrics(states, mission.docking_ports_body_m)
    count = len(states)
    return wrench(states, mission, m, (np.full(count, 30.0), np.zeros(count),
                                       np.zeros(count)))[1]


def run_fleet(initials, mission, navigation, controller=None, threads=0):
    """Run every initial manifest to its end, as `VisualEnv` scores one episode.

    `navigation.update(tick, states, active)` returns `Estimates`. Returns per-episode arrays.
    """
    count = len(initials)
    first = initials[0]
    states = np.stack([manifest_state(initial) for initial in initials])
    batch = RigidBodyBatch(states, first["mu_m3_s2"], first["dt_ns"], first["masses_kg"],
                           first["inertias_kg_m2"], first["force_limits_n"],
                           first["torque_limits_nm"], threads=threads)
    controller = controller or FleetController(mission, count)
    dt_ns, max_steps = first["dt_ns"], first["max_steps"]
    chaser_mass, force_limit = first["masses_kg"][1], first["force_limits_n"][1]
    success_dwell, retreat_dwell = Dwells(count, mission.dwell_ns), Dwells(count, mission.dwell_ns)
    active = np.ones(count, bool)
    result = {
        "outcome": np.zeros(count, dtype=np.int8), "end_tick": np.zeros(count, dtype=np.int64),
        "minimum_port_gap_m": np.full(count, np.inf), "minimum_com_range_m": np.full(count, np.inf),
        "final_port_speed_mps": np.zeros(count), "final_gap_m": np.zeros(count),
        "delta_v_mps": np.zeros(count),
    }
    target_torques = np.zeros((count, 3))

    def finish(rows, codes, tick, m):
        rows = rows & active
        result["outcome"][rows] = np.broadcast_to(codes, (count,))[rows]
        result["end_tick"][rows] = tick
        result["final_port_speed_mps"][rows] = norm(m["port_v"][rows])
        result["final_gap_m"][rows] = m["gap"][rows]
        active[rows] = False

    for tick in range(max_steps + 1):
        time_ns = tick * dt_ns
        states = batch.states()
        truth = metrics(states, mission.docking_ports_body_m)
        result["minimum_port_gap_m"][active] = np.minimum(result["minimum_port_gap_m"],
                                                          truth["gap"])[active]
        result["minimum_com_range_m"][active] = np.minimum(
            result["minimum_com_range_m"], norm(truth["rho"]))[active]
        finish(hard_guard(truth), VIOLATION, tick, truth)
        if not active.any():
            break
        claims = controller.update(time_ns, states[:, 1], navigation.update(tick, states, active),
                                   active)
        valid = {SUCCESS: success_dwell.update(time_ns, physical_valid(truth, mission.final_gap_m)),
                 RETREATED: retreat_dwell.update(time_ns,
                                                 physical_valid(truth, mission.retreat_gap_m))}
        for code, satisfied in valid.items():
            finish((claims == code) & ~satisfied, FALSE_COMPLETION, tick, truth)
        finish(claims > 0, claims, tick, truth)
        if tick == max_steps:
            finish(active, TIME_LIMIT, tick, truth)
        if not active.any():
            break
        if tick % first["control_period_ticks"] == first["control_offset_ticks"]:
            target_torques = target_torque(states, mission)
        forces = np.zeros((count, 2, 3))
        forces[:, 1] = controller.force
        torques = np.stack([target_torques, controller.torque], axis=1)
        applied = np.minimum(norm(controller.force), force_limit)
        result["delta_v_mps"][active] += applied[active] / chaser_mass * dt_ns * 1e-9
        status = batch.step(forces, torques, active)
        finish(status > 0, NUMERICAL, tick, truth)
    result["abort_reason"] = controller.abort_reason
    result["brakes"] = controller.brakes
    return result
