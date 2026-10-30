"""Truth-state pre-contact approach, tick-sampled supervision, and retreat."""

import math

import numpy as np

from . import RigidBodyEngine
from .frames import (
    matrix_quaternion,
    port_relative_state,
    quaternion_matrices,
    quaternion_product,
    relative_state,
)
from .scenarios import APPROACH_SCENARIOS, EARTH_RADIUS_M, MU_M3_S2, ORBIT_RADIUS_M

PHASES = {0: "hold", 1: "approach", 2: "retreat", 3: "complete"}
STATE_FIELDS = (
    "positions_m",
    "velocities_mps",
    "quaternions_xyzw",
    "angular_velocities_body_radps",
)
COMMAND_FIELDS = (
    "requested_forces_body_n",
    "requested_torques_body_nm",
    "applied_forces_body_n",
    "applied_torques_body_nm",
)


def make_approach_scenario(name):
    if name not in APPROACH_SCENARIOS:
        raise ValueError(f"unknown approach scenario: {name!r}")
    radius = ORBIT_RADIUS_M
    rate = math.sqrt(MU_M3_S2 / radius**3)
    phase = -100.0 / radius
    positions = [
        [radius, 0, 0],
        [radius * math.cos(phase), radius * math.sin(phase), 0],
    ]
    speed = rate * radius
    velocities = [[0, speed, 0], [-speed * math.sin(phase), speed * math.cos(phase), 0]]
    quaternions = [[0, 0, 0, 1], [0, 0, 0, 1]]
    body_rates = [[0, 0, rate], [0, 0, rate]]
    if name == "approach-disturbed":
        positions[1] = (np.asarray(positions[1]) + [2, 0, 1]).tolist()
        velocities[1] = (np.asarray(velocities[1]) + [0, 0.03, 0]).tolist()
        angle = math.radians(12)
        quaternions[1] = [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]
        body_rates[1] = [0.005, -0.003, rate + 0.002]
    return {
        "schema_version": 2,
        "scenario": name,
        "body_order": ["target", "chaser"],
        "mu_m3_s2": MU_M3_S2,
        "earth_radius_m": EARTH_RADIUS_M,
        "dt_ns": 100_000_000,
        "num_steps": 12_000,
        "control_period_ticks": 10,
        "initial_positions_m": positions,
        "initial_velocities_mps": velocities,
        "initial_quaternions_xyzw": quaternions,
        "initial_angular_velocities_body_radps": body_rates,
        "masses_kg": [200.0, 100.0],
        "inertias_kg_m2": [[100.0, 120.0, 140.0], [20.0, 25.0, 30.0]],
        "force_limits_n": [2.0, 2.0],
        "torque_limits_nm": [0.5, 0.5],
        "docking_ports_body_m": [[0, -1, 0], [0, 0.5, 0]],
        "port_outward_normals_body": [[0, -1, 0], [0, 1, 0]],
        "goals": {"hold_gap_m": 30.0, "final_gap_m": 5.0, "retreat_gap_m": 60.0},
        "guidance": {"speed_mps": 0.15, "acceleration_mps2": 0.005},
        "controller": {
            "position_kp_s2": 0.0016,
            "velocity_kd_s": 0.08,
            "attitude_kp_s2": 0.16,
            "rate_kd_s": 0.8,
        },
        "acceptance": {
            "position_error_m": 0.2,
            "speed_mps": 0.02,
            "alignment_deg": 2.0,
            "relative_rate_radps": 0.002,
            "dwell_ns": 10_000_000_000,
        },
        "safety": {
            "minimum_com_range_m": 3.0,
            "minimum_port_gap_m": 2.0,
            "maximum_closing_speed_mps": 0.4,
            "near_zone_gap_m": 35.0,
            "maximum_lateral_m": 5.0,
            "maximum_alignment_deg": 15.0,
        },
        "scheduled_abort_gap_m": 20.0 if name == "approach-retreat" else None,
        "observation_model": "exact current truth state, zero latency, no camera or estimator",
        "timing": (
            "control every 10 physics ticks; supervisor every tick; phase changes immediately "
            "recompute held commands; commands apply to the following interval"
        ),
        "units": {
            "position": "m",
            "velocity": "m/s",
            "time": "ns",
            "angular_velocity": "rad/s",
            "force": "N",
            "torque": "N m",
        },
        "frames": {
            "inertial": "Earth-centered fixed axes, target starts on +X moving +Y",
            "target_lvlh": "+X radial, +Y along track, +Z orbit normal",
            "quaternion": "Hamilton XYZW, active body-to-inertial rotation",
            "port_gap": (
                "negative Y component of chaser-port minus target-port "
                "in target COM-defined LVLH"
            ),
            "desired_attitude": "both body frames track target LVLH; opposing port normals",
        },
        "model": (
            "central gravity, constant mass, diagonal principal inertia, "
            "body-frame ideal norm-bounded wrench, RK4; no contact"
        ),
    }


def make_engine(manifest):
    return RigidBodyEngine(
        manifest["mu_m3_s2"],
        manifest["dt_ns"],
        manifest["initial_positions_m"],
        manifest["initial_velocities_mps"],
        manifest["initial_quaternions_xyzw"],
        manifest["initial_angular_velocities_body_radps"],
        manifest["masses_kg"],
        manifest["inertias_kg_m2"],
        manifest["force_limits_n"],
        manifest["torque_limits_nm"],
    )


def gap_profile(start, end, elapsed_s, speed, acceleration):
    """Rest-to-rest trapezoid (triangular for short transfers), in axial meters."""
    distance = abs(end - start)
    if distance == 0:
        return end, 0.0, 0.0
    direction = math.copysign(1.0, end - start)
    ramp = min(speed / acceleration, math.sqrt(distance / acceleration))
    peak = acceleration * ramp
    cruise = max(0.0, (distance - acceleration * ramp**2) / peak)
    if elapsed_s < ramp:
        moved = 0.5 * acceleration * elapsed_s**2
        velocity, acc = acceleration * elapsed_s, acceleration
    elif elapsed_s < ramp + cruise:
        moved = 0.5 * acceleration * ramp**2 + peak * (elapsed_s - ramp)
        velocity, acc = peak, 0.0
    elif elapsed_s < 2 * ramp + cruise:
        remaining = 2 * ramp + cruise - elapsed_s
        moved = distance - 0.5 * acceleration * remaining**2
        velocity, acc = acceleration * remaining, -acceleration
    else:
        return end, 0.0, 0.0
    return start + direction * moved, direction * velocity, direction * acc


class Dwell:
    """Consecutive tick-sampled validity, measured from the first valid sample."""

    def __init__(self, duration_ns):
        self.duration_ns = duration_ns
        self.started_ns = None

    def update(self, time_ns, valid):
        if not valid:
            self.started_ns = None
            return False
        if self.started_ns is None:
            self.started_ns = time_ns
        return time_ns - self.started_ns >= self.duration_ns


def state_metrics(snapshot, manifest):
    positions, velocities = np.asarray(snapshot.positions_m), np.asarray(snapshot.velocities_mps)
    q, omega = (
        np.asarray(snapshot.quaternions_xyzw),
        np.asarray(snapshot.angular_velocities_body_radps),
    )
    rho, rho_dot, basis = relative_state(positions, velocities)
    port_r, port_v, _ = port_relative_state(
        positions, velocities, q, omega, manifest["docking_ports_body_m"]
    )
    rotations = quaternion_matrices(q)
    cosine = np.clip((np.trace(rotations[0].T @ rotations[1]) - 1) / 2, -1, 1)
    inertial_rates = np.einsum("bij,bj->bi", rotations, omega)
    return {
        "rho": rho,
        "rho_dot": rho_dot,
        "basis": basis,
        "rotations": rotations,
        "port_r": port_r,
        "port_v": port_v,
        "gap": -float(port_r[1]),
        "alignment_deg": math.degrees(math.acos(float(cosine))),
        "relative_rate": float(np.linalg.norm(inertial_rates[1] - inertial_rates[0])),
    }


def commanded_wrench(snapshot, manifest, metrics, reference):
    """PD tracking with exact gravity and circular-LVLH transport compensation."""
    positions = np.asarray(snapshot.positions_m)
    velocities = np.asarray(snapshot.velocities_mps)
    quaternions = np.asarray(snapshot.quaternions_xyzw)
    body_rates = np.asarray(snapshot.angular_velocities_body_radps)
    rotations, basis = metrics["rotations"], metrics["basis"]
    rate = np.linalg.norm(np.cross(positions[0], velocities[0])) / (positions[0] @ positions[0])
    omega_l = np.array([0.0, 0.0, rate])
    gap, gap_rate, gap_acc = reference
    offset = manifest["docking_ports_body_m"][1][1] - manifest["docking_ports_body_m"][0][1]
    desired_r = np.array([0.0, -gap - offset, 0.0])
    desired_v = np.array([0.0, -gap_rate, 0.0])
    desired_a = np.array([0.0, -gap_acc, 0.0])
    gains = manifest["controller"]
    local_acc = (
        desired_a
        + gains["position_kp_s2"] * (desired_r - metrics["rho"])
        + gains["velocity_kd_s"] * (desired_v - metrics["rho_dot"])
        + 2 * np.cross(omega_l, metrics["rho_dot"])
        + np.cross(omega_l, np.cross(omega_l, metrics["rho"]))
    )
    gravity = -manifest["mu_m3_s2"] * positions / np.linalg.norm(positions, axis=-1)[:, None]**3
    forces = np.zeros((2, 3))
    force_n = manifest["masses_kg"][1] * (basis @ local_acc + gravity[0] - gravity[1])
    forces[1] = rotations[1].T @ force_n
    desired_q = matrix_quaternion(basis)
    desired_rate_n = basis @ omega_l
    torques = np.empty((2, 3))
    for body in range(2):
        conjugate = quaternions[body] * [-1, -1, -1, 1]
        error = quaternion_product(conjugate, desired_q)
        if error[3] < 0:
            error = -error
        length = np.linalg.norm(error[:3])
        rotation_error = (
            2 * error[:3]
            if length < 1e-12
            else 2 * math.atan2(length, error[3]) * error[:3] / length
        )
        desired_rate_b = rotations[body].T @ desired_rate_n
        rate_error = desired_rate_b - body_rates[body]
        desired_acc = -np.cross(body_rates[body], desired_rate_b)
        alpha = (
            gains["attitude_kp_s2"] * rotation_error
            + gains["rate_kd_s"] * rate_error
            + desired_acc
        )
        inertia = np.asarray(manifest["inertias_kg_m2"][body])
        torques[body] = inertia * alpha + np.cross(body_rates[body], inertia * body_rates[body])
    return forces, torques


def simulate_approach(manifest):
    """Run one supervised experiment; records align commands to [tick,tick+1)."""
    engine = make_engine(manifest)
    snapshot = engine.snapshot()
    initial_metrics = state_metrics(snapshot, manifest)
    phase, phase_start_ns = 0, 0
    profile_start = initial_metrics["gap"]
    goal = manifest["goals"]["hold_gap_m"]
    dwell = Dwell(manifest["acceptance"]["dwell_ns"])
    states = {field: [] for field in STATE_FIELDS}
    states.update({"tick": [], "phase": [], "desired_gap_m": []})
    commands = {field: [] for field in COMMAND_FIELDS}
    commands["control_update"] = []
    events = []
    abort_reason = None
    outcome = None
    forces, torques = np.zeros((2, 3)), np.zeros((2, 3))

    def event(kind, reason):
        events.append({
            "tick": snapshot.tick,
            "time_ns": snapshot.time_ns,
            "kind": kind,
            "reason": reason,
            "phase": phase,
        })

    event("supervisor_transition", "hold_transfer_started")
    while True:
        metrics = state_metrics(snapshot, manifest)
        safety = manifest["safety"]
        changed = False
        hard_violation = None
        if np.linalg.norm(metrics["rho"]) <= safety["minimum_com_range_m"]:
            hard_violation = "minimum_com_separation"
        elif metrics["gap"] <= safety["minimum_port_gap_m"]:
            hard_violation = "minimum_axial_port_gap"
        if hard_violation:
            event("physical_boundary", hard_violation)
            outcome = "failed"
            abort_reason = abort_reason or hard_violation
        if outcome is None and phase != 2:
            reason = None
            if metrics["port_v"][1] > safety["maximum_closing_speed_mps"]:
                reason = "closing_speed_limit"
            elif metrics["gap"] <= safety["near_zone_gap_m"]:
                if np.linalg.norm(metrics["port_r"][[0, 2]]) > safety["maximum_lateral_m"]:
                    reason = "near_zone_corridor_limit"
                elif metrics["alignment_deg"] > safety["maximum_alignment_deg"]:
                    reason = "near_zone_alignment_limit"
            abort_at = manifest["scheduled_abort_gap_m"]
            if reason:
                event("safety_detection", reason)
            elif phase == 1 and abort_at is not None and metrics["gap"] <= abort_at:
                reason = "scheduled_abort_at_20m"
                event("abort_request", reason)
            if reason:
                abort_reason = reason
                phase, phase_start_ns = 2, snapshot.time_ns
                profile_start, goal = metrics["gap"], manifest["goals"]["retreat_gap_m"]
                dwell = Dwell(manifest["acceptance"]["dwell_ns"])
                changed = True
                event("supervisor_transition", "retreat_started")
        reference = gap_profile(
            profile_start,
            goal,
            (snapshot.time_ns - phase_start_ns) / 1e9,
            manifest["guidance"]["speed_mps"],
            manifest["guidance"]["acceleration_mps2"],
        )
        acceptance = manifest["acceptance"]
        at_goal = (
            reference[0] == goal
            and reference[1] == 0
            and np.linalg.norm(metrics["port_r"] - [0, -goal, 0]) <= acceptance["position_error_m"]
            and np.linalg.norm(metrics["port_v"]) <= acceptance["speed_mps"]
            and metrics["alignment_deg"] <= acceptance["alignment_deg"]
            and metrics["relative_rate"] <= acceptance["relative_rate_radps"]
        )
        if outcome is None and dwell.update(snapshot.time_ns, at_goal):
            if phase == 0:
                event("supervisor_transition", "hold_dwell_satisfied")
                phase, phase_start_ns = 1, snapshot.time_ns
                profile_start, goal = goal, manifest["goals"]["final_gap_m"]
                dwell = Dwell(acceptance["dwell_ns"])
                reference = (profile_start, 0.0, 0.0)
                changed = True
                event("supervisor_transition", "terminal_transfer_started")
            else:
                outcome = "retreated" if phase == 2 else "success"
        if outcome is None and snapshot.tick >= manifest["num_steps"]:
            outcome = "failed"
            abort_reason = abort_reason or "time_limit"
        if outcome:
            phase = 3
            event("outcome", outcome if abort_reason is None else f"{outcome}: {abort_reason}")
        for field in STATE_FIELDS:
            states[field].append(getattr(snapshot, field))
        states["tick"].append(snapshot.tick)
        states["phase"].append(phase)
        states["desired_gap_m"].append(reference[0])
        if outcome:
            break
        update = changed or snapshot.tick % manifest["control_period_ticks"] == 0
        if update:
            forces, torques = commanded_wrench(snapshot, manifest, metrics, reference)
        try:
            next_snapshot = engine.step(forces, torques)
        except ValueError as error:
            outcome, phase = "failed", 3
            abort_reason = abort_reason or f"numerical_error: {error}"
            states["phase"][-1] = phase
            event("outcome", abort_reason)
            break
        commands["requested_forces_body_n"].append(forces.copy())
        commands["requested_torques_body_nm"].append(torques.copy())
        commands["applied_forces_body_n"].append(next_snapshot.applied_forces_body_n)
        commands["applied_torques_body_nm"].append(next_snapshot.applied_torques_body_nm)
        commands["control_update"].append(update)
        snapshot = next_snapshot
    trajectory = {key: np.asarray(value, dtype=np.float64) for key, value in states.items()}
    trajectory["tick"] = np.asarray(states["tick"], dtype=np.uint64)
    trajectory["phase"] = np.asarray(states["phase"], dtype=np.uint8)
    for key, value in commands.items():
        trajectory[key] = (
            np.asarray(value, dtype=np.bool_)
            if key == "control_update"
            else np.asarray(value, dtype=np.float64).reshape(-1, 2, 3)
        )
    achieved_dwell = (
        snapshot.time_ns - dwell.started_ns
        if outcome in ("success", "retreated") and dwell.started_ns is not None
        else 0
    )
    return trajectory, events, {
        "outcome": outcome,
        "abort_reason": abort_reason,
        "final_gap_m": metrics["gap"],
        "final_port_speed_mps": float(np.linalg.norm(metrics["port_v"])),
        "final_alignment_deg": metrics["alignment_deg"],
        "final_relative_rate_radps": metrics["relative_rate"],
        "dwell_ns": acceptance["dwell_ns"],
        "achieved_final_dwell_ns": achieved_dwell,
    }
