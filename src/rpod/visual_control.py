"""Onboard visual guidance: only own navigation, visual estimates, and mission data."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .approach import Dwell, commanded_wrench, gap_profile, state_metrics
from .frames import matrix_quaternion, quaternion_matrices, quaternion_product, relative_state
from .observations import ChaserInputs


@dataclass(frozen=True)
class MissionSpec:
    """Public vehicle/mission constants, with no initial target state or truth channel."""

    mu_m3_s2: float = 3.986004418e14
    dt_ns: int = 100_000_000
    masses_kg: tuple = (200.0, 100.0)
    inertias_kg_m2: tuple = ((100.0, 120.0, 140.0), (20.0, 25.0, 30.0))
    docking_ports_body_m: tuple = ((0.0, -1.0, 0.0), (0.0, 0.5, 0.0))
    position_kp_s2: float = 0.0016
    velocity_kd_s: float = 0.08
    attitude_kp_s2: float = 0.16
    rate_kd_s: float = 0.8
    final_gap_m: float = 5.0
    retreat_gap_m: float = 30.0
    abort_gap_m: float | None = None
    speed_mps: float = 0.15
    acceleration_mps2: float = 0.005
    dwell_ns: int = 10_000_000_000
    acquisition_deadline_ns: int = 10_000_000_000
    maximum_age_ns: int = 3_000_000_000

    def controller_config(self):
        return {
            "mu_m3_s2": self.mu_m3_s2,
            "masses_kg": self.masses_kg,
            "inertias_kg_m2": self.inertias_kg_m2,
            "docking_ports_body_m": self.docking_ports_body_m,
            "controller": {
                "position_kp_s2": self.position_kp_s2,
                "velocity_kd_s": self.velocity_kd_s,
                "attitude_kp_s2": self.attitude_kp_s2,
                "rate_kd_s": self.rate_kd_s,
            },
        }


@dataclass(frozen=True)
class ControlDecision:
    force_body_n: np.ndarray
    torque_body_nm: np.ndarray
    phase: int
    desired_gap_m: float
    goal_gap_m: float
    control_update: bool
    events: tuple
    outcome: str | None
    abort_reason: str | None
    reacquired: bool
    metrics: dict | None
    achieved_dwell_ns: int


def estimated_pair(inputs: ChaserInputs):
    """Reconstruct a pair solely from the declared onboard inputs."""
    target, own = inputs.target, inputs.own
    if target is None:
        raise ValueError("a visual target estimate is required")
    target_rotation = quaternion_matrices(target.quaternion_xyzw)
    return SimpleNamespace(
        positions_m=np.array([target.position_m, own.position_m]),
        velocities_mps=np.array([target.velocity_mps, own.velocity_mps]),
        quaternions_xyzw=np.array([target.quaternion_xyzw, own.quaternion_xyzw]),
        angular_velocities_body_radps=np.array([
            target_rotation.T @ target.angular_velocity_inertial_radps,
            own.angular_velocity_body_radps,
        ]),
    )


class VisualController:
    """Tick-sampled supervisor and held chaser wrench, without environment access."""

    def __init__(self, mission: MissionSpec):
        self.mission = mission
        self.phase = 4  # acquiring; 1 approach, 2 retreat, 5 braking, 6 lost
        self.resume_phase = 1
        self.goal = mission.final_gap_m
        self.profile_start = 30.0
        self.profile_start_ns = 0
        self.valid_count = 0
        self.last_capture_ns = None
        self.acquired = False
        self.abort_reason = None
        self.dwell = Dwell(mission.dwell_ns)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def _own_attitude_torque(self, own):
        _, _, basis = relative_state(
            [own.position_m, own.position_m], [own.velocity_mps, own.velocity_mps]
        )
        rotation = quaternion_matrices(own.quaternion_xyzw)
        error = quaternion_product(
            np.asarray(own.quaternion_xyzw) * [-1, -1, -1, 1],
            matrix_quaternion(basis),
        )
        if error[3] < 0:
            error = -error
        length = np.linalg.norm(error[:3])
        angle = (2 * error[:3] if length < 1e-12 else
                 2 * np.arctan2(length, error[3]) * error[:3] / length)
        rate_n = np.cross(own.position_m, own.velocity_mps) / np.dot(
            own.position_m, own.position_m
        )
        desired_rate = rotation.T @ rate_n
        omega = np.asarray(own.angular_velocity_body_radps)
        alpha = (self.mission.attitude_kp_s2 * angle
                 + self.mission.rate_kd_s * (desired_rate - omega)
                 - np.cross(omega, desired_rate))
        inertia = np.asarray(self.mission.inertias_kg_m2[1])
        return inertia * alpha + np.cross(omega, inertia * omega)

    def _metrics(self, pair):
        """Port and attitude metrics of the estimated pair, in target LVLH."""
        return state_metrics(pair, self.mission.controller_config())

    def _reference_complete(self, reference):
        """Completion waits for the gap profile to reach its goal at rest."""
        return reference[0] == self.goal and reference[1] == 0

    def _wrench(self, pair, metrics, reference, inputs):
        """The reference law's chaser force and torque."""
        forces, torques = commanded_wrench(
            pair, self.mission.controller_config(), metrics, reference
        )
        return forces[1], torques[1]

    def update(self, inputs: ChaserInputs) -> ControlDecision:
        now, mission = inputs.time_ns, self.mission
        if inputs.own.time_ns != now:
            raise ValueError("own navigation must match the control timestamp")
        if inputs.target is not None and inputs.target.time_ns != now:
            raise ValueError("target prediction must match the control timestamp")
        if inputs.target is not None and inputs.target.capture_time_ns > now:
            raise ValueError("future observations cannot drive control")
        events, changed, reacquired = [], False, False
        outcome = None

        def transition(phase, reason):
            nonlocal changed
            self.phase, changed = phase, True
            events.append({"kind": "supervisor_transition", "reason": reason,
                           "phase": phase, "time_ns": now,
                           "tick": now // mission.dt_ns})

        pair = estimated_pair(inputs) if inputs.target is not None else None
        metrics = self._metrics(pair) if pair is not None else None
        if (inputs.visual_update is True and inputs.target is not None
                and (self.last_capture_ns is None
                     or inputs.target.capture_time_ns > self.last_capture_ns)):
            self.valid_count += 1
            self.last_capture_ns = inputs.target.capture_time_ns
        elif inputs.visual_update is False:
            self.valid_count = 0
            self.dwell.update(now, False)
            if self.phase in (1, 2):
                self.resume_phase = self.phase
                self.profile_start = metrics["gap"] if metrics else self.profile_start
                transition(5, "visual_brake")

        if self.acquired and (self.last_capture_ns is None
                              or now - self.last_capture_ns > mission.maximum_age_ns):
            transition(6, "navigation_lost")
            outcome = "navigation_lost"
        elif self.phase == 4:
            if self.valid_count >= 3 and metrics is not None:
                self.acquired, reacquired = True, True
                self.profile_start, self.profile_start_ns = metrics["gap"], now
                transition(1, "vision_acquired")
            elif now >= mission.acquisition_deadline_ns:
                transition(6, "acquisition_deadline")
                outcome = "acquisition_failed"
        elif self.phase == 5 and self.valid_count >= 2 and metrics is not None:
            self.profile_start, self.profile_start_ns = metrics["gap"], now
            self.dwell.update(now, False)
            reacquired = True
            transition(self.resume_phase, "vision_reacquired")

        if self.phase in (1, 2) and metrics is not None:
            reason = None
            if self.phase == 1:
                if metrics["port_v"][1] > 0.4:
                    reason = "estimated_closing_speed_limit"
                elif np.linalg.norm(metrics["port_r"][[0, 2]]) > 5:
                    reason = "estimated_corridor_limit"
                elif metrics["alignment_deg"] > 15:
                    reason = "estimated_alignment_limit"
                elif mission.abort_gap_m is not None and metrics["gap"] <= mission.abort_gap_m:
                    reason = "estimated_gap_abort_request"
            if reason:
                self.abort_reason = self.abort_reason or reason
                self.goal = mission.retreat_gap_m
                self.profile_start, self.profile_start_ns = metrics["gap"], now
                self.dwell.update(now, False)
                transition(2, reason)

        if self.phase in (1, 2):
            reference = gap_profile(
                self.profile_start, self.goal, (now - self.profile_start_ns) / 1e9,
                mission.speed_mps, mission.acceleration_mps2,
            )
        else:
            reference = (self.profile_start, 0.0, 0.0)
        at_goal = (
            self.phase in (1, 2) and metrics is not None
            and inputs.target.health == "fresh"
            and now - inputs.target.capture_time_ns <= 1_200_000_000
            and self._reference_complete(reference)
            and np.linalg.norm(metrics["port_r"] - [0, -self.goal, 0]) <= 0.1
            and np.linalg.norm(metrics["port_v"]) <= 0.01
            and metrics["alignment_deg"] <= 1
            and metrics["relative_rate"] <= 0.001
        )
        if outcome is None and self.dwell.update(now, at_goal):
            outcome = "retreated" if self.phase == 2 else "success"
            transition(3, "onboard_completion")
        update = changed or (now // mission.dt_ns) % 10 == 2
        if update and outcome is None:
            if pair is None or self.phase == 4:
                self.force = np.zeros(3)
                self.torque = self._own_attitude_torque(inputs.own)
            else:
                self.force, self.torque = self._wrench(pair, metrics, reference, inputs)
        achieved = (now - self.dwell.started_ns
                    if self.dwell.started_ns is not None else 0)
        return ControlDecision(
            self.force.copy(), self.torque.copy(), self.phase, reference[0], self.goal,
            update, tuple(events), outcome, self.abort_reason, reacquired, metrics, achieved,
        )
