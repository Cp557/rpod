"""M13: docking with the tumbling target.

The chaser carries one flash lidar on its docking axis, 90 degrees wide (512x512), so the whole
docking face stays in view down to its minimum range, a 1.3 m gap; closer in, a fit to the face
alone has a competing minimum a few centimeters off. Inside 2.5 m the lidar tracker holds the
relative attitude, which the face cannot pin down, and fits translation only, while the
navigation filter, now with little translation process noise, predicts attitude from the
torque-free model. From the 5 m hold the chaser flies a terminal segment to contact in the
rotating port frame, commits inside 1.3 m (sensor failures no longer brake it) and closes the
last 1.24 m on prediction. At contact the environment checks the capture envelope: inside it,
a soft-capture latch joins the bodies, conserving momentum (`StackEngine`); outside it, the
episode ends as a physical violation. Once latched, the chaser detumbles the stack with its
own torque.
"""

from dataclasses import replace
from functools import partial
import math
from types import SimpleNamespace

import numpy as np

from ._core import RigidBodyEngine
from .approach import Dwell
from .frames import matrix_quaternion, quaternion_matrices
from .lidar import LidarModel
from .lidar_faults import CamTumbleController, CoastingIcpTracker
from .natural_lidar import (HUBER_M, ITERATIONS, MINIMUM_INLIER_FRACTION, MINIMUM_POINTS, POINTS,
                            RMS_LIMIT_M, Track, _fit_quality, _Scan, exponential, rotation_vector,
                            scan_mesh)
from .navigation import _POSITION, _STEP_NS, _VELOCITY
from .observations import LidarPose
from .tumble import CHASER_SURFACE_BODY_M, TumbleEnv, TumbleNavigation, torque_free_propagate
from .visual_control import ControlDecision, estimated_pair

DOCKING_LIDAR = {"width": 512, "height": 512, "hfov_deg": 90.0,
                 "mount_position_body_m": (0.0, 0.5, 0.0)}
# The hatch stands 6.3 cm proud of the target port, so an aligned chaser face touches it there.
CONTACT_GAP_M = 0.063
# Inside this gap the lidar sees too little of the face to fit, and the chaser commits: it
# closes the last 1.24 m (about 31 s) on its torque-free prediction.
COMMIT_GAP_M = 1.3
# Capture envelope at contact, in the target port frame.
CAPTURE = {"lateral_m": 0.05, "alignment_deg": 3.0, "closing_mps": (0.01, 0.06),
           "lateral_speed_mps": 0.02, "relative_rate_radps": 0.0035}
# Terminal segment: a slow closing profile whose goal lies past contact, so contact comes at
# cruise speed, with stiffer gains (natural frequency 0.1 rad/s, critically damped), and an
# estimate age limit that covers the committed coast.
TERMINAL = {"final_gap_m": -0.3, "speed_mps": 0.04, "acceleration_mps2": 0.002,
            "position_kp_s2": 0.01, "velocity_kd_s": 0.2,
            "maximum_age_ns": 45_000_000_000}
# Detumble: rate damping at the chaser's 0.5 N m, done below this stack rate for the dwell.
DETUMBLE_GAIN_NMS, DETUMBLED_RADPS, DETUMBLED_PHYSICAL_RADPS = 100.0, 0.002, 0.004
# Before capture, hull points away from the docking interface must keep this clearance.
INTERFACE_HALF_WIDTH_M, CONTACT_CLEARANCE_M = 0.25, 0.02
_INTERFACE = ((CHASER_SURFACE_BODY_M[:, 1] == 0.5)
              & (np.abs(CHASER_SURFACE_BODY_M[:, 0]) <= INTERFACE_HALF_WIDTH_M)
              & (np.abs(CHASER_SURFACE_BODY_M[:, 2]) <= INTERFACE_HALF_WIDTH_M))
OUTER_HULL_BODY_M = CHASER_SURFACE_BODY_M[~_INTERFACE]
# Lidar-to-target-origin distance bins; the tracker reports nothing inside MINIMUM_RANGE_M.
CALIBRATION_BINS_M = (3.0, 4.5, 6.5, 8.45, 15.95, 25.95, 31.95, 1e9)
MINIMUM_RANGE_M = 1.0 + COMMIT_GAP_M
# Inside this lidar-to-target-center distance (a 2.5 m gap) the tracker holds the relative
# attitude and fits translation only; at 3-4.5 m a free fit still measures attitude to about
# 0.1 degrees.
ATTITUDE_HOLD_RANGE_M = 3.5
TRANSLATION_ONLY = [0, 1, 2]
UNINFORMATIVE_ATTITUDE_RAD = 1.0
# Per-axis variance floor under the calibration: M11's lidar level (0.8 cm, 0.15 degrees).
COVARIANCE_FLOOR = [0.008 ** 2] * 3 + [math.radians(0.15) ** 2] * 3
# The translation process noise `navigation._propagate` adds (m^2/s^3).
M4_TRANSLATION_SPECTRAL_DENSITY = 2.5e-5
# Inside this lidar-to-target-center distance (a 6 m gap) the docking filter lowers it.
LOW_NOISE_RANGE_M = 7.0


def docking_lidar(model=None):
    """The docking lidar, from M11's lidar settings (period, noise, mount rotation)."""
    return LidarModel(**{**(model or {}), **DOCKING_LIDAR})


def icp_free(points_l, mesh, rotation, translation, iterations, free):
    """`natural_lidar.icp` over the free parameters only: translation (0-2) and rotation (3-5)
    in lidar axes."""
    for _ in range(iterations):
        body = (points_l - translation) @ rotation
        closest, normals, _ = mesh.closest_points(np.ascontiguousarray(body))
        model_l, normals_l = closest @ rotation.T, normals @ rotation.T
        residual = np.sum(normals_l * (points_l - model_l - translation), axis=1)
        weight = np.minimum(1.0, HUBER_M / np.maximum(np.abs(residual), 1e-12))
        jacobian = np.hstack([normals_l, np.cross(model_l, normals_l)])[:, free]
        hessian = jacobian.T @ (jacobian * weight[:, None])
        step = np.zeros(6)
        step[free] = np.linalg.solve(hessian + 1e-9 * np.eye(len(free)),
                                     jacobian.T @ (weight * residual))
        rotation = exponential(step[3:]) @ rotation
        translation = translation + step[:3]
        if np.linalg.norm(step[:3]) < 1e-5 and np.linalg.norm(step[3:]) < 1e-6:
            break
    body = (points_l - translation) @ rotation
    return rotation, translation, mesh.closest_points(np.ascontiguousarray(body))[2]


class DockingIcpTracker(CoastingIcpTracker):
    """`CoastingIcpTracker` for the docking lidar.

    - Inside `ATTITUDE_HOLD_RANGE_M` of the target's center only the docking face is in view:
      roll about the boresight is unobservable, and a tilt about the target's center, a meter
      behind the face, trades against a sideways shift. A free fit lets attitude wander and
      drags translation by centimeters. There the tracker holds the relative attitude at its
      last value (the chaser's attitude loop keeps it nearly constant), fits translation only,
      reports attitude as uninformative so the filter keeps its torque-free prediction, and
      keeps its track through a poor fit, since a fresh acquisition is ambiguous that close.
    - Inside the minimum range, judged by its track or, without one, by the scan's median
      return, it reports nothing.
    - A fit without inliers reports no RMS.
    """

    def _covariance(self, distance_m):
        # Far out the fits have heavier tails than their mean squares suggest (4-6 sigma
        # excursions at 30 m), and the filter's attitude prediction is still settling to tenths of
        # a degree in its first minute; without a floor the gate refuses good scans.
        return np.maximum(super()._covariance(distance_m), np.diag(COVARIANCE_FLOOR))

    def estimate(self, scan):
        capture, delivery = scan.capture_time_ns, scan.delivery_time_ns
        points = (self.lidar.directions() * scan.ranges_m[..., None]).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)].astype(np.float64)
        previous = self.previous
        distance = np.linalg.norm(previous.translation) if previous is not None else math.inf
        if distance < MINIMUM_RANGE_M or (len(points) and np.median(
                np.linalg.norm(points, axis=1)) < COMMIT_GAP_M):
            return LidarPose(capture, delivery, False, reason="below_minimum_range")
        if distance >= ATTITUDE_HOLD_RANGE_M or len(points) < MINIMUM_POINTS:
            pose = super().estimate(scan)
            return replace(pose, rms_m=None) if pose.rms_m == math.inf else pose
        rng = np.random.default_rng([*self.seed, scan.capture_id])
        sample = points[rng.choice(len(points), min(POINTS, len(points)), replace=False)]
        centroid = points.mean(axis=0)
        translation = previous.translation + centroid - previous.centroid
        rotation, translation, residual = icp_free(sample, self.mesh, previous.rotation,
                                                   translation, ITERATIONS, TRANSLATION_ONLY)
        rms, fraction = _fit_quality(residual)
        if rms > RMS_LIMIT_M or fraction < MINIMUM_INLIER_FRACTION:
            return LidarPose(capture, delivery, False, reason="icp_mismatch",
                             rms_m=rms if math.isfinite(rms) else None)
        self.previous = Track(rotation, translation, centroid, np.zeros(3), capture)
        # The held attitude is no measurement: the chaser turns with the filter's attitude, so
        # fusing it would let that attitude walk.
        covariance = self._covariance(float(np.linalg.norm(translation)))
        covariance[3:, 3:] = np.eye(3) * UNINFORMATIVE_ATTITUDE_RAD ** 2
        return LidarPose(capture, delivery, True, rotation, translation, covariance, rms_m=rms)


def calibrate_docking_lidar(lidar, mesh, samples, seed):
    """Per-bin mean-square errors of tracked docking-lidar poses against truth.

    The samples follow the approach, which always looks along the docking axis: the chaser on
    the port axis to within 0.1 m plus 1% of the gap and 3 degrees, the target at any roll about
    it. Half the gaps lie between the minimum range and 5.5 m, half between 5.5 and 33 m. Priors
    are off by up to 1 degree and 0.05 m inside 5.5 m, 3 degrees and 0.15 m farther out.
    """
    rng = np.random.default_rng(seed)
    rotation_bc = np.asarray(lidar.rotation_bc)
    mount = np.asarray(lidar.mount_position_body_m)
    errors = {index: [] for index in range(len(CALIBRATION_BINS_M))}
    for sample in range(samples):
        near = sample % 2 == 0
        gap = rng.uniform(COMMIT_GAP_M, 5.5) if near else rng.uniform(5.5, 33.0)
        roll = rng.uniform(0, 2 * math.pi)
        tilt = rng.normal(size=3) * [1, 0, 1]
        tilt *= math.radians(rng.uniform(0, 3)) / max(np.linalg.norm(tilt), 1e-12)
        lateral = rng.uniform(-1, 1, 3) * [1, 0, 1] * (0.1 + 0.01 * gap)
        body = exponential(tilt) @ exponential(np.array([0, roll, 0]))
        rotation = rotation_bc.T @ body
        translation = rotation_bc.T @ (np.array([0.0, 1.5 + gap, 0.0]) + lateral - mount)
        prior_deg, prior_m = (1.0, 0.05) if near else (3.0, 0.15)
        ranges, _ = scan_mesh(rotation, translation, lidar, mesh, rng)
        points = (lidar.directions() * ranges[..., None]).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            continue
        tracker = DockingIcpTracker(lidar, mesh, {"range_bins_m": list(CALIBRATION_BINS_M),
                                                  "variance": [[1.0] * 6]
                                                  * len(CALIBRATION_BINS_M)})
        nudge = rng.normal(size=3)
        tracker.previous = Track(
            exponential(nudge / np.linalg.norm(nudge) * math.radians(rng.uniform(0, prior_deg)))
            @ rotation, translation + rng.uniform(-prior_m, prior_m, 3), points.mean(axis=0),
            np.zeros(3), 0)
        pose = tracker.estimate(_Scan(int(rng.integers(1 << 30)), ranges))
        distance = float(np.linalg.norm(translation))
        index = min(int(np.searchsorted(CALIBRATION_BINS_M, distance)),
                    len(CALIBRATION_BINS_M) - 1)
        if pose.valid:
            errors[index].append(np.r_[pose.translation_lt - translation,
                                       rotation_vector(pose.rotation_lt @ rotation.T)])
    pooled = np.array([row for rows in errors.values() for row in rows])
    variance, counts = [], []
    for index in range(len(CALIBRATION_BINS_M)):
        rows = np.array(errors[index])
        counts.append(len(rows))
        variance.append(np.mean((pooled if len(rows) < 30 else rows) ** 2, axis=0).tolist())
    return {"artifact_type": "docking_lidar_calibration",
            "range_bins_m": list(CALIBRATION_BINS_M), "variance": variance, "samples": counts,
            "seed": seed, "lidar": DOCKING_LIDAR,
            "parameterization": "lidar axes [t_lt error, left attitude error of R_lt]"}


def capture_conditions(metrics):
    """The contact state the capture envelope judges, from target port-frame truth."""
    return {"lateral_m": float(np.linalg.norm(metrics["port_r"][[0, 2]])),
            "alignment_deg": float(metrics["alignment_deg"]),
            "closing_mps": float(metrics["port_v"][1]),
            "lateral_speed_mps": float(np.linalg.norm(metrics["port_v"][[0, 2]])),
            "relative_rate_radps": float(metrics["relative_rate"])}


def inside_envelope(conditions):
    low, high = CAPTURE["closing_mps"]
    return (conditions["lateral_m"] <= CAPTURE["lateral_m"]
            and conditions["alignment_deg"] <= CAPTURE["alignment_deg"]
            and low <= conditions["closing_mps"] <= high
            and conditions["lateral_speed_mps"] <= CAPTURE["lateral_speed_mps"]
            and conditions["relative_rate_radps"] <= CAPTURE["relative_rate_radps"])


class DockingNavigation(TumbleNavigation):
    """`TumbleNavigation` with far less translation process noise near the target.

    The target's translation is exactly modeled (central gravity). With M4's translation noise
    the velocity estimate follows measurement noise by a few mm/s, and the committed coast to
    contact integrates that into centimeters of lateral error. Far out, though, M4's noise is
    what lets the filter converge from its wide priors without settling on an early error and
    refusing good scans. So the noise drops to `translation_spectral_density` once the last
    accepted lidar fix is within `LOW_NOISE_RANGE_M` of the target's center: through the hold,
    the terminal segment and the coast. `navigation._propagate` adds M4's noise on every step;
    each step here removes the excess, which is exact because process noise enters the
    covariance additively.
    """

    translation_spectral_density = 2.5e-8

    def __init__(self, camera, calibration, lidar, inertia_kg_m2):
        super().__init__(camera, calibration, lidar, inertia_kg_m2)
        self.distance_m = math.inf

    def observe(self, pose, own_at_capture):
        accepted = super().observe(pose, own_at_capture)
        if accepted and isinstance(pose, LidarPose):
            self.distance_m = float(np.linalg.norm(pose.translation_lt))
        return accepted

    def _path(self, state, time_ns, high_noise_intervals=None):
        intervals = (self._high_noise_intervals if high_noise_intervals is None
                     else high_noise_intervals)
        excess = (M4_TRANSLATION_SPECTRAL_DENSITY - self.translation_spectral_density
                  if self.distance_m < LOW_NOISE_RANGE_M else 0.0)
        path = []
        while state.time_ns < time_ns:
            stop = min(time_ns, state.time_ns + _STEP_NS)
            for start, end in intervals:
                if state.time_ns < start < stop:
                    stop = start
                if state.time_ns < end < stop:
                    stop = end
            boosted = any(start <= state.time_ns < end for start, end in intervals)
            duration = (stop - state.time_ns) / 1e9
            state = torque_free_propagate(state, stop, self.acquisition_rotation_spectral_density
                                          if boosted else self.rotation_spectral_density,
                                          self.inertia)
            covariance = state.covariance
            covariance[_POSITION, _POSITION] -= np.eye(3) * excess * duration ** 3 / 3
            covariance[_POSITION, _VELOCITY] -= np.eye(3) * excess * duration ** 2 / 2
            covariance[_VELOCITY, _POSITION] -= np.eye(3) * excess * duration ** 2 / 2
            covariance[_VELOCITY, _VELOCITY] -= np.eye(3) * excess * duration
            path.append(state)
        return path


class DockingMechanism:
    """Soft capture: the latch status the docking controller reads, and the contact the
    environment measured."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.latched, self.contact, self.latch_tick = False, None, None


def _bounded(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


class StackEngine:
    """The latched pair as one rigid body, conserving linear and angular momentum at the
    latch. A fresh engine steps the stack about its principal axes (twice, as its two bodies);
    snapshots place both spacecraft rigidly on it, on the original clock, so the environment
    and its records carry on unchanged. Wrenches are the chaser's, bounded by its limits."""

    def __init__(self, snapshot, manifest):
        positions = np.asarray(snapshot.positions_m, dtype=np.float64)
        velocities = np.asarray(snapshot.velocities_mps, dtype=np.float64)
        rotations = quaternion_matrices(np.asarray(snapshot.quaternions_xyzw))
        rates = np.asarray(snapshot.angular_velocities_body_radps, dtype=np.float64)
        masses = np.asarray(manifest["masses_kg"], dtype=np.float64)
        inertias = np.asarray(manifest["inertias_kg_m2"], dtype=np.float64)
        total = masses.sum()
        center, velocity = masses @ positions / total, masses @ velocities / total
        offsets = positions - center
        inertia, momentum = np.zeros((3, 3)), np.zeros(3)
        for body in range(2):
            local = rotations[body] @ np.diag(inertias[body]) @ rotations[body].T
            inertia += local + masses[body] * (offsets[body] @ offsets[body] * np.eye(3)
                                               - np.outer(offsets[body], offsets[body]))
            momentum += (local @ (rotations[body] @ rates[body])
                         + masses[body] * np.cross(offsets[body], velocities[body] - velocity))
        omega = np.linalg.solve(inertia, momentum)
        principal, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]
        self.offsets = offsets @ axes                       # body centers in stack axes
        self.attitudes = np.einsum("ji,bjk->bik", axes, rotations)  # body -> stack axes
        self.limits = (np.asarray(manifest["force_limits_n"], dtype=np.float64),
                       np.asarray(manifest["torque_limits_nm"], dtype=np.float64))
        lever = float(np.linalg.norm(self.offsets[1]))
        stack_torque_limit = float(self.limits[1][1] + lever * self.limits[0][1]) + 1e-9
        quaternion, rate = matrix_quaternion(axes), axes.T @ omega
        self.engine = RigidBodyEngine(
            manifest["mu_m3_s2"], manifest["dt_ns"], [center.tolist()] * 2,
            [velocity.tolist()] * 2, [quaternion.tolist()] * 2, [rate.tolist()] * 2,
            [float(total)] * 2, [principal.tolist()] * 2, [float(self.limits[0][1])] * 2,
            [stack_torque_limit] * 2)
        self.tick, self.time_ns = int(snapshot.tick), int(snapshot.time_ns)
        self.applied = (np.zeros((2, 3)), np.zeros((2, 3)))

    def snapshot(self):
        stack = self.engine.snapshot()
        rotation = quaternion_matrices(np.asarray(stack.quaternions_xyzw[0]))
        omega = rotation @ np.asarray(stack.angular_velocities_body_radps[0])
        center, velocity = np.asarray(stack.positions_m[0]), np.asarray(stack.velocities_mps[0])
        arms = self.offsets @ rotation.T
        rotations = np.einsum("ij,bjk->bik", rotation, self.attitudes)
        return SimpleNamespace(
            tick=self.tick + int(stack.tick), time_ns=self.time_ns + int(stack.time_ns),
            positions_m=center + arms, velocities_mps=velocity + np.cross(omega, arms),
            quaternions_xyzw=np.array([matrix_quaternion(matrix) for matrix in rotations]),
            angular_velocities_body_radps=np.einsum("bji,j->bi", rotations, omega),
            applied_forces_body_n=self.applied[0], applied_torques_body_nm=self.applied[1])

    def step(self, forces_body_n, torques_body_nm):
        forces = np.array([_bounded(np.asarray(force, dtype=np.float64), limit)
                           for force, limit in zip(forces_body_n, self.limits[0])])
        torques = np.array([_bounded(np.asarray(torque, dtype=np.float64), limit)
                            for torque, limit in zip(torques_body_nm, self.limits[1])])
        force = np.einsum("bij,bj->i", self.attitudes, forces)
        torque = (np.einsum("bij,bj->i", self.attitudes, torques)
                  + np.cross(self.offsets, np.einsum("bij,bj->bi", self.attitudes,
                                                     forces)).sum(axis=0))
        self.engine.step([force.tolist()] * 2, [torque.tolist()] * 2)
        self.applied = (forces, torques)
        return self.snapshot()


class DockingEnv(TumbleEnv):
    """`TumbleEnv` through contact: the docking lidar and its tracker, a capture-envelope check
    at contact that latches the bodies (`StackEngine`) or ends the episode, and, as "success",
    a latched stack turning slower than `DETUMBLED_PHYSICAL_RADPS`."""

    def __init__(self, camera, mesh, lidar_calibration, keypoints=None, mechanism=None):
        super().__init__(camera, mesh, lidar_calibration, keypoints)
        self.mechanism = mechanism

    def reset(self, initial, output):
        self.mechanism.reset()
        observation = super().reset(initial, output)
        # No scan is due at tick 0, so the replacement precedes the first scan.
        self.tracker = DockingIcpTracker(self.tracker.lidar, self.mesh, self.lidar_calibration,
                                         seed=self.tracker.seed)
        return observation

    def outer_clearance(self, snapshot):
        rotations = quaternion_matrices(snapshot.quaternions_xyzw)
        hull = (np.asarray(snapshot.positions_m[1]) + OUTER_HULL_BODY_M @ rotations[1].T
                - np.asarray(snapshot.positions_m[0])) @ rotations[0]
        return float(self.mesh.closest_points(np.ascontiguousarray(hull))[2].min())

    def _boundary(self, snapshot, metrics):
        if self.mechanism.latched:
            return None
        clearance = self.outer_clearance(snapshot)
        self.minimum_clearance_m = min(self.minimum_clearance_m, clearance)
        if metrics["gap"] <= CONTACT_GAP_M:
            self.mechanism.contact = {"tick": int(snapshot.tick), **capture_conditions(metrics)}
            if not inside_envelope(self.mechanism.contact):
                return "contact_outside_capture_envelope"
            self.engine = StackEngine(snapshot, self.initial)
            self.mechanism.latched, self.mechanism.latch_tick = True, int(snapshot.tick)
            return None
        return "minimum_mesh_clearance" if clearance < CONTACT_CLEARANCE_M else None

    def _physically_valid(self, metrics, goal):
        if goal == self.goals["success"]:
            return (self.mechanism.latched and np.linalg.norm(
                self.snapshot.angular_velocities_body_radps[0]) <= DETUMBLED_PHYSICAL_RADPS)
        return super()._physically_valid(metrics, goal)


def docking_env(mesh, lidar_calibration, mechanism, keypoints=None):
    """An `env_class` for `visual_run.run_episode`."""
    return partial(DockingEnv, mesh=mesh, lidar_calibration=lidar_calibration,
                   keypoints=keypoints, mechanism=mechanism)


class DockingController(CamTumbleController):
    """The M12 supervisor (with its collision-avoidance burn) to the 5 m hold, then a terminal
    segment to contact, then a detumble once the mechanism reports a latch."""

    def __init__(self, mission, mechanism):
        super().__init__(mission)
        self.mechanism, self.segment = mechanism, "approach"
        self.detumbled = Dwell(mission.dwell_ns)

    def _event(self, now, phase, reason):
        return {"kind": "supervisor_transition", "reason": reason, "phase": phase,
                "time_ns": now, "tick": now // self.mission.dt_ns}

    def update(self, inputs):
        if self.mechanism.latched:
            return self._detumble(inputs)
        if (self.segment == "terminal" and self.burn is None and inputs.target is not None
                and self._metrics(estimated_pair(inputs))["gap"] <= COMMIT_GAP_M):
            # Committed: the lidar is inside its minimum range, so failures no longer brake.
            inputs = replace(inputs, visual_update=None)
        decision = super().update(inputs)
        if decision.outcome != "success" or self.segment != "approach":
            return decision
        now = inputs.time_ns
        self.segment, self.goal, self.phase = "terminal", TERMINAL["final_gap_m"], 1
        self.mission = replace(self.mission, **TERMINAL)
        self.dwell = Dwell(self.mission.dwell_ns)
        self.profile_start, self.profile_start_ns = decision.metrics["gap"], now
        return replace(decision, phase=1, outcome=None, goal_gap_m=self.goal,
                       events=(self._event(now, 1, "terminal_approach"),))

    def _detumble(self, inputs):
        now, events, outcome = inputs.time_ns, (), None
        if self.segment != "captured":
            self.segment, self.phase = "captured", 0
            events = (self._event(now, 0, "captured"),)
        rate = np.asarray(inputs.own.angular_velocity_body_radps, dtype=np.float64)
        torque = _bounded(-DETUMBLE_GAIN_NMS * rate, 0.5)
        if self.detumbled.update(now, np.linalg.norm(rate) <= DETUMBLED_RADPS):
            self.phase, outcome = 3, "success"
            events += (self._event(now, 3, "detumbled"),)
        started = self.detumbled.started_ns
        achieved = now - started if started is not None else 0
        return ControlDecision(np.zeros(3), torque, self.phase, self.profile_start, self.goal,
                               True, events, outcome, self.abort_reason, False, None, achieved)

