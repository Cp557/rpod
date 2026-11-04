"""M11: terminal approach to an unmarked, tumbling QuikSCAT.

The target turns freely. The chaser starts synchronized 30 m out on the rotating docking axis,
as after an inspection and synchronization phase, and must approach along that axis, hold
5 m off the port in the port's rotating frame, and dwell. Metrics, guidance and scoring use
the target body frame; navigation is natural-feature lidar ICP, optionally fused with camera
keypoints, in the M4 filter with torque-free rotation prediction.
"""

from functools import partial
import math
from time import perf_counter

import numpy as np

from .approach import Dwell, make_engine
from .frames import matrix_quaternion, quaternion_matrices, quaternion_product
from .lidar import LidarModel, LidarNavigation
from .natural_lidar import IcpTracker, scan_mesh
from .navigation import _STEP_NS, _propagate, _rotation_quaternion
from .observations import ChaserInputs
from .target_mesh import UnmarkedRenderer
from .vision_records import own_navigation
from .visual_control import VisualController
from .visual_env import VisualEnv
from .visual_run import _truth_estimate, initial_state

START_GAP_M = 30.0
CLEARANCE_M = 0.5
# The chaser hull, a 1 m cube, as a 5 x 5 grid on each face, for clearance checks.
_GRID = np.linspace(-0.5, 0.5, 5)
CHASER_SURFACE_BODY_M = np.unique(np.array([
    point for axis in range(3) for side in (-0.5, 0.5) for a in _GRID for b in _GRID
    for point in [np.insert([a, b], axis, side)]
]), axis=0)


def _cross(a, b):
    return np.cross(a, b)


def tumble_metrics(positions, velocities, quaternions, rates, ports):
    """`approach.state_metrics` in the target body frame instead of LVLH.

    `rho`, `port_r` and their rates are chaser minus target, in target body axes, with rates as
    seen in that rotating frame; `basis` is the target's body-to-inertial rotation.
    """
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    rotations = quaternion_matrices(quaternions)
    rates, ports = np.asarray(rates), np.asarray(ports, dtype=np.float64)
    target, omega = rotations[..., 0, :, :], rates[..., 0, :]
    lever = np.einsum("...bij,bj->...bi", rotations, ports)
    lever_rate = np.einsum("...bij,...bj->...bi", rotations, _cross(rates, ports))

    def body(vector):
        return np.einsum("...ji,...j->...i", target, vector)

    rho = body(positions[..., 1, :] - positions[..., 0, :])
    rho_dot = body(velocities[..., 1, :] - velocities[..., 0, :]) - _cross(omega, rho)
    port_r = body(positions[..., 1, :] + lever[..., 1, :] - positions[..., 0, :] - lever[..., 0, :])
    port_v = body(velocities[..., 1, :] + lever_rate[..., 1, :] - velocities[..., 0, :]
                  - lever_rate[..., 0, :]) - _cross(omega, port_r)
    trace = np.einsum("...ij,...ij->...", rotations[..., 0, :, :], rotations[..., 1, :, :])
    inertial = np.einsum("...bij,...bj->...bi", rotations, rates)
    return {
        "rho": rho, "rho_dot": rho_dot, "basis": target, "rotations": rotations,
        "port_r": port_r, "port_v": port_v, "gap": -port_r[..., 1],
        "alignment_deg": np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))),
        "relative_rate": np.linalg.norm(inertial[..., 1, :] - inertial[..., 0, :], axis=-1),
    }


def _metrics(state, ports):
    metrics = tumble_metrics(state.positions_m, state.velocities_mps, state.quaternions_xyzw,
                             state.angular_velocities_body_radps, ports)
    return {**metrics, "gap": float(metrics["gap"]),
            "alignment_deg": float(metrics["alignment_deg"]),
            "relative_rate": float(metrics["relative_rate"])}


def _attitude_error(quaternion, desired):
    error = quaternion_product(np.asarray(quaternion) * [-1, -1, -1, 1], desired)
    if error[3] < 0:
        error = -error
    length = np.linalg.norm(error[:3])
    return 2 * error[:3] if length < 1e-12 else 2 * math.atan2(length, error[3]) * error[:3] / length


def tumble_wrench(pair, mission, metrics, reference):
    """Chaser force and torque holding the rotating docking axis.

    PD tracking of the gap profile in the target body frame with rotating-frame feedforward
    (2ω×ρ̇, ω̇×ρ, ω×(ω×ρ)), the target's torque-free ω̇ from its known inertia, and the exact
    gravity difference; the chaser's attitude tracks the target's, with rate and
    acceleration feedforward.
    """
    positions, velocities = np.asarray(pair.positions_m), np.asarray(pair.velocities_mps)
    target_rotation, chaser_rotation = metrics["rotations"]
    omega = np.asarray(pair.angular_velocities_body_radps[0])
    inertia = np.asarray(mission.inertias_kg_m2[0])
    omega_dot = -_cross(omega, inertia * omega) / inertia
    gap, gap_rate, gap_acc = reference
    offset = mission.docking_ports_body_m[1][1] - mission.docking_ports_body_m[0][1]
    desired = np.array([0.0, -gap - offset, 0.0])
    rho, rho_dot = metrics["rho"], metrics["rho_dot"]
    relative = (np.array([0.0, -gap_acc, 0.0])
                + mission.position_kp_s2 * (desired - rho)
                + mission.velocity_kd_s * (np.array([0.0, -gap_rate, 0.0]) - rho_dot))
    body = (relative + 2 * _cross(omega, rho_dot) + _cross(omega_dot, rho)
            + _cross(omega, _cross(omega, rho)))
    gravity = -mission.mu_m3_s2 * positions / np.linalg.norm(positions, axis=-1)[:, None] ** 3
    force_n = mission.masses_kg[1] * (target_rotation @ body + gravity[0] - gravity[1])
    rate = np.asarray(pair.angular_velocities_body_radps[1])
    to_chaser = chaser_rotation.T @ target_rotation
    desired_rate, desired_acc = to_chaser @ omega, to_chaser @ omega_dot
    alpha = (mission.attitude_kp_s2 * _attitude_error(pair.quaternions_xyzw[1],
                                                      pair.quaternions_xyzw[0])
             + mission.rate_kd_s * (desired_rate - rate)
             - _cross(rate, desired_rate) + desired_acc)
    chaser_inertia = np.asarray(mission.inertias_kg_m2[1])
    return (chaser_rotation.T @ force_n,
            chaser_inertia * alpha + _cross(rate, chaser_inertia * rate))


def torque_free_propagate(state, time_ns, angular_noise, inertia):
    """`navigation._propagate` for a torque-free target of known principal inertia: its
    inertial angular momentum stays fixed while its rate precesses with its attitude (midpoint
    rule). The covariance keeps the constant-rate linearization; at these rates precession
    changes a 0.1 s transition by under 0.1%."""
    following = _propagate(state, time_ns, angular_noise)
    duration = (time_ns - state.time_ns) / 1e9
    rotation = quaternion_matrices(state.quaternion)
    momentum = rotation @ (inertia * (rotation.T @ state.angular_velocity))

    def rate(quaternion):
        rotation = quaternion_matrices(quaternion)
        return rotation @ (rotation.T @ momentum / inertia)

    middle = quaternion_product(_rotation_quaternion(state.angular_velocity * duration / 2),
                                state.quaternion)
    quaternion = quaternion_product(_rotation_quaternion(rate(middle) * duration),
                                    state.quaternion)
    following.quaternion = quaternion / np.linalg.norm(quaternion)
    following.angular_velocity = rate(following.quaternion)
    return following


class TumbleNavigation(LidarNavigation):
    """The M4 filter with lidar, for a freely tumbling target of known inertia.

    - Initialization: the chaser co-rotates 30 m out, up to 0.7 m/s relative, so the velocity
      prior widens from 0.1 to 1 m/s and the rate prior from 0.005 rad/s (0.3 deg/s) to
      0.02 rad/s.
    - Prediction follows torque-free rotation instead of a constant rate, so the rotation
      process noise drops from 4e-8 to 1e-10 rad^2/s^3, chosen on episodes outside the suite.
      A noisy rate estimate is a noisy rotating frame: 3e-4 rad/s of rate error is 9 mm/s of
      apparent port velocity at 30 m.
    """

    initial_velocity_sigma_mps = 1.0
    initial_rate_sigma_radps = 0.02
    rotation_spectral_density = 1e-10

    def __init__(self, camera, calibration, lidar, inertia_kg_m2):
        self.inertia = np.asarray(inertia_kg_m2, dtype=np.float64)
        super().__init__(camera, calibration, lidar)

    def _path(self, state, time_ns, high_noise_intervals=None):
        """`VisualNavigation._path` with torque-free prediction."""
        intervals = (self._high_noise_intervals if high_noise_intervals is None
                     else high_noise_intervals)
        path = []
        while state.time_ns < time_ns:
            stop = min(time_ns, state.time_ns + _STEP_NS)
            for start, end in intervals:
                if state.time_ns < start < stop:
                    stop = start
                if state.time_ns < end < stop:
                    stop = end
            boosted = any(start <= state.time_ns < end for start, end in intervals)
            state = torque_free_propagate(state, stop, self.acquisition_rotation_spectral_density
                                          if boosted else self.rotation_spectral_density,
                                          self.inertia)
            path.append(state)
        return path


class TumbleController(VisualController):
    """The reference supervisor, with metrics and guidance in the rotating target frame."""

    def _metrics(self, pair):
        return _metrics(pair, self.mission.docking_ports_body_m)

    def _wrench(self, pair, metrics, reference, inputs):
        return tumble_wrench(pair, self.mission, metrics, reference)


def tumble_initial_state(scenario, seed, *, rate_deg_s, rate_axis, chaser_disturbance=None,
                         lidar_noise_seed, **conditions):
    """The M5 initialization, then a free tumble and a synchronized chaser 30 m out on the
    rotating docking axis; the chaser disturbance applies in target body axes."""
    manifest = initial_state(scenario, seed, lidar_noise_seed=lidar_noise_seed,
                             scene="realistic", **conditions)
    positions = np.asarray(manifest["initial_positions_m"])
    velocities = np.asarray(manifest["initial_velocities_mps"])
    quaternions = np.asarray(manifest["initial_quaternions_xyzw"])
    rates = np.asarray(manifest["initial_angular_velocities_body_radps"])
    axis = np.asarray(rate_axis, dtype=np.float64)
    rates[0] = math.radians(rate_deg_s) * axis / np.linalg.norm(axis)
    rotation = quaternion_matrices(quaternions[0])
    ports = np.asarray(manifest["docking_ports_body_m"])
    offset_b = ports[0] + [0.0, -(START_GAP_M + ports[1][1]), 0.0]
    delta_v_b = np.zeros(3)
    quaternions[1], rates[1] = quaternions[0], rates[0]
    if chaser_disturbance is not None:
        offset_b = offset_b + chaser_disturbance["offset_lvlh_m"]
        delta_v_b = np.asarray(chaser_disturbance["delta_v_lvlh_mps"])
        half = math.radians(chaser_disturbance["rotation_deg"]) / 2
        quaternions[1] = quaternion_product(quaternions[0], np.r_[
            np.asarray(chaser_disturbance["rotation_axis_body"]) * math.sin(half),
            math.cos(half)])
    positions[1] = positions[0] + rotation @ offset_b
    velocities[1] = velocities[0] + rotation @ (_cross(rates[0], offset_b) + delta_v_b)
    manifest.update({
        "initial_positions_m": positions.tolist(), "initial_velocities_mps": velocities.tolist(),
        "initial_quaternions_xyzw": quaternions.tolist(),
        "initial_angular_velocities_body_radps": rates.tolist(),
        "chaser_disturbance": chaser_disturbance,
        "tumble": {"rate_deg_s": rate_deg_s, "axis_body": (axis / np.linalg.norm(axis)).tolist(),
                   "start": "synchronized on the rotating docking axis",
                   "start_gap_m": START_GAP_M},
        "observation_model": ("unmarked target: Blender RGB with keypoint pose, simulated flash "
                              "lidar with mesh ICP pose; exact own navigation"),
    })
    return manifest


class _NoRenderer:
    """Stands in for Blender in lidar-only runs."""

    provenance = {"renderer": "none: lidar-only run"}
    performance = {"startup_seconds": 0.0, "render_seconds": [], "request_seconds": [],
                   "peak_rss_bytes": None, "rss_scope": "no renderer"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class TumbleEnv(VisualEnv):
    """`VisualEnv` for the tumbling, unmarked target.

    The target gets no torque; scoring uses target-frame metrics and also ends the episode
    if the chaser hull comes within `CLEARANCE_M` of the target mesh. The lidar ray-casts the
    mesh and its own ICP tracker estimates the pose; without `keypoints` the camera is off.
    """

    def __init__(self, camera, mesh, lidar_calibration, keypoints=None):
        super().__init__(camera)
        self.mesh, self.lidar_calibration, self.keypoints = mesh, lidar_calibration, keypoints

    def reset(self, initial, output):
        lidar = initial["lidar"]
        self.tracker = IcpTracker(LidarModel(**lidar["model"]), self.mesh,
                                  self.lidar_calibration, seed=lidar["noise_seed"])
        self.minimum_clearance_m = math.inf
        return super().reset(initial, output)

    def _renderer(self):
        if self.keypoints is None:
            return _NoRenderer()
        return UnmarkedRenderer(self.output, camera=self.camera)

    def _target_command(self, snapshot):
        return np.zeros(3)

    def _scoring_metrics(self, snapshot):
        return _metrics(snapshot, self.initial["docking_ports_body_m"])

    def clearance(self, snapshot):
        rotations = quaternion_matrices(snapshot.quaternions_xyzw)
        hull = (np.asarray(snapshot.positions_m[1]) + CHASER_SURFACE_BODY_M @ rotations[1].T
                - np.asarray(snapshot.positions_m[0])) @ rotations[0]
        return float(self.mesh.closest_points(np.ascontiguousarray(hull))[2].min())

    def _boundary(self, snapshot, metrics):
        reason = super()._boundary(snapshot, metrics)
        clearance = self.clearance(snapshot)
        self.minimum_clearance_m = min(self.minimum_clearance_m, clearance)
        return reason or ("minimum_mesh_clearance" if clearance < CLEARANCE_M else None)

    def _capture_due(self, snapshot):
        return self.keypoints is not None and super()._capture_due(snapshot)

    def _image_pose(self, image):
        return self.keypoints.estimate(image)

    def _scan_ranges(self, rotation_lt, translation_lt, rng):
        return scan_mesh(rotation_lt, translation_lt, self.lidar, self.mesh, rng)

    def _lidar_pose(self, scan):
        return self.tracker.estimate(scan)


def tumble_env(mesh, lidar_calibration, keypoints=None):
    """An `env_class` for `visual_run.run_episode`."""
    return partial(TumbleEnv, mesh=mesh, lidar_calibration=lidar_calibration,
                   keypoints=keypoints)


def tumble_truth_baseline(initial, mission):
    """The paired oracle for a tumbling episode: `TumbleController` on truth, no rendering."""
    started = perf_counter()
    engine, controller = make_engine(initial), TumbleController(mission)
    physical, last_capture = Dwell(mission.dwell_ns), None
    ports = initial["docking_ports_body_m"]
    for tick in range(initial["max_steps"] + 1):
        snapshot = engine.snapshot()
        if tick % 10 == 2:
            last_capture = (tick - 2) * initial["dt_ns"]
        estimate = _truth_estimate(snapshot, last_capture) if last_capture is not None else None
        decision = controller.update(ChaserInputs(snapshot.time_ns, own_navigation(snapshot),
                                                  estimate, True if tick % 10 == 2 else None))
        metrics = _metrics(snapshot, ports)
        target = np.array([0.0, -decision.goal_gap_m, 0.0])
        valid = physical.update(snapshot.time_ns, (
            np.linalg.norm(metrics["port_r"] - target) <= 0.2
            and np.linalg.norm(metrics["port_v"]) <= 0.02
            and metrics["alignment_deg"] <= 2 and metrics["relative_rate"] <= 0.002))
        boundary = np.linalg.norm(metrics["rho"]) <= 3 or metrics["gap"] <= 2
        outcome = "physical_violation" if boundary else decision.outcome
        if outcome or tick == initial["max_steps"]:
            outcome = outcome or "time_limit"
            if outcome in ("success", "retreated") and not valid:
                outcome = "false_completion"
            return {"outcome": outcome, "elapsed_time_ns": snapshot.time_ns,
                    "wall_seconds": perf_counter() - started,
                    "model": "oracle target state, exact own navigation; no rendering",
                    "same_initialization_and_mission": True}
        engine.step([np.zeros(3), decision.force_body_n],
                    [np.zeros(3), decision.torque_body_nm])


def target_quaternion(rotation):
    return matrix_quaternion(rotation)
