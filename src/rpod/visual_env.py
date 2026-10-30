"""Visual RPOD environment: truth, camera, optional lidar, and scoring behind a one-tick
command interface."""

from pathlib import Path
from time import perf_counter

import numpy as np

from .approach import (
    COMMAND_FIELDS, STATE_FIELDS, Dwell, commanded_wrench, make_engine, state_metrics,
)
from .camera import camera_pose, earth_pose, relative_target_pose
from .lidar import LidarModel, estimate_lidar_pose, scan_target
from .observations import (
    CLAIMS, Delivery, EnvObservation, ImageObservation, LidarDelivery, LidarScan, PoseEstimate,
)
from .perception import estimate_pose
from .vision_records import (
    degrade_image, json_value, own_navigation, read_rgb, sha256_file, write_rgb,
)

OWN_FIELDS = (
    "own_position_m", "own_velocity_mps", "own_quaternion_xyzw",
    "own_angular_velocity_body_radps",
)


def _target_torque(snapshot, manifest):
    """The environment's target controller is separate from chaser autonomy."""
    metrics = state_metrics(snapshot, manifest)
    _, torques = commanded_wrench(snapshot, manifest, metrics, (30.0, 0.0, 0.0))
    return torques[0]


def _physical_valid(metrics, goal):
    return (
        np.linalg.norm(metrics["port_r"] - [0, -goal, 0]) <= 0.2
        and np.linalg.norm(metrics["port_v"]) <= 0.02
        and metrics["alignment_deg"] <= 2
        and metrics["relative_rate"] <= 0.002
    )


def _hard_guard(metrics):
    if np.linalg.norm(metrics["rho"]) <= 3:
        return "minimum_com_separation"
    if metrics["gap"] <= 2:
        return "minimum_axial_port_gap"
    return None


class VisualEnv:
    """One episode per reset. Clients see only EnvObservation and return ChaserCommand.

    Each step holds the commanded chaser wrench for one physics tick. Target truth,
    hidden initialization, faults, and physical scoring stay inside the environment.
    """

    def __init__(self, camera):
        self.camera = camera
        self.engine = self.renderer = None
        self._entered = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._entered:
            self._entered = False
            self.renderer.__exit__(*exc)

    def reset(self, initial, output) -> EnvObservation | None:
        from .rendering import BlenderRenderer, realistic_scene

        self.__exit__(None, None, None)
        self.engine = None
        self.initial, self.output = initial, Path(output)
        self.engine = make_engine(initial)
        mission = initial["mission"]
        self.goals = {"success": mission["final_gap_m"], "retreated": mission["retreat_gap_m"]}
        self.dwells = {claim: Dwell(mission["dwell_ns"]) for claim in self.goals}
        self.sunlight_n = np.asarray(initial["sunlight_direction_inertial"])
        self.states = {field: [] for field in (*STATE_FIELDS, "tick", *OWN_FIELDS)}
        self.commands = {field: [] for field in COMMAND_FIELDS}
        self.observations, self.events, self.pending, self.perception_seconds = [], [], [], []
        lidar = initial.get("lidar")
        self.lidar = LidarModel(**lidar["model"]) if lidar else None
        self.lidar_rows, self.lidar_pending, self.lidar_seconds = [], [], []
        self.target_torque = np.zeros(3)
        self.done, self.outcome, self.failure_reason, self.physical_dwell_ns = False, None, None, 0
        (self.output / "images").mkdir(parents=True, exist_ok=True)
        if self.lidar:
            (self.output / "lidar").mkdir(exist_ok=True)
        self.realistic = initial.get("scene") == "realistic"
        self.renderer = BlenderRenderer(self.output, camera=self.camera,
                                        scene=realistic_scene() if self.realistic else None)
        self.renderer.__enter__()
        self._entered = True
        return self._observe()

    def step(self, command) -> EnvObservation | None:
        """Apply one command; return the next observation, or None when the episode ends."""
        if self.done:
            raise RuntimeError("the episode has ended; reset before stepping")
        force = np.asarray(command.force_body_n, dtype=np.float64)
        torque = np.asarray(command.torque_body_nm, dtype=np.float64)
        if (force.shape != (3,) or torque.shape != (3,)
                or not np.isfinite(force).all() or not np.isfinite(torque).all()):
            raise ValueError("chaser force and torque must be finite three-vectors")
        if command.claim is not None and command.claim not in CLAIMS:
            raise ValueError(f"unknown chaser claim: {command.claim!r}")
        snapshot, tick = self.snapshot, int(self.snapshot.tick)
        valid = {claim: dwell.update(snapshot.time_ns,
                                     _physical_valid(self.metrics, self.goals[claim]))
                 for claim, dwell in self.dwells.items()}
        if command.claim in valid and not valid[command.claim]:
            self._finish("false_completion", "physical_dwell_failed")
            return None
        if command.claim or tick == self.initial["max_steps"]:
            self._finish(command.claim or "time_limit")
            return None
        if tick % self.initial["control_period_ticks"] == self.initial["control_offset_ticks"]:
            self.target_torque = _target_torque(snapshot, self.initial)
        forces = np.array([np.zeros(3), force])
        torques = np.array([self.target_torque, torque])
        try:
            after = self.engine.step(forces, torques)
        except ValueError as error:
            self._finish("numerical_failure", str(error))
            return None
        self.commands["requested_forces_body_n"].append(forces)
        self.commands["requested_torques_body_nm"].append(torques)
        self.commands["applied_forces_body_n"].append(after.applied_forces_body_n)
        self.commands["applied_torques_body_nm"].append(after.applied_torques_body_nm)
        return self._observe()

    def fail(self, reason):
        """End the episode after a renderer or pipeline error, keeping the initial state."""
        if not self.states["tick"]:
            self._record(self.engine.snapshot())
        self.done, self.outcome, self.failure_reason = True, "render_or_pipeline_failure", reason

    def _finish(self, outcome, reason=None):
        self.done, self.outcome, self.failure_reason = True, outcome, reason
        dwell = self.dwells.get(outcome)
        if dwell is not None and dwell.started_ns is not None:
            self.physical_dwell_ns = self.snapshot.time_ns - dwell.started_ns

    def _record(self, snapshot):
        own = own_navigation(snapshot)
        for field in STATE_FIELDS:
            self.states[field].append(getattr(snapshot, field))
        self.states["tick"].append(snapshot.tick)
        for field, value in zip(OWN_FIELDS, (own.position_m, own.velocity_mps,
                                             own.quaternion_xyzw, own.angular_velocity_body_radps)):
            self.states[field].append(value)
        return own

    def _observe(self):
        snapshot = self.engine.snapshot()
        own = self._record(snapshot)
        self.snapshot, self.metrics = snapshot, state_metrics(snapshot, self.initial)
        reason = _hard_guard(self.metrics)
        if reason:
            self.events.append({"tick": snapshot.tick, "time_ns": snapshot.time_ns,
                                "kind": "physical_boundary", "reason": reason})
            self._finish("physical_violation", reason)
            return None
        if snapshot.tick % self.initial["camera_period_ticks"] == 0:
            self._capture(snapshot, own)
        if self.lidar and snapshot.tick % self.lidar.period_ticks == self.lidar.offset_ticks:
            self._scan(snapshot, own)
        deliveries = []
        while self.pending and self.pending[0][0].delivery_time_ns <= snapshot.time_ns:
            image, captured_own, row = self.pending.pop(0)
            if image.capture_time_ns > snapshot.time_ns:
                raise ValueError("attempted delivery of a future image")
            if image.pixels is not None:
                started = perf_counter()
                pose = estimate_pose(image, self.camera)
                self.perception_seconds.append(perf_counter() - started)
            else:
                pose = PoseEstimate(image.capture_time_ns, image.delivery_time_ns, False,
                                    reason="missing_capture")
            row.update({"actual_delivery_time_ns": snapshot.time_ns,
                        "pose_estimate": json_value(pose)})
            deliveries.append(Delivery(image, pose, captured_own))
        scans = []
        while self.lidar_pending and self.lidar_pending[0][0].delivery_time_ns <= snapshot.time_ns:
            scan, captured_own, row = self.lidar_pending.pop(0)
            started = perf_counter()
            pose = estimate_lidar_pose(scan, self.lidar)
            self.lidar_seconds.append(perf_counter() - started)
            row.update({"actual_delivery_time_ns": snapshot.time_ns,
                        "pose_estimate": json_value(pose)})
            scans.append(LidarDelivery(scan, pose, captured_own))
        return EnvObservation(snapshot.time_ns, own, tuple(deliveries), tuple(scans))

    def _capture(self, snapshot, own):
        initial = self.initial
        capture_id = snapshot.tick // initial["camera_period_ticks"]
        scheduled = snapshot.time_ns + initial["delivery_delay_ticks"] * initial["dt_ns"]
        missing = capture_id in initial["loss_schedule"]["missing_capture_s"]
        black = capture_id in initial["loss_schedule"]["black_capture_s"]
        image_path = self.output / "images" / f"{capture_id:06d}.png"
        row = {"capture_id": capture_id, "capture_time_ns": snapshot.time_ns,
               "scheduled_delivery_time_ns": scheduled, "actual_delivery_time_ns": None,
               "image_path": None, "image_sha256": None,
               "fault": "missing_capture" if missing else "black_frame" if black else None}
        pixels = None
        if not missing:
            rotation_ct, translation_ct = relative_target_pose(
                own, snapshot.positions_m[0], snapshot.quaternions_xyzw[0], self.camera
            )
            light_direction_c = camera_pose(own, self.camera)[1].T @ self.sunlight_n
            row["light_direction_c"] = light_direction_c.tolist()
            row["render_metadata"] = self.renderer.render(
                rotation_ct, translation_ct, image_path, light_direction_c=light_direction_c,
                earth=earth_pose(own, self.camera) if self.realistic else None,
            )
            pixels = read_rgb(image_path)
            degradation = initial["image_degradation"]
            if black:
                pixels = np.zeros_like(pixels)
                write_rgb(image_path, pixels)
            elif degradation is not None:
                pixels = degrade_image(pixels, [*degradation["noise_seed"], capture_id],
                                       degradation["noise_dn"], degradation["blur_px"])
                write_rgb(image_path, pixels)
            row["image_path"] = image_path.relative_to(self.output).as_posix()
            row["image_sha256"] = sha256_file(image_path)
        self.observations.append(row)
        self.pending.append((ImageObservation(capture_id, snapshot.time_ns, scheduled, pixels),
                             own, row))

    def _scan(self, snapshot, own):
        lidar, initial = self.lidar, self.initial
        capture_id = snapshot.tick // lidar.period_ticks
        scheduled = snapshot.time_ns + lidar.delivery_delay_ticks * initial["dt_ns"]
        rotation_lt, translation_lt = relative_target_pose(
            own, snapshot.positions_m[0], snapshot.quaternions_xyzw[0], lidar
        )
        rng = np.random.default_rng([*initial["lidar"]["noise_seed"], capture_id])
        ranges, retro = scan_target(rotation_lt, translation_lt, lidar, rng)
        scan_path = self.output / "lidar" / f"{capture_id:06d}.npz"
        np.savez_compressed(scan_path, ranges_m=ranges, retroreflective=retro)
        row = {"capture_id": capture_id, "capture_time_ns": snapshot.time_ns,
               "scheduled_delivery_time_ns": scheduled, "actual_delivery_time_ns": None,
               "scan_path": scan_path.relative_to(self.output).as_posix(),
               "scan_sha256": sha256_file(scan_path),
               "returns": int(np.isfinite(ranges).sum()),
               "retroreflective_returns": int(retro.sum())}
        self.lidar_rows.append(row)
        self.lidar_pending.append(
            (LidarScan(capture_id, snapshot.time_ns, scheduled, ranges, retro), own, row)
        )
