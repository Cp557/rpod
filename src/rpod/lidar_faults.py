"""M12: scheduled lidar faults on the tumbling target, and safety once navigation is lost.

`LidarFaultEnv` is `TumbleEnv` with one lidar fault. The fault starts at the first scan whose
true port gap is at most `trigger_gap_m` (the environment owns truth, so a fault can be placed
far out, mid-range or at the hold) and lasts `scans` scans:
- `dropped`: the scan is lost in transmission; its pose is invalid and the ICP tracker keeps
  its track;
- `blank`: the lidar returns nothing; M11's tracker then drops its track and must acquire
  again, while `CoastingIcpTracker` keeps it;
- `biased`: every range reads `bias_m` long, as from a timing error; ICP fits the shifted
  cloud, so only the navigation filter's innovation gate can refuse it.

`CamTumbleController` answers a navigation loss with a collision-avoidance burn planned from
its last estimate, then declares the loss. `drift_clearance` lets both bodies coast with zero
wrench and reports how close the chaser hull comes to the target mesh.
"""

from functools import partial
import json
from pathlib import Path

import numpy as np

from ._core import RigidBodyBatch
from .frames import quaternion_matrices
from .natural_lidar import IcpTracker
from .observations import LidarPose
from .tumble import (CHASER_SURFACE_BODY_M, CLEARANCE_M, TumbleController, TumbleEnv,
                     tumble_metrics)
from .visual_control import ControlDecision

# The collision-avoidance burn leaves the chaser moving away from the last estimated target
# center at CAM_SEPARATION_MPS, plus CAM_DRIFT_MPS along-track, so the relative orbit drifts
# off instead of closing again after one orbit. It uses the chaser's full 2 N.
CAM_SEPARATION_MPS, CAM_DRIFT_MPS, CAM_FORCE_N = 0.05, 0.05, 2.0
# Longer than one orbit, so a coast that comes back round to the target is caught.
DRIFT_S = 6000.0
# Hull points lie within this distance of the chaser's center of mass.
HULL_RADIUS_M = float(np.linalg.norm(CHASER_SURFACE_BODY_M, axis=1).max())
# Below this bound on hull clearance, the hull points are checked every tick.
EXACT_BELOW_M = 1.0


class CoastingIcpTracker(IcpTracker):
    """`IcpTracker` that keeps its track through an empty scan, which carries no evidence
    against it; a poor fit still drops the track."""

    def estimate(self, scan):
        previous = self.previous
        pose = super().estimate(scan)
        if pose.reason == "too_few_returns":
            self.previous = previous
        return pose


class LidarFaultEnv(TumbleEnv):
    """`TumbleEnv` whose lidar fails once, as `initial["lidar_fault"]` describes; with
    `coast_empty_scans`, its tracker is a `CoastingIcpTracker`."""

    def __init__(self, camera, mesh, lidar_calibration, keypoints=None, coast_empty_scans=False):
        super().__init__(camera, mesh, lidar_calibration, keypoints)
        self.coast_empty_scans = coast_empty_scans

    def reset(self, initial, output):
        self.fault, self.faulted, self.faulting = initial["lidar_fault"], (), False
        observation = super().reset(initial, output)
        if self.coast_empty_scans:
            # No scan is due at tick 0, so the replacement precedes the first scan.
            self.tracker = CoastingIcpTracker(self.tracker.lidar, self.mesh,
                                              self.lidar_calibration, seed=self.tracker.seed)
        return observation

    def _scan(self, snapshot, own):
        capture_id = snapshot.tick // self.lidar.period_ticks
        if not self.faulted and self.metrics["gap"] <= self.fault["trigger_gap_m"]:
            self.faulted = tuple(range(capture_id, capture_id + self.fault["scans"]))
        self.faulting = capture_id in self.faulted
        super()._scan(snapshot, own)
        if self.faulting:
            self.lidar_rows[-1]["fault"] = self.fault["kind"]

    def _scan_ranges(self, rotation_lt, translation_lt, rng):
        ranges, retro = super()._scan_ranges(rotation_lt, translation_lt, rng)
        if self.faulting and self.fault["kind"] == "blank":
            return np.full_like(ranges, np.nan), np.zeros_like(retro)
        if self.faulting and self.fault["kind"] == "biased":
            return ranges + self.fault["bias_m"], retro
        return ranges, retro

    def _lidar_pose(self, scan):
        if self.fault["kind"] == "dropped" and scan.capture_id in self.faulted:
            return LidarPose(scan.capture_time_ns, scan.delivery_time_ns, False,
                             reason="dropped_scan")
        return super()._lidar_pose(scan)


def lidar_fault_env(mesh, lidar_calibration, keypoints=None, coast_empty_scans=False):
    """An `env_class` for `visual_run.run_episode`."""
    return partial(LidarFaultEnv, mesh=mesh, lidar_calibration=lidar_calibration,
                   keypoints=keypoints, coast_empty_scans=coast_empty_scans)


def cam_burn(inputs, mission):
    """The collision-avoidance burn from onboard knowledge at a navigation loss: the inertial
    thrust direction and the time it ends."""
    own, target = inputs.own, inputs.target
    away = own.position_m - target.position_m
    radial = own.position_m / np.linalg.norm(own.position_m)
    normal = np.cross(own.position_m, own.velocity_mps)
    along = np.cross(normal / np.linalg.norm(normal), radial)
    away = away / np.linalg.norm(away)
    side = 1.0 if away @ along >= 0 else -1.0
    change = (CAM_SEPARATION_MPS * away + side * CAM_DRIFT_MPS * along
              - (own.velocity_mps - target.velocity_mps))
    seconds = np.linalg.norm(change) * mission.masses_kg[1] / CAM_FORCE_N
    return change / np.linalg.norm(change), inputs.time_ns + int(round(seconds * 1e9))


class CamTumbleController(TumbleController):
    """`TumbleController` that answers a navigation loss with `cam_burn`, holding the burn's
    inertial direction as the chaser turns, and declares the loss once the burn ends."""

    def __init__(self, mission):
        super().__init__(mission)
        self.burn = None

    def update(self, inputs):
        events = ()
        if self.burn is None:
            decision = super().update(inputs)
            if decision.outcome != "navigation_lost" or inputs.target is None:
                return decision
            self.burn, events = cam_burn(inputs, self.mission), decision.events
        direction, end_ns = self.burn
        done = inputs.time_ns >= end_ns
        rotation = quaternion_matrices(np.asarray(inputs.own.quaternion_xyzw))
        force = np.zeros(3) if done else CAM_FORCE_N * rotation.T @ direction
        return ControlDecision(force, np.zeros(3), 6, self.profile_start, self.goal, True, events,
                               "navigation_lost" if done else None, self.abort_reason, False,
                               None, 0)


def hull_clearance(mesh, positions, quaternions):
    """Closest distance from the chaser hull to the target mesh, for (k, 2, 3) positions and
    (k, 2, 4) quaternions."""
    rotations = quaternion_matrices(quaternions)
    hull = (positions[:, 1, None] + np.einsum("kij,pj->kpi", rotations[:, 1], CHASER_SURFACE_BODY_M)
            - positions[:, 0, None])
    hull = np.einsum("kji,kpj->kpi", rotations[:, 0], hull)
    distances = mesh.closest_points(np.ascontiguousarray(hull.reshape(-1, 3)))[2]
    return distances.reshape(len(positions), -1).min(axis=1)


def drift_clearance(manifest, states, mesh, seconds=DRIFT_S, threads=0):
    """Coast every (2, 13) state with zero wrench for `seconds`; per state, the hull's closest
    approach to the mesh, when it happened, and the closest center-of-mass range.

    Far from the mesh the clearance is bounded below by the chaser center's distance to the
    mesh less the hull radius, so the hull points are only checked once that bound is small;
    the reported minimum is therefore exact when it is below `EXACT_BELOW_M` and conservative
    otherwise.
    """
    states = np.ascontiguousarray(states, dtype=np.float64)
    count = len(states)
    batch = RigidBodyBatch(states, manifest["mu_m3_s2"], manifest["dt_ns"], manifest["masses_kg"],
                           manifest["inertias_kg_m2"], manifest["force_limits_n"],
                           manifest["torque_limits_nm"], threads=threads)
    zeros, active = np.zeros((count, 2, 3)), np.ones(count, bool)
    minimum, when = np.full(count, np.inf), np.zeros(count)
    closest_range = np.full(count, np.inf)
    dt_s = manifest["dt_ns"] / 1e9
    for tick in range(int(round(seconds / dt_s)) + 1):
        current = batch.states()
        positions, quaternions = current[:, :, 0:3], current[:, :, 6:10]
        offset = positions[:, 1] - positions[:, 0]
        closest_range = np.minimum(closest_range, np.linalg.norm(offset, axis=1))
        center = np.einsum("kji,kj->ki", quaternion_matrices(quaternions[:, 0]), offset)
        clearance = mesh.closest_points(np.ascontiguousarray(center))[2] - HULL_RADIUS_M
        near = clearance < EXACT_BELOW_M
        if near.any():
            clearance[near] = hull_clearance(mesh, positions[near], quaternions[near])
        closer = clearance < minimum
        minimum[closer], when[closer] = clearance[closer], tick * dt_s
        if tick * dt_s < seconds:
            status = batch.step(zeros, zeros, active)
            if np.any(status):
                raise ValueError("the coasting engine failed")
    return {"minimum_clearance_m": minimum, "time_of_minimum_s": when,
            "minimum_com_range_m": closest_range}


def drift_from(manifest, trajectory, tick, mesh, seconds=DRIFT_S):
    """`drift_clearance` from one recorded tick of an episode, as a summary dict."""
    state = np.concatenate([np.asarray(trajectory[field][tick]) for field in (
        "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps")],
        axis=-1)
    result = drift_clearance(manifest, state[None], mesh, seconds)
    return {"seconds": seconds, **{key: float(value[0]) for key, value in result.items()},
            "passively_safe": bool(result["minimum_clearance_m"][0] >= CLEARANCE_M)}


def passive_safety_map(run_directory, mesh, every_s=10.0, seconds=DRIFT_S, threads=0):
    """Coast from every `every_s` seconds of each operational episode in a tumbling suite run,
    as if all control stopped then; the hull's closest approach to the mesh, by port gap."""
    samples, states, manifest = [], [], None
    for path in sorted((Path(run_directory) / "episodes").iterdir()):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["m5_episode"]["category"] != "operational":
            continue
        with np.load(path / "trajectory.npz") as data:
            trajectory = {key: data[key] for key in data.files}
        every = max(1, int(round(every_s * 1e9 / manifest["dt_ns"])))
        ticks = np.arange(0, len(trajectory["positions_m"]), every)
        fields = [trajectory[key][ticks] for key in (
            "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps")]
        gaps = tumble_metrics(*fields, manifest["docking_ports_body_m"])["gap"]
        states.append(np.concatenate(fields, axis=-1))
        samples += [{"episode": path.name, "time_s": float(tick * manifest["dt_ns"] / 1e9),
                     "gap_m": float(gap)} for tick, gap in zip(ticks, gaps)]
    result = drift_clearance(manifest, np.concatenate(states), mesh, seconds, threads)
    for index, sample in enumerate(samples):
        sample.update({key: float(value[index]) for key, value in result.items()})
    bins, lower = [], 0.0
    for upper in (6.0, 8.0, 12.0, 18.0, 25.0, 1e9):
        rows = [row for row in samples if lower <= row["gap_m"] < upper]
        if rows:
            bins.append({"gap_m": [lower, upper], "samples": len(rows),
                         "minimum_clearance_m": min(row["minimum_clearance_m"] for row in rows),
                         "passively_safe": sum(row["minimum_clearance_m"] >= CLEARANCE_M
                                               for row in rows)})
        lower = upper
    return {"artifact_type": "m12_passive_safety_map", "run": Path(run_directory).as_posix(),
            "every_s": every_s, "drift_s": seconds, "clearance_margin_m": CLEARANCE_M,
            "model": "both bodies coast with zero wrench from the recorded state; central "
                     "gravity, torque-free tumble",
            "samples": len(samples),
            "passively_safe": sum(row["minimum_clearance_m"] >= CLEARANCE_M for row in samples),
            "by_gap": bins,
            "worst": sorted(samples, key=lambda row: row["minimum_clearance_m"])[:5]}
