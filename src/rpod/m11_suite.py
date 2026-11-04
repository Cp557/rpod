"""M11 evaluation: the 40 M5 episodes against an unmarked, tumbling QuikSCAT.

Each episode keeps its M5 conditions and faults, adds a tumble from stream [11, index, 5]
(U(0.2, 1.0) deg/s about a uniform random axis), and starts synchronized 30 m out on the
rotating docking axis. Navigation is natural-feature lidar ICP, alone or fused with camera
keypoints. The records and report reuse the M5 machinery; physical metrics are recomputed in
the target's port frame, completion is timed against the tumbling truth oracle, and the
chaser hull's clearance from the target mesh is checked.
"""

import json
from pathlib import Path
import shutil

import numpy as np

from .keypoints import KeypointEstimator, load_keypoints
from .lidar import LidarModel
from .m5_suite import FROZEN_FILES, suite_report, suite_specs
from .natural_lidar import load_calibration
from .recording import write_json
from .target_mesh import MESH_PATH, load_target_mesh
from .tumble import (CHASER_SURFACE_BODY_M, TumbleController, TumbleNavigation, tumble_env,
                     tumble_initial_state, tumble_metrics, tumble_truth_baseline)
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_run import camera_from_settings, load_qualified_calibration, run_episode
from .frames import quaternion_matrices

M11_SUITE_SEED = 11
M11_FILES = ("tumble.py", "natural_lidar.py", "target_mesh.py", "m11_worker.py", "keypoints.py",
             "m11_suite.py")
LIDAR_CALIBRATION = Path("models/m11-lidar-calibration.json")
KEYPOINTS = Path("models/m11-keypoints.json")


def m11_specs(seed=M11_SUITE_SEED):
    """The M5 lidar-suite episodes with a seeded tumble each."""
    specs = []
    for spec in suite_specs(seed, lidar=True, scene="realistic"):
        rng = np.random.default_rng([seed, spec["index"], 5])
        axis = rng.normal(size=3)
        specs.append({**spec, "streams": {**spec["streams"], "tumble": [seed, spec["index"], 5]},
                      "tumble": {"rate_deg_s": float(rng.uniform(0.2, 1.0)),
                                 "axis_body": (axis / np.linalg.norm(axis)).tolist()}})
    return specs


def episode_initial_state(spec):
    initial = tumble_initial_state(f"visual-m11-{spec['category']}", spec["init_seed"],
                                   rate_deg_s=spec["tumble"]["rate_deg_s"],
                                   rate_axis=spec["tumble"]["axis_body"],
                                   lidar_noise_seed=spec["streams"]["lidar_noise"],
                                   **spec["conditions"])
    initial["m5_episode"] = spec
    return initial


def physical_report(trajectory, initial, mesh, mission):
    """Port-frame replacements for the runner's LVLH summary fields, and mesh clearance."""
    metrics = tumble_metrics(trajectory["positions_m"], trajectory["velocities_mps"],
                             trajectory["quaternions_xyzw"],
                             trajectory["angular_velocities_body_radps"],
                             initial["docking_ports_body_m"])
    rotations = quaternion_matrices(trajectory["quaternions_xyzw"])
    every = slice(None, None, 10)
    hull = (trajectory["positions_m"][every, 1, None, :]
            + np.einsum("tij,kj->tki", rotations[every, 1], CHASER_SURFACE_BODY_M)
            - trajectory["positions_m"][every, 0, None, :])
    hull = np.einsum("tji,tkj->tki", rotations[every, 0], hull).reshape(-1, 3)
    clearance = mesh.closest_points(np.ascontiguousarray(hull))[2]
    return {
        "final_gap_m": float(metrics["gap"][-1]),
        "final_port_speed_mps": float(np.linalg.norm(metrics["port_v"][-1])),
        "final_alignment_deg": float(metrics["alignment_deg"][-1]),
        "final_relative_rate_radps": float(metrics["relative_rate"][-1]),
        "minimum_port_gap_m": float(metrics["gap"].min()),
        "minimum_com_range_m": float(np.linalg.norm(metrics["rho"], axis=-1).min()),
        "minimum_mesh_clearance_m": float(clearance.min()),
        "physical_metrics_frame": "target body (rotating port frame)",
    }


def run_m11_suite(output, camera_calibration_path, keypoint_model=None):
    """Run missing M11 episodes, lidar-only or fused with keypoints, then score the records."""
    output, camera_calibration_path = Path(output), Path(camera_calibration_path)
    calibration = load_qualified_calibration(camera_calibration_path)
    camera = camera_from_settings(calibration["camera"])
    mesh, lidar_calibration = load_target_mesh(MESH_PATH), load_calibration(LIDAR_CALIBRATION)
    source = Path(__file__).parent
    artifacts = {path.as_posix(): sha256_file(path)
                 for path in (MESH_PATH, LIDAR_CALIBRATION, KEYPOINTS)}
    if keypoint_model is not None:
        artifacts[Path(keypoint_model).as_posix()] = sha256_file(keypoint_model)
    suite = json_value({
        "artifact_type": "m5_evaluation_suite", "milestone": "M11", "suite_seed": M11_SUITE_SEED,
        "calibration_sha256": sha256_file(camera_calibration_path),
        "calibration_range_bins_m": calibration["range_bins"],
        "implementation_sha256": {name: sha256_file(source / name)
                                  for name in (*FROZEN_FILES, *M11_FILES)},
        "artifacts_sha256": artifacts,
        "episodes": m11_specs(),
        "sensors": ["camera", "lidar"] if keypoint_model else ["lidar"],
        "scene": "realistic", "target": "unmarked, tumbling QuikSCAT",
    })
    path = output / "suite.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != suite:
            raise ValueError("this suite was frozen with different episodes, calibration, "
                             "artifacts or implementation; use a new output directory")
    else:
        new_output(output)
        write_json(path, suite)
    keypoints = (KeypointEstimator(keypoint_model, load_keypoints(KEYPOINTS), camera)
                 if keypoint_model else None)
    episodes = output / "episodes"
    episodes.mkdir(exist_ok=True)
    for partial in episodes.glob("*.partial"):
        shutil.rmtree(partial)
    for spec in suite["episodes"]:
        final = episodes / spec["id"]
        if final.exists():
            continue
        partial = episodes / f"{spec['id']}.partial"
        print(f"M11 episode {spec['id']}", flush=True)
        mission, initial = MissionSpec(abort_gap_m=spec["abort_gap_m"]), episode_initial_state(spec)
        oracle = (tumble_truth_baseline(initial, mission)
                  if spec["category"] == "operational" else None)

        def qualify(summary, trajectory, *records, initial=initial, oracle=oracle):
            ratio = (summary["elapsed_time_ns"] / oracle["elapsed_time_ns"]
                     if oracle and oracle["elapsed_time_ns"] > 0 else None)
            return {**physical_report(trajectory, initial, mesh, mission),
                    "paired_truth_baseline": oracle, "completion_time_ratio": ratio,
                    "completion_within_1_5x": ratio <= 1.5 if ratio is not None else None}

        navigator = TumbleNavigation(camera, calibration,
                                     LidarModel(**initial["lidar"]["model"]),
                                     mission.inertias_kg_m2[0])
        summary = run_episode(initial, mission, calibration, camera_calibration_path, partial,
                              qualify=qualify, controller=TumbleController(mission),
                              env_class=tumble_env(mesh, lidar_calibration, keypoints),
                              navigator=navigator)
        partial.rename(final)
        print(f"  {summary['outcome']} after {summary['elapsed_time_ns'] / 1e9:.1f} s simulated, "
              f"{summary['wall_seconds']:.0f} s wall", flush=True)
    report = suite_report(output)
    report.update(artifact_type="m11_evaluation_report", scope=(
        "Fixed 40-episode suite against the unmarked QuikSCAT tumbling at 0.2-1.0 deg/s, "
        "chaser starting synchronized 30 m out on the rotating docking axis; natural-feature "
        "lidar ICP" + (" fused with camera keypoints" if keypoint_model else "")
        + "; port-frame metrics; not a population reliability estimate or contact "
        "validation"))
    clearances = []
    for spec in suite["episodes"]:
        summary_path = episodes / spec["id"] / "summary.json"
        if summary_path.is_file():
            clearances.append(json.loads(summary_path.read_text(encoding="utf-8"))
                              .get("minimum_mesh_clearance_m"))
    report["minimum_mesh_clearance_m"] = min((value for value in clearances if value is not None),
                                             default=None)
    write_json(output / "report.json", report)
    return report


def prepare_assets():
    """Export the unmarked mesh, choose keypoints, and calibrate lidar ICP; all seeded."""
    from .keypoints import select_keypoints
    from .natural_lidar import calibrate
    from .target_mesh import export_target_mesh

    removed = export_target_mesh(MESH_PATH)
    with np.load(MESH_PATH) as data:
        points = select_keypoints(data["vertices_body_m"])
    KEYPOINTS.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"mesh": MESH_PATH.as_posix(), "mesh_sha256": sha256_file(MESH_PATH),
                  "cooperative_aids_removed": removed}
    KEYPOINTS.write_text(json.dumps({
        "artifact_type": "m11_keypoints", **provenance,
        "selection": "12 farthest-point mesh vertices, then 8 on the docking face within "
                     "1.2 m of the axis", "points_body_m": points.tolist()}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    lidar = calibrate(LidarModel(), load_target_mesh(MESH_PATH), (5.5, 35.0), 900, M11_SUITE_SEED,
                      [5.7, 8.45, 15.95, 25.95, 31.95, 1e9])
    LIDAR_CALIBRATION.write_text(json.dumps({**lidar, **provenance}, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
    return removed
