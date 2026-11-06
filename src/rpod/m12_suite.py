"""M12 evaluation: lidar faults on the tumbling target, and safety after a navigation loss.

Each episode flies one of M11's operational episodes (same conditions, tumble and seeds), with
one lidar fault placed by the true port gap. The chaser answers a navigation loss with a
collision-avoidance burn (`CamTumbleController`). The suite gates safety, not recovery: every
episode must end in success, a retreat or a declared navigation loss, with no false
completion or boundary violation, and after every loss the burn and the coast that follows
must keep the hull clear of the target mesh. Recovery is reported per fault and trigger gap,
and so is the coast from the moment of the loss without the burn.
"""

from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np

from . import _core
from .keypoints import KeypointEstimator, load_keypoints
from .lidar import LidarModel
from .lidar_faults import CamTumbleController, drift_from, lidar_fault_env
from .m11_suite import (KEYPOINTS, LIDAR_CALIBRATION, M11_FILES, episode_initial_state,
                        m11_specs, physical_report)
from .m5_suite import FROZEN_FILES, _episode, _safe
from .natural_lidar import load_calibration
from .recording import write_json
from .target_mesh import MESH_PATH, load_target_mesh
from .tumble import CLEARANCE_M, TumbleNavigation
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_recording import _load_visual
from .visual_run import camera_from_settings, load_qualified_calibration, run_episode

M12_FILES = ("lidar_faults.py", "m12_suite.py")
# (kind, scans): one lost scan, one empty scan, three scans 0.25 m long, a 4 s outage.
FAULTS = (("dropped", 1), ("blank", 1), ("biased", 3), ("dropped", 4))
TRIGGER_GAPS_M = (25.0, 12.0, 5.1)
BIAS_M = 0.25
SAFE_OUTCOMES = ("success", "retreated", "navigation_lost")
# `m11` is M11's autonomy; `robust` keeps the ICP track through empty scans and declares a
# navigation loss after 10 s without an accepted measurement instead of 3 s. Both answer a
# loss with the collision-avoidance burn.
CONFIGURATIONS = {"m11": {"maximum_age_s": 3.0, "coast_empty_scans": False},
                  "robust": {"maximum_age_s": 10.0, "coast_empty_scans": True}}


def m12_specs():
    """Every fault at every trigger gap, each on its own M11 operational episode."""
    operational = [spec for spec in m11_specs() if spec["category"] == "operational"]
    specs = []
    for base, ((kind, scans), gap) in zip(operational, product(FAULTS, TRIGGER_GAPS_M)):
        name = f"{kind}-{scans}" if kind != "dropped" or scans == 1 else f"outage-{scans}"
        specs.append({**base, "id": f"{base['index']:02d}-{name}-at-{gap:g}m",
                      "category": "lidar-fault", "kind": name, "expected_outcome": None,
                      "lidar_fault": {"kind": kind, "scans": scans, "trigger_gap_m": gap,
                                      "bias_m": BIAS_M if kind == "biased" else 0.0}})
    return specs


def _fault_response(lidar, events, estimates, outcome):
    """When the fault hit, and how the supervisor answered, from the saved records."""
    faulted = [row for row in lidar if row.get("fault")]
    if not faulted:
        return {"triggered": False}
    delivered = faulted[0]["actual_delivery_time_ns"]
    brakes = [event["time_ns"] for event in events
              if event.get("reason") == "visual_brake" and event["time_ns"] >= delivered]
    resumes = [event["time_ns"] for event in events
               if event.get("reason") == "vision_reacquired" and brakes
               and event["time_ns"] > brakes[0]]
    lost = [event for event in events if event.get("kind") == "supervisor_transition"
            and event["reason"] == "navigation_lost"]
    at_loss = estimates[lost[0]["tick"]] if lost else None
    return {
        "triggered": True, "captures": [row["capture_id"] for row in faulted],
        "trigger_time_s": faulted[0]["capture_time_ns"] / 1e9,
        "faulted_scans_accepted": sum(bool(row.get("navigation_accepted")) for row in faulted),
        "brake_latency_s": (brakes[0] - delivered) / 1e9 if brakes else None,
        "reacquisition_s": (resumes[0] - brakes[0]) / 1e9 if resumes else None,
        "loss_time_s": lost[0]["time_ns"] / 1e9 if lost else None,
        "estimate_age_at_loss_s": ((at_loss["time_ns"] - at_loss["target"]["capture_time_ns"])
                                   / 1e9 if at_loss else None),
        "avoidance_burn_s": ((events[-1]["time_ns"] - lost[0]["time_ns"]) / 1e9
                             if lost and outcome == "navigation_lost" else None),
    }


def m12_report(directory):
    """Score the suite from saved records only."""
    directory = Path(directory)
    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    rows = []
    for spec in suite["episodes"]:
        row, _ = _episode(directory, suite, spec)
        if row["record_valid"]:
            path = directory / row["path"]
            _, _, events, _, estimates, lidar = _load_visual(path)
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            row.update(fault=spec["lidar_fault"],
                       fault_response=_fault_response(lidar, events, estimates, row["outcome"]),
                       minimum_mesh_clearance_m=summary["minimum_mesh_clearance_m"],
                       coast_after_episode=summary["coast_after_episode"],
                       coast_without_burn=summary["coast_without_burn"])
        rows.append(row)
    valid = [row for row in rows if row["record_valid"]]
    safe = [row for row in valid if _safe(row) and row["outcome"] in SAFE_OUTCOMES
            and row["minimum_mesh_clearance_m"] >= CLEARANCE_M]
    lost = [row for row in valid if row["outcome"] == "navigation_lost"]
    gates = {
        "records_valid_and_uniform": len(valid) == len(rows) and all(
            row["uniform_implementation"] for row in valid),
        "every_fault_triggered": all(row["fault_response"]["triggered"] for row in valid),
        "safe_under_every_fault": len(safe) == len(rows),
        "clear_after_every_loss": all(row["coast_after_episode"]["passively_safe"]
                                      for row in lost),
    }
    recovery = {}
    for row in valid:
        cell = recovery.setdefault(row["kind"], {})
        cell[f"{row['fault']['trigger_gap_m']:g} m"] = row["outcome"]
    return {
        "artifact_type": "m12_evaluation_report", "passed": all(gates.values()), "gates": gates,
        "scope": ("12 lidar faults, one per episode, on M11 operational episodes 00-11 against "
                  "the unmarked QuikSCAT tumbling at 0.2-1.0 deg/s; gates safety and safety "
                  "after a navigation loss, reports recovery; not a population estimate or "
                  "contact validation"),
        "sensors": suite["sensors"], "configuration": suite["configuration"],
        "recovery": recovery,
        "recovered": sum(row["outcome"] == "success" for row in valid),
        "navigation_lost": len(lost),
        "minimum_mesh_clearance_m": min((row["minimum_mesh_clearance_m"] for row in valid),
                                        default=None),
        "minimum_coast_clearance_m": min((row["coast_after_episode"]["minimum_clearance_m"]
                                          for row in valid), default=None),
        "losses_unsafe_without_burn": sum(not row["coast_without_burn"]["passively_safe"]
                                          for row in lost),
        "episodes": rows,
    }


def run_m12_suite(output, camera_calibration_path, keypoint_model=None, configuration="m11"):
    """Run missing M12 episodes, lidar-only or fused with keypoints, then score the records."""
    output, camera_calibration_path = Path(output), Path(camera_calibration_path)
    settings = CONFIGURATIONS[configuration]
    calibration = load_qualified_calibration(camera_calibration_path)
    camera = camera_from_settings(calibration["camera"])
    mesh, lidar_calibration = load_target_mesh(MESH_PATH), load_calibration(LIDAR_CALIBRATION)
    source = Path(__file__).parent
    artifacts = {path.as_posix(): sha256_file(path)
                 for path in (MESH_PATH, LIDAR_CALIBRATION, KEYPOINTS)}
    if keypoint_model is not None:
        artifacts[Path(keypoint_model).as_posix()] = sha256_file(keypoint_model)
    # The Python hashes do not cover the native core, whose behavior can change on its own.
    artifacts["native_core"] = sha256_file(Path(_core.__file__))
    suite = json_value({
        "artifact_type": "m12_evaluation_suite", "milestone": "M12",
        "calibration_sha256": sha256_file(camera_calibration_path),
        "implementation_sha256": {name: sha256_file(source / name)
                                  for name in (*FROZEN_FILES, *M11_FILES, *M12_FILES)},
        "artifacts_sha256": artifacts, "episodes": m12_specs(),
        "configuration": {"name": configuration, **settings},
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
        print(f"M12 episode {spec['id']}", flush=True)
        mission = MissionSpec(maximum_age_ns=int(settings["maximum_age_s"] * 1e9))
        initial = episode_initial_state(spec)
        initial["lidar_fault"] = spec["lidar_fault"]

        def qualify(summary, trajectory, *records, initial=initial):
            lost = np.flatnonzero(np.asarray(trajectory["phase"]) == 6)
            return {**physical_report(trajectory, initial, mesh, mission),
                    "coast_after_episode": drift_from(initial, trajectory, -1, mesh),
                    "coast_without_burn": (drift_from(initial, trajectory, lost[0], mesh)
                                          if len(lost) else None)}

        navigator = TumbleNavigation(camera, calibration,
                                     LidarModel(**initial["lidar"]["model"]),
                                     mission.inertias_kg_m2[0])
        summary = run_episode(initial, mission, calibration, camera_calibration_path, partial,
                              qualify=qualify, controller=CamTumbleController(mission),
                              env_class=lidar_fault_env(mesh, lidar_calibration, keypoints,
                                                        settings["coast_empty_scans"]),
                              navigator=navigator)
        partial.rename(final)
        print(f"  {summary['outcome']} after {summary['elapsed_time_ns'] / 1e9:.1f} s simulated, "
              f"{summary['wall_seconds']:.0f} s wall", flush=True)
    report = m12_report(output)
    write_json(output / "report.json", report)
    return report
