"""M13 evaluation: docking with the tumbling target, lidar only.

The 28 M11 operational episodes (same conditions, tumbles and seeds), flown with the docking
lidar through the 5 m hold to contact, capture and detumble, under M12's `robust` settings
(10 s estimate age limit, a tracker that keeps its track through empty scans) and its
collision-avoidance burn. Every episode must dock: contact inside the capture envelope, a
latch, and a stack slowed below the detumble threshold, with no false completion or physical
violation.
"""

import json
from pathlib import Path
import shutil

import numpy as np

from . import _core
from .docking import (CAPTURE, CONTACT_GAP_M, DOCKING_LIDAR, OUTER_HULL_BODY_M,
                      DockingController, DockingMechanism, DockingNavigation,
                      calibrate_docking_lidar,
                      docking_env, docking_lidar, inside_envelope)
from .frames import quaternion_matrices
from .lidar import LidarModel
from .lidar_faults import drift_from
from .m11_suite import M11_FILES, episode_initial_state, m11_specs
from .m5_suite import FROZEN_FILES, _episode, _stats
from .natural_lidar import load_calibration
from .recording import write_json
from .target_mesh import MESH_PATH, load_target_mesh
from .tumble import tumble_metrics
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_run import camera_from_settings, load_qualified_calibration, run_episode

M13_FILES = ("lidar_faults.py", "docking.py", "m13_suite.py")
LIDAR_CALIBRATION = Path("models/m13-lidar-calibration.json")
MAXIMUM_AGE_S = 10.0


def m13_specs():
    """The 28 M11 operational episodes, each flown to capture."""
    return [{**spec, "id": f"{spec['index']:02d}-dock", "category": "docking", "kind": "dock",
             "expected_outcome": "success"}
            for spec in m11_specs() if spec["category"] == "operational"]


def docking_initial_state(spec):
    initial = episode_initial_state(spec)
    initial["lidar"]["model"] = {**initial["lidar"]["model"], **json_value(DOCKING_LIDAR)}
    return initial


def docking_report(trajectory, initial, mechanism, estimates, events, mesh):
    """Summary fields for a docking episode: port-frame extremes before contact, the contact
    the envelope judged, the filter's port error there, the detumble, and delta-v."""
    states = [trajectory[key] for key in ("positions_m", "velocities_mps", "quaternions_xyzw",
                                          "angular_velocities_body_radps")]
    metrics = tumble_metrics(*states, initial["docking_ports_body_m"])
    end = mechanism.latch_tick if mechanism.latched else len(trajectory["tick"])
    rotations = quaternion_matrices(trajectory["quaternions_xyzw"][:end])
    hull = (trajectory["positions_m"][:end, 1, None, :]
            + np.einsum("tij,kj->tki", rotations[:, 1], OUTER_HULL_BODY_M)
            - trajectory["positions_m"][:end, 0, None, :])
    hull = np.einsum("tji,tkj->tki", rotations[:, 0], hull).reshape(-1, 3)
    clearance = mesh.closest_points(np.ascontiguousarray(hull))[2]
    applied = np.linalg.norm(trajectory["applied_forces_body_n"][:, 1], axis=-1)
    transitions = {event["reason"]: event["time_ns"] / 1e9 for event in events
                   if event["kind"] == "supervisor_transition"}
    report = {
        "physical_metrics_frame": "target body (rotating port frame)",
        "minimum_port_gap_m": float(metrics["gap"][:end].min()),
        "minimum_com_range_m": float(np.linalg.norm(metrics["rho"][:end], axis=-1).min()),
        "minimum_outer_clearance_m": float(clearance.min()),
        "latched": mechanism.latched, "contact": mechanism.contact,
        "capture_inside_envelope": (inside_envelope(mechanism.contact)
                                    if mechanism.contact else None),
        "terminal_start_s": transitions.get("terminal_approach"),
        "delta_v_mps": float(applied.sum() / initial["masses_kg"][1] * initial["dt_ns"] / 1e9),
    }
    if mechanism.latched:
        tick = mechanism.latch_tick
        # At the latch tick itself the controller already detumbles; its last estimate is the
        # tick before.
        onboard = estimates[tick - 1]["onboard_metrics"]
        true_port = metrics["port_r"][tick - 1]
        error = np.asarray(onboard["port_r"]) - true_port if onboard else None
        report.update({
            "latch_time_s": tick * initial["dt_ns"] / 1e9,
            "port_estimate_error_at_contact_m": (
                {"lateral": float(np.linalg.norm(error[[0, 2]])), "axial": float(error[1])}
                if error is not None else None),
            "detumble_s": (len(trajectory["tick"]) - 1 - tick) * initial["dt_ns"] / 1e9,
            "final_stack_rate_radps": float(np.linalg.norm(
                trajectory["angular_velocities_body_radps"][-1, 0])),
        })
    else:
        report["coast_after_episode"] = drift_from(initial, trajectory, -1, mesh)
    return report


def m13_report(directory):
    """Score the suite from saved records only."""
    directory = Path(directory)
    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    rows = []
    for spec in suite["episodes"]:
        row, _ = _episode(directory, suite, spec)
        if row["record_valid"]:
            summary = json.loads((directory / row["path"] / "summary.json")
                                 .read_text(encoding="utf-8"))
            # The M5 row flags any port gap under 2 m; docking closes to contact on purpose.
            row.update({key: summary.get(key) for key in (
                "latched", "contact", "capture_inside_envelope", "terminal_start_s",
                "latch_time_s", "port_estimate_error_at_contact_m", "detumble_s",
                "final_stack_rate_radps", "minimum_outer_clearance_m", "delta_v_mps",
                "coast_after_episode")},
                boundary_violation=row["outcome"] == "physical_violation")
        rows.append(row)
    valid = [row for row in rows if row["record_valid"]]
    docked = [row for row in valid if row["outcome"] == "success" and row["latched"]
              and row["capture_inside_envelope"] and row["dwell_satisfied"]
              and row["actuator_bounds"]]
    gates = {
        "records_valid_and_uniform": len(valid) == len(rows) and all(
            row["uniform_implementation"] for row in valid),
        "every_episode_docked": len(docked) == len(rows),
        "no_false_completion_or_violation": not any(
            row["false_completion"] or row["boundary_violation"] for row in valid),
    }
    contacts = [row["contact"] for row in docked]
    errors = [row["port_estimate_error_at_contact_m"] for row in docked
              if row["port_estimate_error_at_contact_m"]]
    return {
        "artifact_type": "m13_evaluation_report", "passed": all(gates.values()), "gates": gates,
        "scope": ("28 M11 operational episodes flown to contact, capture and detumble against "
                  "the unmarked QuikSCAT tumbling at 0.2-1.0 deg/s, docking lidar only; "
                  "capture is a momentum-conserving latch inside a stated envelope, not a "
                  "contact-dynamics model; not a population estimate"),
        "capture_envelope": CAPTURE, "contact_gap_m": CONTACT_GAP_M,
        "docked": len(docked),
        "contact": {key: _stats([contact[key] for contact in contacts])
                    for key in ("lateral_m", "alignment_deg", "closing_mps",
                                "lateral_speed_mps", "relative_rate_radps")},
        "port_estimate_error_at_contact_m": {
            key: _stats([abs(error[key]) for error in errors]) for key in ("lateral", "axial")},
        "latch_time_s": _stats([row["latch_time_s"] for row in docked]),
        "detumble_s": _stats([row["detumble_s"] for row in docked]),
        "delta_v_mps": _stats([row["delta_v_mps"] for row in docked]),
        "minimum_outer_clearance_m": min((row["minimum_outer_clearance_m"] for row in valid),
                                         default=None),
        "episodes": rows,
    }


def run_m13_suite(output, camera_calibration_path):
    """Run missing M13 episodes, then score the records."""
    output, camera_calibration_path = Path(output), Path(camera_calibration_path)
    calibration = load_qualified_calibration(camera_calibration_path)
    camera = camera_from_settings(calibration["camera"])
    mesh, lidar_calibration = load_target_mesh(MESH_PATH), load_calibration(LIDAR_CALIBRATION)
    source = Path(__file__).parent
    suite = json_value({
        "artifact_type": "m13_evaluation_suite", "milestone": "M13",
        "calibration_sha256": sha256_file(camera_calibration_path),
        "implementation_sha256": {name: sha256_file(source / name)
                                  for name in (*FROZEN_FILES, *M11_FILES, *M13_FILES)},
        "artifacts_sha256": {**{path.as_posix(): sha256_file(path)
                                for path in (MESH_PATH, LIDAR_CALIBRATION)},
                             "native_core": sha256_file(Path(_core.__file__))},
        "episodes": m13_specs(), "maximum_age_s": MAXIMUM_AGE_S, "sensors": ["docking lidar"],
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
    episodes = output / "episodes"
    episodes.mkdir(exist_ok=True)
    for partial in episodes.glob("*.partial"):
        shutil.rmtree(partial)
    for spec in suite["episodes"]:
        final = episodes / spec["id"]
        if final.exists():
            continue
        partial = episodes / f"{spec['id']}.partial"
        print(f"M13 episode {spec['id']}", flush=True)
        mission = MissionSpec(maximum_age_ns=int(MAXIMUM_AGE_S * 1e9))
        initial, mechanism = docking_initial_state(spec), DockingMechanism()

        def qualify(summary, trajectory, observations, estimates, events, initial=initial,
                    mechanism=mechanism):
            return docking_report(trajectory, initial, mechanism, estimates, events, mesh)

        navigator = DockingNavigation(camera, calibration,
                                      LidarModel(**initial["lidar"]["model"]),
                                      mission.inertias_kg_m2[0])
        summary = run_episode(initial, mission, calibration, camera_calibration_path, partial,
                              qualify=qualify, controller=DockingController(mission, mechanism),
                              env_class=docking_env(mesh, lidar_calibration, mechanism),
                              navigator=navigator)
        partial.rename(final)
        print(f"  {summary['outcome']} after {summary['elapsed_time_ns'] / 1e9:.1f} s simulated, "
              f"{summary['wall_seconds']:.0f} s wall", flush=True)
    report = m13_report(output)
    write_json(output / "report.json", report)
    return report


def prepare_docking_calibration(samples=1800, seed=13):
    """Calibrate the docking lidar's tracked poses against truth, seeded, and save it."""
    mesh = load_target_mesh(MESH_PATH)
    calibration = calibrate_docking_lidar(docking_lidar(), mesh, samples, seed)
    LIDAR_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    LIDAR_CALIBRATION.write_text(json.dumps({
        **calibration, "mesh": MESH_PATH.as_posix(), "mesh_sha256": sha256_file(MESH_PATH)},
        indent=2) + "\n", encoding="utf-8", newline="\n")
    return calibration
