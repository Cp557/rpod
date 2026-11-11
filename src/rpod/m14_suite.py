"""M14 evaluation: docking on thrusters, nominal and with one thruster fault.

Nominal: M13's 28 docking episodes (same conditions, tumbles, seeds and autonomy) flown on the
thruster set. Every episode must dock, the engine must apply exactly the force and torque the
recorded thruster on-times produce, and fault detection must isolate nothing.

Faults: M13 episodes 00-11, each with one thruster failing stuck on or stuck off, from the
first tick whose true port gap is at most 12 m, 5.1 m (the hold) or 1 m (the committed
coast). The failed thruster is the closing (pushes +Y, toward the target) or lateral (pushes
along x or z) thruster that fired longest in the 10 s before. Detection must isolate that
thruster, and only it, within 2 s of the trigger (stuck on) or of its first command after
the trigger (stuck off; a thruster never commanded again needs no detection), and the chaser
must still dock.
"""

from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np

from . import _core
from .docking import DockingController, DockingMechanism, DockingNavigation
from .lidar import LidarModel
from .m11_suite import M11_FILES, M11_SUITE_SEED, m11_specs
from .m13_suite import (LIDAR_CALIBRATION, M13_FILES, MAXIMUM_AGE_S, docking_initial_state,
                        docking_report)
from .m5_suite import FROZEN_FILES, _episode, _stats
from .natural_lidar import load_calibration
from .recording import write_json
from .target_mesh import MESH_PATH, load_target_mesh
from .thrusters import (COUNT, DETECTION_THRESHOLD, FAULT_LOOKBACK_TICKS, MINIMUM_ON_TIME_S,
                        MISALIGNMENT_DEG, SPECIFIC_IMPULSE_S, THRUST_ERROR, THRUST_N, Thrusters,
                        hardware_matrix, thruster_docking_env, wrench_bounds)
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_recording import _load_visual
from .visual_run import camera_from_settings, load_qualified_calibration, run_episode

M14_FILES = ("thrusters.py", "m14_suite.py")
# Candidate thrusters by what they push: +Y (toward the target), or along x or z.
FAULT_GROUPS = {"closing": [2, 6, 10, 14], "lateral": [0, 1, 4, 5, 8, 9, 12, 13]}
FAULT_KINDS = ("stuck_on", "stuck_off")
TRIGGER_GAPS_M = (12.0, 5.1, 1.0)
DETECTION_LIMIT_S = 2.0
# Agreement between recorded applied wrenches and the recorded on-times (float round-off).
APPLIED_TOLERANCE = 1e-9


def m14_specs(seed=M11_SUITE_SEED):
    """The 28 docking episodes nominal, then the 12 faults on episodes 00-11. Seed 11 is the
    suite; any other seed gives development episodes for setting expectations."""
    docking = [spec for spec in m11_specs(seed) if spec["category"] == "operational"]

    def streams(spec):
        return {**spec["streams"], "thruster_errors": [seed, spec["index"], 6],
                "imu": [seed, spec["index"], 7]}

    specs = [{**spec, "id": f"{spec['index']:02d}-nominal", "category": "nominal",
              "kind": "nominal", "expected_outcome": "success", "streams": streams(spec),
              "thruster_fault": None} for spec in docking]
    grid = product(FAULT_KINDS, FAULT_GROUPS.items(), TRIGGER_GAPS_M)
    for spec, (kind, (name, candidates), gap) in zip(docking, grid):
        specs.append({**spec, "id": f"{spec['index']:02d}-{kind}-{name}-at-{gap:g}m",
                      "category": "thruster-fault", "kind": f"{kind}-{name}",
                      "expected_outcome": "success", "streams": streams(spec),
                      "thruster_fault": {"kind": kind, "group": name,
                                         "candidates": candidates, "trigger_gap_m": gap}})
    return specs


def thruster_initial_state(spec):
    """M13's initial state with the chaser's limits raised to the thruster set's bounds."""
    initial = docking_initial_state(spec)
    force, torque = wrench_bounds()
    initial["force_limits_n"] = [initial["force_limits_n"][0], force]
    initial["torque_limits_nm"] = [initial["torque_limits_nm"][0], torque]
    initial["thrusters"] = {
        "count": COUNT, "thrust_n": THRUST_N, "specific_impulse_s": SPECIFIC_IMPULSE_S,
        "minimum_on_time_s": MINIMUM_ON_TIME_S, "thrust_error": THRUST_ERROR,
        "misalignment_deg": MISALIGNMENT_DEG, "detection_threshold": DETECTION_THRESHOLD,
        "allocation": "linear program: largest request scale, then least total on-time; "
                      "request bounded to 2 N and 0.5 N m",
        "limits": "chaser limits are bounds on any thruster combination, so the engine "
                  "never clips"}
    initial["thruster_fault"] = spec["thruster_fault"]
    return initial


def thruster_summary(thrusters):
    """Summary fields from the live thruster set."""
    actual = thrusters.records()["actual_on_s"]
    return {"thrusters": {
        "propellant_kg": thrusters.propellant_kg(),
        "total_impulse_ns": float(np.sum(actual @ np.linalg.norm(thrusters.hardware[:3],
                                                                 axis=0))),
        "fault_tick": thrusters.failed_tick, "failed_thruster": thrusters.failed_thruster,
        "isolations": thrusters.isolations,
        "saturated_ticks": int(np.sum(np.asarray(thrusters.log["scale"]) < 1 - 1e-9)),
        "minimum_scale": float(min(thrusters.log["scale"], default=1.0)),
    }}


def _thruster_check(directory, manifest, trajectory, events, summary, spec):
    """Score one episode's thrusters from its saved records."""
    with np.load(directory / "thrusters.npz", allow_pickle=False) as data:
        records = {key: data[key] for key in data.files}
    dt_s = manifest["dt_ns"] / 1e9
    hardware = hardware_matrix(spec["streams"]["thruster_errors"])
    applied = np.hstack([trajectory["applied_forces_body_n"][:, 1],
                         trajectory["applied_torques_body_nm"][:, 1]])
    produced = records["actual_on_s"] @ hardware.T / dt_s
    applied_error = (float(np.abs(applied - produced).max()) if len(produced) == len(applied)
                     else float("inf"))
    isolations = [event for event in events if event["kind"] == "safety_detection"]
    fault, fault_tick = spec["thruster_fault"], summary["thrusters"]["fault_tick"]
    # Step k runs from tick k; the fault acts from step `fault_tick` on.
    monitored = records["monitoring"] & (np.arange(len(applied)) < (
        fault_tick if fault_tick is not None else np.inf))
    result = {
        "hardware_recorded": bool(np.array_equal(records["hardware_matrix"], hardware)),
        "applied_error": applied_error,
        "isolations": [event["reason"] for event in isolations],
        "maximum_statistic_before_fault": float(records["statistic"][monitored].max(initial=0)),
        "propellant_kg": summary["thrusters"]["propellant_kg"],
        "saturated_ticks": summary["thrusters"]["saturated_ticks"],
        "triggered": fault_tick is not None,
    }
    if fault is None:
        result["correct"] = not isolations
        return result
    thruster = summary["thrusters"]["failed_thruster"]
    if fault_tick is None:
        result.update(failed_thruster=None, exercised=False, detection_latency_s=None,
                      correct=False)
        return result
    candidates = fault["candidates"]
    busiest = records["commanded_on_s"][max(0, fault_tick - FAULT_LOOKBACK_TICKS):fault_tick,
                                        candidates].sum(axis=0)
    if thruster != candidates[int(np.argmax(busiest))]:
        raise ValueError("the failed thruster is not the candidate that fired longest before")
    start = fault_tick
    if fault["kind"] == "stuck_off":
        # Only a command while monitoring runs (before the latch) can reveal it.
        commanded = np.nonzero((records["commanded_on_s"][fault_tick:, thruster] > 0)
                               & records["monitoring"][fault_tick:])[0]
        start = fault_tick + int(commanded[0]) if len(commanded) else None
    latency = ((isolations[0]["tick"] - start) * dt_s if isolations and start is not None
               else None)
    expected = [f"thruster_{thruster:02d}_{fault['kind']}_isolated"]
    result.update(
        failed_thruster=thruster, exercised=start is not None, detection_latency_s=latency,
        correct=(not isolations if start is None else (
            [event["reason"] for event in isolations] == expected
            and isolations[0]["tick"] >= fault_tick and latency <= DETECTION_LIMIT_S)))
    return result


def m14_report(directory):
    """Score the suite from saved records only."""
    directory = Path(directory)
    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    rows = []
    for spec in suite["episodes"]:
        row, _ = _episode(directory, suite, spec)
        if row["record_valid"]:
            path = directory / row["path"]
            manifest, trajectory, events, *_ = _load_visual(path)
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            # The M5 row's 2 m port gap and 2 N bounds do not apply to docking on thrusters.
            row.update({key: summary.get(key) for key in (
                "latched", "contact", "capture_inside_envelope", "latch_time_s",
                "port_estimate_error_at_contact_m", "detumble_s", "minimum_outer_clearance_m",
                "delta_v_mps")}, boundary_violation=row["outcome"] == "physical_violation")
            row.pop("actuator_bounds")
            try:
                row["thrusters"] = _thruster_check(path, manifest, trajectory, events, summary,
                                                   spec)
            except (OSError, KeyError, ValueError) as error:
                row.update(record_valid=False, record_error=f"thrusters: {error}")
        rows.append(row)
    valid = [row for row in rows if row["record_valid"]]
    docked = [row for row in valid if row["outcome"] == "success" and row["latched"]
              and row["capture_inside_envelope"] and row["dwell_satisfied"]]
    faults = [row for row in valid if row["category"] == "thruster-fault"]
    gates = {
        "records_valid_and_uniform": len(valid) == len(rows) and all(
            row["uniform_implementation"] and row["thrusters"]["hardware_recorded"]
            for row in valid),
        "every_episode_docked": len(docked) == len(rows),
        "applied_equals_thruster_output": all(
            row["thrusters"]["applied_error"] <= APPLIED_TOLERANCE for row in valid),
        "no_false_isolation": all(row["thrusters"]["correct"] for row in valid
                                  if row["category"] == "nominal"),
        "every_fault_triggered": all(row["thrusters"]["triggered"] for row in faults),
        "every_fault_isolated_within_2s": all(row["thrusters"]["correct"] for row in faults),
        "no_false_completion_or_violation": not any(
            row["false_completion"] or row["boundary_violation"] for row in valid),
    }
    nominal = [row for row in docked if row["category"] == "nominal"]
    return {
        "artifact_type": "m14_evaluation_report", "passed": all(gates.values()), "gates": gates,
        "scope": ("M13's 28 docking episodes on 16 thrusters with seeded hardware errors, and "
                  "12 single-thruster faults on episodes 00-11; detection by IMU residuals; "
                  "not a population estimate"),
        "seed": suite["seed"], "docked": len(docked),
        "propellant_kg": _stats([row["thrusters"]["propellant_kg"] for row in nominal]),
        "delta_v_mps": _stats([row["delta_v_mps"] for row in nominal]),
        "maximum_statistic_before_fault": max(
            (row["thrusters"]["maximum_statistic_before_fault"] for row in valid), default=None),
        "detection_threshold": DETECTION_THRESHOLD,
        "detection_latency_s": _stats([row["thrusters"].get("detection_latency_s")
                                       for row in faults]),
        # A stuck-off thruster never commanded again before the latch passes vacuously.
        "faults_exercised": sum(bool(row["thrusters"].get("exercised")) for row in faults),
        "contact": {key: _stats([row["contact"][key] for row in docked]) for key in (
            "lateral_m", "alignment_deg", "closing_mps", "lateral_speed_mps",
            "relative_rate_radps")},
        "faults": {row["id"]: {"outcome": row["outcome"], **{
            key: row["thrusters"].get(key) for key in (
                "failed_thruster", "exercised", "detection_latency_s", "isolations",
                "correct")}}
            for row in faults},
        "episodes": rows,
    }


def run_m14_suite(output, camera_calibration_path, seed=M11_SUITE_SEED, ids=None):
    """Run missing M14 episodes (only `ids`, if given), then score the records."""
    output, camera_calibration_path = Path(output), Path(camera_calibration_path)
    calibration = load_qualified_calibration(camera_calibration_path)
    camera = camera_from_settings(calibration["camera"])
    mesh, lidar_calibration = load_target_mesh(MESH_PATH), load_calibration(LIDAR_CALIBRATION)
    source = Path(__file__).parent
    suite = json_value({
        "artifact_type": "m14_evaluation_suite", "milestone": "M14", "seed": seed,
        "calibration_sha256": sha256_file(camera_calibration_path),
        "implementation_sha256": {name: sha256_file(source / name) for name in (
            *FROZEN_FILES, *M11_FILES, *M13_FILES, *M14_FILES)},
        "artifacts_sha256": {**{path.as_posix(): sha256_file(path)
                                for path in (MESH_PATH, LIDAR_CALIBRATION)},
                             "native_core": sha256_file(Path(_core.__file__))},
        "episodes": m14_specs(seed), "maximum_age_s": MAXIMUM_AGE_S,
        "sensors": ["docking lidar", "IMU (thruster fault detection)"],
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
        if final.exists() or (ids is not None and spec["id"] not in ids):
            continue
        partial = episodes / f"{spec['id']}.partial"
        print(f"M14 episode {spec['id']}", flush=True)
        mission = MissionSpec(maximum_age_ns=int(MAXIMUM_AGE_S * 1e9))
        initial, mechanism = thruster_initial_state(spec), DockingMechanism()
        thrusters = Thrusters(spec["streams"]["thruster_errors"], spec["streams"]["imu"],
                              mission.masses_kg[1], mission.inertias_kg_m2[1],
                              initial["dt_ns"] / 1e9, spec["thruster_fault"])

        def qualify(summary, trajectory, observations, estimates, events, initial=initial,
                    mechanism=mechanism, thrusters=thrusters, partial=partial):
            np.savez_compressed(partial / "thrusters.npz", **thrusters.records())
            return {**docking_report(trajectory, initial, mechanism, estimates, events, mesh),
                    **thruster_summary(thrusters)}

        navigator = DockingNavigation(camera, calibration,
                                      LidarModel(**initial["lidar"]["model"]),
                                      mission.inertias_kg_m2[0])
        summary = run_episode(initial, mission, calibration, camera_calibration_path, partial,
                              qualify=qualify, controller=DockingController(mission, mechanism),
                              env_class=thruster_docking_env(mesh, lidar_calibration,
                                                             mechanism, thrusters),
                              navigator=navigator)
        partial.rename(final)
        isolations = summary["thrusters"]["isolations"]
        print(f"  {summary['outcome']} after {summary['elapsed_time_ns'] / 1e9:.1f} s simulated, "
              f"{summary['wall_seconds']:.0f} s wall; isolations "
              f"{[(row['thruster'], row['kind'], row['tick']) for row in isolations]}",
              flush=True)
    report = m14_report(output)
    write_json(output / "report.json", report)
    return report
