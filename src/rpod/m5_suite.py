"""M5 frozen evaluation: seeded visual-environment episodes and a records-only report."""

from functools import partial
import json
from pathlib import Path
import shutil

import numpy as np

from .camera import relative_target_pose
from .lidar import LidarModel
from .observations import OwnNavigation
from .recording import write_json
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_recording import _load_visual
from .visual_run import (
    AUTONOMY_FILES, camera_from_settings, initial_state, load_qualified_calibration, run_episode,
)

M5_SUITE_SEED = 5
FROZEN_FILES = (*AUTONOMY_FILES, "approach.py", "blender_worker.py", "camera.py", "frames.py",
                "observations.py", "perception.py", "rendering.py", "vision_records.py")
QUALIFIED_DEGRADATION = (2.0, 0.5)
STRESS_KINDS = ("init-errors-2x", "heavy-degradation", "oblique-sun", "chaser-disturbance-2x",
                "degraded-dropped-oblique", "combined-moderate")


def _degradation(seed, index, noise_dn, blur_px):
    return {"noise_dn": noise_dn, "blur_px": blur_px, "noise_seed": [seed, index, 3]}


def _chaser_disturbance(rng, scale=1.0):
    """Within the M4 'disturbed' magnitudes, in a random direction, times `scale`."""
    axis = rng.normal(size=3)
    return {
        "offset_lvlh_m": (rng.uniform(-1, 1, 3) * [0.5, 0.5, 0.25] * scale).tolist(),
        "delta_v_lvlh_mps": (rng.uniform(-0.01, 0.01, 3) * scale).tolist(),
        "rotation_axis_body": (axis / np.linalg.norm(axis)).tolist(),
        "rotation_deg": float(rng.uniform(0, 3) * scale),
    }


def _signed(rng, low, high):
    return float(rng.choice([-1, 1]) * rng.uniform(low, high))


def operational_conditions(seed, index):
    """The operational draw from stream [seed, index, 1]: Sun angle, a chaser disturbance,
    and qualified image degradation on even indices."""
    rng = np.random.default_rng([seed, index, 1])
    return {
        "sun_angle_deg": float(rng.uniform(-30, 30)),
        "image_degradation": (_degradation(seed, index, *QUALIFIED_DEGRADATION)
                              if index % 2 == 0 else None),
        "chaser_disturbance": _chaser_disturbance(rng),
    }


def suite_specs(seed=M5_SUITE_SEED, lidar=False, scene="simple"):
    """Resolve all 40 episodes before any rendering: 28 operational, 6 fault, 6 stress.

    With lidar, every episode adds a lidar noise stream, and extended camera loss is
    expected to succeed because lidar captures keep navigation current. A realistic
    scene renders the same episodes with the detailed satellite and Earth.
    """
    plan = ([("operational", "approach")] * 28
            + [("fault", kind) for kind in ("abort", "dropped-frame", "extended-loss") for _ in (0, 1)]
            + [("stress", kind) for kind in STRESS_KINDS])
    specs = []
    for index, (category, kind) in enumerate(plan):
        conditions_rng = np.random.default_rng([seed, index, 1])
        fault_rng = np.random.default_rng([seed, index, 2])
        conditions, abort_gap, expected = {}, None, None
        if category in ("operational", "fault"):
            conditions, expected = operational_conditions(seed, index), "success"
        if kind == "abort":
            abort_gap, expected = float(fault_rng.uniform(10, 25)), "retreated"
        elif kind == "dropped-frame":
            conditions["loss_schedule"] = {
                "missing_capture_s": [int(fault_rng.integers(20, 151))], "black_capture_s": [],
            }
        elif kind == "extended-loss":
            start, length = int(fault_rng.integers(20, 151)), int(fault_rng.integers(2, 7))
            lost = list(range(start, start + length))
            missing = index % 2 == 0
            conditions["loss_schedule"] = {"missing_capture_s": lost if missing else [],
                                           "black_capture_s": [] if missing else lost}
            expected = "success" if lidar else "navigation_lost"
        elif kind == "init-errors-2x":
            conditions = {"error_scale": 2.0}
        elif kind == "heavy-degradation":
            conditions = {"image_degradation": _degradation(seed, index, 4.0, 1.0)}
        elif kind == "oblique-sun":
            conditions = {"sun_angle_deg": _signed(conditions_rng, 45, 60)}
        elif kind == "chaser-disturbance-2x":
            conditions = {"chaser_disturbance": _chaser_disturbance(conditions_rng, 2.0)}
        elif kind == "degraded-dropped-oblique":
            conditions = {
                "sun_angle_deg": _signed(conditions_rng, 30, 45),
                "image_degradation": _degradation(seed, index, *QUALIFIED_DEGRADATION),
                "loss_schedule": {"missing_capture_s": [int(fault_rng.integers(20, 151))],
                                  "black_capture_s": []},
            }
        elif kind == "combined-moderate":
            conditions = {"error_scale": 1.5, "sun_angle_deg": _signed(conditions_rng, 30, 45),
                          "image_degradation": _degradation(seed, index, 3.0, 0.75)}
        streams = {"conditions": [seed, index, 1], "faults": [seed, index, 2],
                   "image_noise": [seed, index, 3]}
        if lidar:
            streams["lidar_noise"] = [seed, index, 4]
        specs.append({
            "id": f"{index:02d}-{category}-{kind}", "index": index, "category": category,
            "kind": kind, "init_seed": 1000 + index, "abort_gap_m": abort_gap,
            "expected_outcome": expected, "conditions": conditions, "streams": streams,
            **({"scene": scene} if scene != "simple" else {}),
        })
    return specs


def implementation_hashes():
    source = Path(__file__).parent
    return {name: sha256_file(source / name) for name in FROZEN_FILES}


def episode_initial_state(spec):
    initial = initial_state(f"visual-m5-{spec['category']}", spec["init_seed"],
                            lidar_noise_seed=spec["streams"].get("lidar_noise"),
                            scene=spec.get("scene", "simple"), **spec["conditions"])
    initial["m5_episode"] = spec
    return initial


def learned_controller_spec(policy_path):
    """Provenance of the learned approach law: policy weights and implementation hashes."""
    from .learned_control import IMPLEMENTATION_FILES

    source = Path(__file__).parent
    return {"name": "learned", "policy": Path(policy_path).as_posix(),
            "policy_sha256": sha256_file(policy_path),
            "implementation_sha256": {name: sha256_file(source / name)
                                      for name in IMPLEMENTATION_FILES}}


def run_suite(output, calibration_path, lidar=False, scene="simple", controller="reference",
              policy_path=None) -> dict:
    """Run missing episodes of the frozen suite, then recompute the report from records.

    `controller="learned"` drives approach with the policy at `policy_path` behind the safety
    filter (`learned_control.LearnedController`); the suite then records its provenance.
    """
    output, calibration_path = Path(output), Path(calibration_path)
    calibration = load_qualified_calibration(calibration_path)
    if calibration.get("scene", "simple") != scene:
        raise ValueError(f"the calibration was qualified on the {calibration.get('scene', 'simple')} "
                         f"scene, not the {scene} scene")
    learned = learned_controller_spec(policy_path) if controller == "learned" else None
    suite = json_value({
        "artifact_type": "m5_evaluation_suite", "suite_seed": M5_SUITE_SEED,
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_range_bins_m": calibration["range_bins"],
        "implementation_sha256": implementation_hashes(),
        "episodes": suite_specs(lidar=lidar, scene=scene),
        **({"sensors": ["camera", "lidar"]} if lidar else {}),
        **({"scene": scene} if scene != "simple" else {}),
        **({"controller": learned} if learned else {}),
    })
    path = output / "suite.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != suite:
            raise ValueError("this suite was frozen with different episodes, calibration, or "
                             "implementation; use a new output directory")
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
        print(f"M5 episode {spec['id']}", flush=True)
        mission, initial = MissionSpec(abort_gap_m=spec["abort_gap_m"]), episode_initial_state(spec)
        agent_controller, qualify = None, None
        if learned:
            from .learned_control import LearnedController
            from .policy import load_policy

            initial["learned_controller"] = learned
            agent_controller = LearnedController(mission, load_policy(policy_path))

            def qualify(summary, *records, controller=agent_controller):
                return {"safety_filter_interventions": controller.interventions}
        summary = run_episode(
            initial, mission, calibration, calibration_path, partial,
            paired_baseline=spec["category"] == "operational", qualify=qualify,
            controller=agent_controller,
        )
        partial.rename(final)
        print(f"  {summary['outcome']} after {summary['elapsed_time_ns'] / 1e9:.1f} s simulated, "
              f"{summary['wall_seconds']:.0f} s wall", flush=True)
    report = suite_report(output)
    write_json(output / "report.json", report)
    return report


def _stats(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return None
    return {"count": int(len(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _angle_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def _estimation_errors(trajectory, estimates):
    """Filtered target error against truth at every tracking tick (approach/retreat/brake)."""
    position, attitude = [], []
    for tick, row in enumerate(estimates):
        if trajectory["phase"][tick] not in (1, 2, 5) or row["target"] is None:
            continue
        target = row["target"]
        position.append(float(np.linalg.norm(
            np.asarray(target["position_m"]) - trajectory["positions_m"][tick, 0]
        )))
        dot = abs(float(np.dot(target["quaternion_xyzw"], trajectory["quaternions_xyzw"][tick, 0])))
        attitude.append(float(np.degrees(2 * np.arccos(min(dot, 1.0)))))
    return position, attitude


def _pose_errors(manifest, trajectory, observations):
    """Accepted image-derived pose error against the truth pose at capture."""
    camera = camera_from_settings(manifest["camera"])
    errors = []
    for row in observations:
        if not row.get("navigation_accepted"):
            continue
        tick = row["capture_time_ns"] // manifest["dt_ns"]
        own = OwnNavigation(row["capture_time_ns"], trajectory["own_position_m"][tick],
                            trajectory["own_velocity_mps"][tick],
                            trajectory["own_quaternion_xyzw"][tick],
                            trajectory["own_angular_velocity_body_radps"][tick])
        rotation_ct, translation_ct = relative_target_pose(
            own, trajectory["positions_m"][tick, 0], trajectory["quaternions_xyzw"][tick, 0], camera
        )
        pose = row["pose_estimate"]
        errors.append({
            "range_m": float(translation_ct[2]),
            "translation_m": float(np.linalg.norm(np.asarray(pose["translation_ct"])
                                                  - translation_ct)),
            "rotation_deg": _angle_deg(np.asarray(pose["rotation_ct"]) @ rotation_ct.T),
        })
    return errors


def _lidar_pose_errors(manifest, trajectory, rows):
    """Accepted reflector-derived pose error against the truth pose at capture."""
    lidar = LidarModel(**manifest["lidar"]["model"])
    errors = []
    for row in rows:
        if not row.get("navigation_accepted"):
            continue
        tick = row["capture_time_ns"] // manifest["dt_ns"]
        own = OwnNavigation(row["capture_time_ns"], trajectory["own_position_m"][tick],
                            trajectory["own_velocity_mps"][tick],
                            trajectory["own_quaternion_xyzw"][tick],
                            trajectory["own_angular_velocity_body_radps"][tick])
        rotation_lt, translation_lt = relative_target_pose(
            own, trajectory["positions_m"][tick, 0], trajectory["quaternions_xyzw"][tick, 0], lidar
        )
        pose = row["pose_estimate"]
        errors.append({
            "range_m": float(translation_lt[2]),
            "translation_m": float(np.linalg.norm(np.asarray(pose["translation_lt"])
                                                  - translation_lt)),
            "rotation_deg": _angle_deg(np.asarray(pose["rotation_lt"]) @ rotation_lt.T),
        })
    return errors


def _fault_response(manifest, events, estimates, outcome):
    lost = sorted(manifest["loss_schedule"]["missing_capture_s"]
                  + manifest["loss_schedule"]["black_capture_s"])
    if not lost:
        return None
    delivered_ns = lost[0] * 1_000_000_000 + manifest["delivery_delay_ticks"] * manifest["dt_ns"]
    brakes = [event["time_ns"] for event in events
              if event["reason"] == "visual_brake" and event["time_ns"] >= delivered_ns]
    resumes = [event["time_ns"] for event in events
               if event["reason"] == "vision_reacquired" and brakes and event["time_ns"] > brakes[0]]
    final = estimates[-1]["target"]
    return {
        "lost_captures": lost,
        "brake_latency_s": (brakes[0] - delivered_ns) / 1e9 if brakes else None,
        "reacquisition_s": (resumes[0] - brakes[0]) / 1e9 if resumes else None,
        "final_estimate_age_s": ((estimates[-1]["time_ns"] - final["capture_time_ns"]) / 1e9
                                 if outcome == "navigation_lost" and final else None),
    }


def _episode(directory, suite, spec):
    path = directory / "episodes" / spec["id"]
    result = {"id": spec["id"], "category": spec["category"], "kind": spec["kind"],
              "expected_outcome": spec["expected_outcome"],
              "path": path.relative_to(directory).as_posix()}
    try:
        manifest, trajectory, events, observations, estimates, lidar = _load_visual(path)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {**result, "record_valid": False, "record_error": str(error)}, None
    outcome, dwell_ns = summary["outcome"], manifest["mission"]["dwell_ns"]
    completed = outcome in ("success", "retreated")
    position_error, attitude_error = _estimation_errors(trajectory, estimates)
    pose_errors = _pose_errors(manifest, trajectory, observations)
    renderer, timing = summary["renderer_performance"], summary["performance"]
    hashes, frozen = manifest["autonomy_implementation_sha256"], suite["implementation_sha256"]
    result.update({
        "record_valid": True,
        "uniform_implementation": (
            manifest["calibration_sha256"] == suite["calibration_sha256"]
            and set(hashes) == set(AUTONOMY_FILES) & set(frozen)
            and hashes == {name: frozen[name] for name in hashes}
            and manifest["m5_episode"] == spec
            and manifest.get("learned_controller") == suite.get("controller")
        ),
        "outcome": outcome, "failure_reason": summary["failure_reason"],
        "abort_reason": summary["abort_reason"],
        "simulated_seconds": summary["elapsed_time_ns"] / 1e9,
        "completion_time_ratio": summary["completion_time_ratio"],
        "dwell_satisfied": not completed or (
            summary["onboard_achieved_dwell_ns"] >= dwell_ns
            and summary["physical_achieved_dwell_ns"] >= dwell_ns
        ),
        "false_completion": outcome == "false_completion",
        "boundary_violation": (outcome == "physical_violation"
                               or summary["minimum_com_range_m"] <= 3
                               or summary["minimum_port_gap_m"] <= 2),
        "actuator_bounds": bool(
            np.all(np.asarray(summary["max_applied_forces_n"]) <= 2 + 1e-10)
            and np.all(np.asarray(summary["max_applied_torques_nm"]) <= 0.5 + 1e-10)
        ),
        **{key: summary[key] for key in (
            "final_gap_m", "final_port_speed_mps", "final_alignment_deg",
            "final_relative_rate_radps", "minimum_com_range_m", "minimum_port_gap_m",
            "max_applied_forces_n", "max_applied_torques_nm", "accepted_observations",
            "images_delivered",
        )},
        "estimated_target_position_error_m": _stats(position_error),
        "estimated_target_attitude_error_deg": _stats(attitude_error),
        "image_pose_translation_error_m": _stats([row["translation_m"] for row in pose_errors]),
        "image_pose_rotation_error_deg": _stats([row["rotation_deg"] for row in pose_errors]),
        "fault_response": _fault_response(manifest, events, estimates, outcome),
        "safety_filter_interventions": summary.get("safety_filter_interventions"),
        "wall_seconds": summary["wall_seconds"],
        "simulated_to_wall_ratio": summary["elapsed_time_ns"] / 1e9 / summary["wall_seconds"],
        "renderer_startup_seconds": renderer["startup_seconds"],
        "sampled_peak_blender_rss_mib": (renderer["peak_rss_bytes"] / 2**20
                                         if renderer["peak_rss_bytes"] else None),
    })
    samples = {"position_error": position_error, "attitude_error": attitude_error,
               "pose_errors": pose_errors, "render": renderer["render_seconds"],
               "perception": timing["perception_seconds"], "autonomy": timing["autonomy_seconds"]}
    if "lidar" in manifest:
        lidar_errors = _lidar_pose_errors(manifest, trajectory, lidar)
        result.update({
            "lidar_scans_delivered": summary["lidar_scans_delivered"],
            "lidar_accepted_observations": summary["lidar_accepted_observations"],
            "lidar_pose_translation_error_m": _stats([row["translation_m"] for row in lidar_errors]),
            "lidar_pose_rotation_error_deg": _stats([row["rotation_deg"] for row in lidar_errors]),
        })
        samples.update(lidar_errors=lidar_errors,
                       lidar_perception=timing["lidar_perception_seconds"])
    return result, samples


def _safe(row):
    return (row["record_valid"] and not row["false_completion"]
            and not row["boundary_violation"] and row["actuator_bounds"])


def _operational_passed(row):
    return (_safe(row) and row["outcome"] == "success" and row["dwell_satisfied"]
            and row["completion_time_ratio"] is not None and row["completion_time_ratio"] <= 1.5)


def _fault_passed(row, lidar=False):
    if not (_safe(row) and row["outcome"] == row["expected_outcome"] and row["dwell_satisfied"]):
        return False
    response = row["fault_response"]
    if row["kind"] == "abort":
        return row["abort_reason"] == "estimated_gap_abort_request"
    if lidar:
        # Lidar captures cover the lost camera deliveries, so the approach never brakes.
        return response["brake_latency_s"] is None
    if row["kind"] == "dropped-frame":
        return response["brake_latency_s"] == 0 and response["reacquisition_s"] is not None
    return (response["brake_latency_s"] == 0 and response["reacquisition_s"] is None
            and 3.0 < response["final_estimate_age_s"] <= 3.1 + 1e-9)


def _range_bins(pose_errors, edges):
    bins, lower = [], 0.0
    for upper in edges:
        rows = [row for row in pose_errors if lower < row["range_m"] <= upper]
        bins.append({"camera_range_m": [lower, upper],
                     "translation_m": _stats([row["translation_m"] for row in rows]),
                     "rotation_deg": _stats([row["rotation_deg"] for row in rows])})
        lower = upper
    return bins


def suite_report(directory) -> dict:
    """Score the suite from saved records only; no physics, rendering, or estimation runs."""
    directory = Path(directory)
    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    lidar = "lidar" in suite.get("sensors", ())
    realistic = suite.get("scene") == "realistic"
    rows, samples = [], {}
    for spec in suite["episodes"]:
        row, sample = _episode(directory, suite, spec)
        rows.append(row)
        samples[spec["id"]] = sample
    categories = {}
    for category in ("operational", "fault", "stress"):
        members = [row for row in rows if row["category"] == category]
        valid = [row for row in members if row["record_valid"]]
        pooled = [samples[row["id"]] for row in valid]
        pose_errors = [error for sample in pooled for error in sample["pose_errors"]]
        outcomes = {}
        for row in valid:
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        categories[category] = {
            "episodes": len(members), "outcomes": outcomes,
            "completion_time_ratio": _stats([row["completion_time_ratio"] for row in valid]),
            "final_gap_m": _stats([row["final_gap_m"] for row in valid
                                   if row["outcome"] == "success"]),
            "final_port_speed_mps": _stats([row["final_port_speed_mps"] for row in valid
                                            if row["outcome"] == "success"]),
            "minimum_com_range_m": min((row["minimum_com_range_m"] for row in valid), default=None),
            "minimum_port_gap_m": min((row["minimum_port_gap_m"] for row in valid), default=None),
            "estimated_target_position_error_m": _stats(
                [value for sample in pooled for value in sample["position_error"]]),
            "estimated_target_attitude_error_deg": _stats(
                [value for sample in pooled for value in sample["attitude_error"]]),
            "image_pose_by_range": _range_bins(pose_errors, suite["calibration_range_bins_m"]),
        }
        if lidar:
            categories[category]["lidar_pose_by_range"] = _range_bins(
                [error for sample in pooled for error in sample["lidar_errors"]],
                suite["calibration_range_bins_m"],
            )
    operational = [row for row in rows if row["category"] == "operational"]
    faults = [row for row in rows if row["category"] == "fault"]
    fault_passed = partial(_fault_passed, lidar=lidar)
    gates = {
        "operational_all_succeed": len(operational) == 28 and all(map(_operational_passed, operational)),
        "fault_expected_responses": len(faults) == 6 and all(map(fault_passed, faults)),
        "records_valid_and_uniform": all(row["record_valid"] and row["uniform_implementation"]
                                         for row in rows),
    }
    valid = [samples[row["id"]] for row in rows if row["record_valid"]]
    report = {
        "artifact_type": "m5_evaluation_report", "suite_seed": suite["suite_seed"],
        "calibration_sha256": suite["calibration_sha256"],
        "implementation_sha256": suite["implementation_sha256"],
        "report_implementation_sha256": sha256_file(Path(__file__)),
        "scope": ("Fixed 40-episode suite with cooperative markers, "
                  + ("realistic Blender imagery (detailed satellite, space lighting, NASA-imaged "
                     "Earth and atmosphere)" if realistic else "simple Blender imagery")
                  + ", and a 30 m M2 handoff; not a population reliability estimate, "
                  "natural-feature perception, or contact validation"),
        "passed": all(gates.values()), "gates": gates,
        "failed_episodes": [
            {"id": row["id"], "category": row["category"], "outcome": row.get("outcome"),
             "expected_outcome": row["expected_outcome"], "path": row["path"]}
            for row in rows if (
                (row["category"] == "operational" and not _operational_passed(row))
                or (row["category"] == "fault" and not fault_passed(row))
                or (row["category"] == "stress" and not (
                    row["record_valid"] and row["outcome"] in ("success", "retreated")))
            )
        ],
        **({"safety_filter_interventions": sum(
            row.get("safety_filter_interventions") or 0 for row in rows)}
           if "controller" in suite else {}),
        "stress_findings": [row["id"] for row in rows
                            if row["category"] == "stress" and row["record_valid"] and not _safe(row)],
        "categories": categories,
        "performance": {
            "total_wall_seconds": sum(row["wall_seconds"] for row in rows if row["record_valid"]),
            "episode_wall_seconds": _stats([row["wall_seconds"] for row in rows
                                            if row["record_valid"]]),
            "simulated_to_wall_ratio": _stats([row["simulated_to_wall_ratio"] for row in rows
                                               if row["record_valid"]]),
            "render_seconds": _stats([value for sample in valid for value in sample["render"]]),
            "perception_seconds": _stats([value for sample in valid
                                          for value in sample["perception"]]),
            "autonomy_step_seconds": _stats([value for sample in valid
                                             for value in sample["autonomy"]]),
            "renderer_startup_seconds": _stats([row["renderer_startup_seconds"] for row in rows
                                                if row["record_valid"]]),
            "sampled_peak_blender_rss_mib": _stats([row["sampled_peak_blender_rss_mib"]
                                                    for row in rows if row["record_valid"]]),
            "scope": ("Wall times on an unisolated desktop; renderer timing includes PNG writes; "
                      "RSS sampled every 0.1 s, not OS high-water memory"),
        },
        "episodes": rows,
    }
    if realistic:
        report["scene"] = suite["scene"]
    if lidar:
        report["sensors"] = suite["sensors"]
        report["scope"] += ("; lidar is a ray-cast flash-lidar model with lidar-only "
                            "retroreflectors, not a measured sensor")
        report["performance"]["lidar_perception_seconds"] = _stats(
            [value for sample in valid for value in sample["lidar_perception"]])
    return report
