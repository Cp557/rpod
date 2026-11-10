"""Closed-loop visual experiments; truth stays in the outer physical environment."""

from dataclasses import asdict
from functools import lru_cache
import copy
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .approach import (
    COMMAND_FIELDS, STATE_FIELDS, Dwell, commanded_wrench, make_approach_scenario,
    make_engine, simulate_approach, state_metrics,
)
from .camera import CameraModel, camera_pose, relative_target_pose
from .frames import port_relative_state, quaternion_matrices, quaternion_product, relative_state
from .observations import ChaserInputs, ImageObservation, NavigationEstimate, PoseEstimate
from .recording import load_events, load_run, software_info, write_json
from .vision_records import (
    json_value, new_output, own_navigation, read_rgb, sha256_file, write_jsonl, write_rgb,
)
from .visual_control import MissionSpec, VisualController

VISUAL_CASES = ("nominal", "disturbed", "hidden", "abort", "short-loss", "long-loss")


@lru_cache(maxsize=1)
def _handoff():
    """Use a recorded M2 terminal handoff, or compute it once without a renderer."""
    source = Path("runs/m2-approach")
    if (source / "manifest.json").is_file():
        manifest, trajectory = load_run(source)
        events = load_events(source, manifest)
    else:
        manifest = make_approach_scenario("approach")
        trajectory, events, _ = simulate_approach(manifest)
        source = None
    transition = next(event for event in events
                      if event["reason"] == "terminal_transfer_started")
    tick = transition["tick"]
    return {
        **{f"initial_{field}": trajectory[field][tick].tolist() for field in STATE_FIELDS},
        "handoff_source": str(source) if source else "headless M2 simulation",
        "handoff_source_tick": tick,
        "handoff_source_time_ns": transition["time_ns"],
    }


def visual_initial_state(case, seed):
    """Environment-only initialization; hidden offsets never enter MissionSpec."""
    if case not in VISUAL_CASES:
        raise ValueError(f"unknown visual case: {case!r}")
    manifest = make_approach_scenario("approach")
    manifest.update(copy.deepcopy(_handoff()))
    positions = np.asarray(manifest["initial_positions_m"])
    velocities = np.asarray(manifest["initial_velocities_mps"])
    quaternions = np.asarray(manifest["initial_quaternions_xyzw"])
    _, _, basis = relative_state(positions, velocities)
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-1, 1, 3) * [0.2, 0.5, 0.2]
    delta_v = rng.uniform(-0.02, 0.02, 3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.deg2rad(2), np.deg2rad(2))
    perturbation = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    positions[0] += basis @ offset
    velocities[0] += basis @ delta_v
    quaternions[0] = quaternion_product(perturbation, quaternions[0])
    if case == "disturbed":
        positions[1] += basis @ [0.5, 0, 0.25]
        velocities[1] += basis @ [0.01, 0.01, -0.01]
        quaternions[1] = quaternion_product(
            quaternions[1], [np.sin(np.deg2rad(3) / 2), 0, 0, np.cos(np.deg2rad(3) / 2)]
        )
    manifest.update({
        "schema_version": 3, "scenario": f"visual-{case}", "visual_case": case,
        "seed": int(seed), "num_steps": 6000, "max_steps": 6000,
        "initial_positions_m": positions.tolist(),
        "initial_velocities_mps": velocities.tolist(),
        "initial_quaternions_xyzw": quaternions.tolist(),
        "hidden_target_offset_lvlh_m": offset.tolist(),
        "hidden_target_delta_v_lvlh_mps": delta_v.tolist(),
        "hidden_target_attitude_perturbation_xyzw": perturbation.tolist(),
        "observation_model": "Blender RGB and image-derived pose; exact own navigation",
        "control_period_ticks": 10, "control_offset_ticks": 2,
        "camera_period_ticks": 10, "delivery_delay_ticks": 2,
        "supervisor_phases": {"0": "hold", "1": "approach", "2": "retreat",
                              "3": "complete", "4": "acquiring", "5": "braking",
                              "6": "nav_lost"},
        "onboard_acceptance": {"position_error_m": 0.1, "speed_mps": 0.01,
                               "alignment_deg": 1.0, "relative_rate_radps": 0.001,
                               "dwell_ns": 10_000_000_000},
        "loss_schedule": {"missing_capture_s": [60] if case == "short-loss" else [],
                          "black_capture_s": list(range(60, 66))
                          if case == "long-loss" else []},
    })
    return manifest


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


def _physical_report(trajectory, initial):
    port_r, port_v, _ = port_relative_state(
        trajectory["positions_m"], trajectory["velocities_mps"],
        trajectory["quaternions_xyzw"], trajectory["angular_velocities_body_radps"],
        initial["docking_ports_body_m"],
    )
    rotations = quaternion_matrices(trajectory["quaternions_xyzw"][-1])
    relative = rotations[0].T @ rotations[1]
    rates = np.einsum("bij,bj->bi", rotations,
                      trajectory["angular_velocities_body_radps"][-1])
    com_range = np.linalg.norm(
        trajectory["positions_m"][:, 1] - trajectory["positions_m"][:, 0], axis=-1
    )
    report = {
        "final_gap_m": -float(port_r[-1, 1]),
        "final_port_speed_mps": float(np.linalg.norm(port_v[-1])),
        "final_alignment_deg": float(np.degrees(np.arccos(np.clip(
            (np.trace(relative) - 1) / 2, -1, 1
        )))),
        "final_relative_rate_radps": float(np.linalg.norm(rates[1] - rates[0])),
        "minimum_com_range_m": float(com_range.min()),
        "minimum_port_gap_m": float((-port_r[:, 1]).min()),
    }
    for quantity, units in (("forces", "n"), ("torques", "nm")):
        norms = np.linalg.norm(trajectory[f"applied_{quantity}_body_{units}"], axis=-1)
        report[f"max_applied_{quantity}_{units}"] = norms.max(axis=0, initial=0).tolist()
    return report


def _qualification(case, summary, trajectory, observations, estimates, events, mission):
    expected = {"nominal": "success", "disturbed": "success", "hidden": "success",
                "abort": "retreated", "short-loss": "success",
                "long-loss": "navigation_lost"}[case]
    delivered = [row for row in observations if row["actual_delivery_time_ns"] is not None]
    acquired = [event for event in events if event["reason"] == "vision_acquired"]
    brakes = [event for event in events if event["reason"] == "visual_brake"]
    resumes = [event for event in events if event["reason"] == "vision_reacquired"]
    completed = summary["outcome"] in ("success", "retreated")
    timings_valid = all(
        row["capture_time_ns"] == row["capture_id"] * 1_000_000_000
        and row["scheduled_delivery_time_ns"] == row["capture_time_ns"] + 200_000_000
        and (row["actual_delivery_time_ns"] == row["scheduled_delivery_time_ns"]
             if row["actual_delivery_time_ns"] is not None
             else row["scheduled_delivery_time_ns"] > summary["elapsed_time_ns"])
        for row in observations
    )
    no_future = all(
        row["target"] is None or (
            row["target"]["time_ns"] == row["time_ns"]
            and row["target"]["capture_time_ns"] <= row["time_ns"]
        ) for row in estimates
    )
    ages_valid = all(
        row["target"] is not None
        and row["time_ns"] - row["target"]["capture_time_ns"] <= mission.maximum_age_ns
        for phase, row in zip(trajectory["phase"], estimates) if phase in (1, 2, 5)
    )
    acquisition_valid = bool(acquired) and acquired[0]["time_ns"] <= mission.acquisition_deadline_ns
    if acquisition_valid:
        acquisition_valid = sum(
            bool(row.get("navigation_accepted"))
            and row["actual_delivery_time_ns"] <= acquired[0]["time_ns"] for row in delivered
        ) >= 3
    baseline = summary["paired_truth_baseline"]
    baseline_valid = (baseline is None and case not in ("nominal", "disturbed", "hidden")) or (
        baseline is not None and baseline["outcome"] == "success"
        and baseline["same_initialization_and_mission"]
        and summary["completion_time_ratio"] <= 1.5
    )
    branch_valid = True
    if case == "abort":
        branch_valid = summary["abort_reason"] == "estimated_gap_abort_request" and any(
            event["reason"] == "estimated_gap_abort_request" and event["phase"] == 2
            for event in events
        )
    elif case == "short-loss":
        branch_valid = bool(brakes) and bool(resumes) and brakes[0]["time_ns"] == 60_200_000_000
        if branch_valid:
            branch_valid = sum(
                bool(row.get("navigation_accepted"))
                and brakes[0]["time_ns"] < row["actual_delivery_time_ns"] <= resumes[0]["time_ns"]
                for row in delivered
            ) >= 2
    elif case == "long-loss":
        final_target = estimates[-1]["target"]
        age = (estimates[-1]["time_ns"] - final_target["capture_time_ns"]
               if final_target is not None else None)
        branch_valid = (
            bool(brakes) and brakes[0]["time_ns"] == 60_200_000_000 and not resumes
            and age is not None and mission.maximum_age_ns < age
            <= mission.maximum_age_ns + mission.dt_ns
        )
    state_valid = all(np.isfinite(array).all() for array in trajectory.values())
    state_valid &= bool(np.allclose(
        np.linalg.norm(trajectory["quaternions_xyzw"], axis=-1), 1, atol=1e-10, rtol=0
    ))
    return expected, {
        "expected_outcome": summary["outcome"] == expected,
        "finite_unit_state": bool(state_valid),
        "physical_separation": (summary["minimum_com_range_m"] > 3
                                and summary["minimum_port_gap_m"] > 2),
        "actuator_bounds": bool(
            np.all(np.asarray(summary["max_applied_forces_n"]) <= 2 + 1e-10)
            and np.all(np.asarray(summary["max_applied_torques_nm"]) <= 0.5 + 1e-10)
        ),
        "capture_and_delivery_timing": timings_valid and no_future,
        "initial_acquisition": acquisition_valid,
        "navigation_age_while_tracking": ages_valid,
        "completion_dwell": (not completed or (
            summary["onboard_achieved_dwell_ns"] >= mission.dwell_ns
            and summary["physical_achieved_dwell_ns"] >= mission.dwell_ns
        )),
        "paired_completion_time": bool(baseline_valid),
        "case_specific_response": bool(branch_valid),
        "no_false_completion": summary["outcome"] != "false_completion",
    }


def _truth_estimate(snapshot, capture_time_ns):
    rotation = quaternion_matrices(snapshot.quaternions_xyzw[0])
    return NavigationEstimate(
        snapshot.time_ns, np.array(snapshot.positions_m[0]),
        np.array(snapshot.velocities_mps[0]), np.array(snapshot.quaternions_xyzw[0]),
        rotation @ np.array(snapshot.angular_velocities_body_radps[0]),
        np.zeros((12, 12)), capture_time_ns, "fresh",
    )


def paired_truth_baseline(initial, mission):
    """Explicit oracle comparison, using the same physical initialization and gains."""
    started = perf_counter()
    engine, controller = make_engine(initial), VisualController(mission)
    target_torque = np.zeros(3)
    physical_dwell = Dwell(mission.dwell_ns)
    last_capture = None
    for tick in range(initial["max_steps"] + 1):
        snapshot = engine.snapshot()
        own = own_navigation(snapshot)
        delivered = tick % 10 == 2
        if delivered:
            last_capture = (tick - 2) * initial["dt_ns"]
        estimate = _truth_estimate(snapshot, last_capture) if last_capture is not None else None
        decision = controller.update(ChaserInputs(
            snapshot.time_ns, own, estimate, True if delivered else None
        ))
        metrics = state_metrics(snapshot, initial)
        physically_valid = physical_dwell.update(
            snapshot.time_ns, _physical_valid(metrics, decision.goal_gap_m)
        )
        boundary = _hard_guard(metrics)
        outcome = "physical_violation" if boundary else decision.outcome
        if outcome or tick == initial["max_steps"]:
            outcome = outcome or "time_limit"
            if outcome in ("success", "retreated") and not physically_valid:
                outcome = "false_completion"
            return {"outcome": outcome, "elapsed_time_ns": snapshot.time_ns,
                    "wall_seconds": perf_counter() - started,
                    "model": "oracle target state, exact own navigation; no rendering",
                    "same_initialization_and_mission": True}
        if tick % 10 == 2:
            target_torque = _target_torque(snapshot, initial)
        engine.step([np.zeros(3), decision.force_body_n],
                    [target_torque, decision.torque_body_nm])


def run_visual(case, seed, output: Path, calibration_path: Path) -> dict:
    """Run only after M3 qualification; save the actual delivered data and decisions."""
    from .navigation import VisualNavigation
    from .perception import estimate_pose
    from .rendering import BlenderRenderer

    autonomy_hashes = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("visual_control.py", "navigation.py", "visual_run.py")
    }
    calibration_path = Path(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("qualification_passed") is not True:
        raise ValueError("M3 qualification must pass before a visual control run")
    initial = visual_initial_state(case, seed)
    mission = MissionSpec(abort_gap_m=20.0 if case == "abort" else None)
    camera_settings = calibration.get("camera", {})
    camera = CameraModel(**{key: camera_settings[key] for key in (
        "width", "height", "hfov_deg", "mount_position_body_m", "rotation_bc", "camera_id"
    ) if key in camera_settings})
    navigator = VisualNavigation(camera=camera, calibration=calibration)
    output = new_output(output)
    (output / "images").mkdir()
    shutil.copyfile(calibration_path, output / "calibration.json")
    initial.update({
        "software": software_info(), "camera": camera.to_dict(),
        "autonomy_implementation_sha256": autonomy_hashes,
        "navigation_configuration": {
            "state": "target inertial r/v/XYZW attitude/spatial rate; left attitude error",
            "error_state_dimension": 12,
            "gravity_mu_m3_s2": 3.986004418e14,
            "translation_integrator": "fixed RK4",
            "maximum_prediction_step_ns": 100_000_000,
            "fixed_lag_ns": 2_000_000_000,
            "translation_acceleration_spectral_density": 2.5e-5,
            "angular_acceleration_spectral_density": 4e-8,
            "acquisition_angular_acceleration_spectral_density": 2.5e-5,
            "high_noise_duration_ns": 10_000_000_000,
            "high_noise_intervals": "first capture and current filter time on reacquisition",
            "measurement_translation_sigma_floor_camera_m": [0.002, 0.002, 0.01],
            "measurement_rotation_sigma_floor_deg": [0.25, 0.25, 0.25],
            "innovation_mahalanobis_squared_limit": 30.0,
            "initial_velocity": "capture own velocity",
            "initial_velocity_sigma_mps": 0.1,
            "initial_angular_velocity_inertial_radps": [0.0, 0.0, 0.0],
            "initial_angular_velocity_sigma_radps": 0.005,
            "fresh_max_capture_age_ns": 1_200_000_000,
            "predicted_max_capture_age_ns": 3_000_000_000,
            "range_bin_selection": "searchsorted upper edges, camera target-origin Z",
            "measurement_covariance": "calibration diagonal in camera, rotated to inertial",
            "covariance_transition": "fourth-order matrix-exponential series",
            "covariance_update": "Joseph plus left-attitude reset",
            "delayed_update": "replay accepted later corrections without regating",
        },
        "mission": json_value(asdict(mission)),
        "calibration_file": "calibration.json",
        "calibration_sha256": sha256_file(output / "calibration.json"),
        "timing": "0.1s physics; 1Hz capture; delivery +0.2s; control at ticks2mod10; "
                  "supervisor and prediction every tick; "
                  "renderer wall time does not advance physics",
    })
    sunlight_n = quaternion_matrices(initial["initial_quaternions_xyzw"][1]) @ [0, 1, 0]
    initial["sunlight_direction_inertial"] = sunlight_n.tolist()
    baseline = (paired_truth_baseline(initial, mission)
                if case in ("nominal", "disturbed", "hidden") else None)
    engine, controller = make_engine(initial), VisualController(mission)
    states = {field: [] for field in STATE_FIELDS}
    states.update({"tick": [], "phase": [], "desired_gap_m": [],
                   "own_position_m": [], "own_velocity_mps": [],
                   "own_quaternion_xyzw": [], "own_angular_velocity_body_radps": []})
    commands = {field: [] for field in COMMAND_FIELDS}
    commands["control_update"] = []
    events, observations, estimates, pending = [], [], [], []
    physical_dwell = Dwell(mission.dwell_ns)
    physical_goal = mission.final_gap_m
    target_torque = np.zeros(3)
    final_outcome, failure_reason, decision = None, None, None
    started = perf_counter()
    renderer = BlenderRenderer(output, camera=camera)
    try:
        with renderer:
            for tick in range(initial["max_steps"] + 1):
                snapshot = engine.snapshot()
                own = own_navigation(snapshot)
                for field in STATE_FIELDS:
                    states[field].append(getattr(snapshot, field))
                states["tick"].append(snapshot.tick)
                states["phase"].append(controller.phase)
                states["desired_gap_m"].append(controller.profile_start)
                states["own_position_m"].append(own.position_m)
                states["own_velocity_mps"].append(own.velocity_mps)
                states["own_quaternion_xyzw"].append(own.quaternion_xyzw)
                states["own_angular_velocity_body_radps"].append(own.angular_velocity_body_radps)
                metrics = state_metrics(snapshot, initial)
                failure_reason = _hard_guard(metrics)
                if failure_reason:
                    events.append({"tick": snapshot.tick, "time_ns": snapshot.time_ns,
                                   "phase": controller.phase, "kind": "physical_boundary",
                                   "reason": failure_reason})
                    final_outcome = "physical_violation"
                    break
                if tick % 10 == 0:
                    capture_id = tick // 10
                    scheduled = snapshot.time_ns + 2 * initial["dt_ns"]
                    missing = capture_id in initial["loss_schedule"]["missing_capture_s"]
                    black = capture_id in initial["loss_schedule"]["black_capture_s"]
                    image_path = output / "images" / f"{capture_id:06d}.png"
                    row = {"capture_id": capture_id, "capture_time_ns": snapshot.time_ns,
                           "scheduled_delivery_time_ns": scheduled,
                           "actual_delivery_time_ns": None,
                           "image_path": None, "image_sha256": None,
                           "fault": ("missing_capture" if missing else
                                     "black_frame" if black else None)}
                    pixels = None
                    if not missing:
                        rotation_ct, translation_ct = relative_target_pose(
                            own, snapshot.positions_m[0], snapshot.quaternions_xyzw[0], camera
                        )
                        light_direction_c = camera_pose(own, camera)[1].T @ sunlight_n
                        row["light_direction_c"] = light_direction_c.tolist()
                        row["render_metadata"] = renderer.render(
                            rotation_ct, translation_ct, image_path,
                            light_direction_c=light_direction_c,
                        )
                        pixels = read_rgb(image_path)
                        if black:
                            pixels = np.zeros_like(pixels)
                            write_rgb(image_path, pixels)
                        row["image_path"] = image_path.relative_to(output).as_posix()
                        row["image_sha256"] = sha256_file(image_path)
                    observations.append(row)
                    pending.append((ImageObservation(capture_id, snapshot.time_ns, scheduled,
                                                     pixels), own, row))
                visual_update = None
                while pending and pending[0][0].delivery_time_ns <= snapshot.time_ns:
                    observation, captured_own, row = pending.pop(0)
                    if observation.capture_time_ns > snapshot.time_ns:
                        raise ValueError("attempted delivery of a future image")
                    pose = (estimate_pose(observation, camera) if observation.pixels is not None
                            else PoseEstimate(observation.capture_time_ns,
                                              observation.delivery_time_ns, False,
                                              reason="missing_capture"))
                    accepted = navigator.observe(pose, captured_own)
                    row.update({"actual_delivery_time_ns": snapshot.time_ns,
                                "pose_estimate": json_value(pose),
                                "navigation_accepted": bool(accepted)})
                    visual_update = bool(accepted)
                estimate = navigator.predict(snapshot.time_ns)
                inputs = ChaserInputs(snapshot.time_ns, own, estimate, visual_update)
                decision = controller.update(inputs)
                if decision.reacquired:
                    navigator.mark_reacquired()
                events.extend(decision.events)
                states["phase"][-1] = decision.phase
                states["desired_gap_m"][-1] = decision.desired_gap_m
                estimates.append({"time_ns": snapshot.time_ns,
                                  "target": json_value(estimate),
                                  "visual_update": visual_update,
                                  "used_for_control": decision.control_update,
                                  "onboard_metrics": json_value(decision.metrics)})
                if physical_goal != decision.goal_gap_m:
                    physical_goal = decision.goal_gap_m
                    physical_dwell = Dwell(mission.dwell_ns)
                physically_valid = physical_dwell.update(
                    snapshot.time_ns, _physical_valid(metrics, physical_goal)
                )
                final_outcome = decision.outcome
                if final_outcome in ("success", "retreated") and not physically_valid:
                    final_outcome, failure_reason = "false_completion", "physical_dwell_failed"
                if final_outcome or tick == initial["max_steps"]:
                    final_outcome = final_outcome or "time_limit"
                    break
                if tick % 10 == 2:
                    target_torque = _target_torque(snapshot, initial)
                forces = np.array([np.zeros(3), decision.force_body_n])
                torques = np.array([target_torque, decision.torque_body_nm])
                try:
                    after = engine.step(forces, torques)
                except ValueError as error:
                    final_outcome, failure_reason = "numerical_failure", str(error)
                    break
                commands["requested_forces_body_n"].append(forces)
                commands["requested_torques_body_nm"].append(torques)
                commands["applied_forces_body_n"].append(after.applied_forces_body_n)
                commands["applied_torques_body_nm"].append(after.applied_torques_body_nm)
                commands["control_update"].append(decision.control_update)
    except (OSError, ValueError, RuntimeError) as error:
        final_outcome, failure_reason = "render_or_pipeline_failure", str(error)
    if not states["tick"]:
        snapshot = engine.snapshot()
        own = own_navigation(snapshot)
        for field in STATE_FIELDS:
            states[field].append(getattr(snapshot, field))
        states.update({"tick": [0], "phase": [3], "desired_gap_m": [30.0],
                       "own_position_m": [own.position_m], "own_velocity_mps": [own.velocity_mps],
                       "own_quaternion_xyzw": [own.quaternion_xyzw],
                       "own_angular_velocity_body_radps": [own.angular_velocity_body_radps]})
    states["phase"][-1] = 6 if final_outcome in (
        "navigation_lost", "acquisition_failed"
    ) else 3
    terminal_ns = int(states["tick"][-1]) * initial["dt_ns"]
    for row in observations:
        if (row["actual_delivery_time_ns"] is None
                and row["scheduled_delivery_time_ns"] <= terminal_ns):
            row.update({
                "actual_delivery_time_ns": terminal_ns, "navigation_accepted": False,
                "delivery_failure": True,
                "pose_estimate": json_value(PoseEstimate(
                    row["capture_time_ns"], row["scheduled_delivery_time_ns"], False,
                    reason="pipeline_failure",
                )),
            })
    while len(estimates) < len(states["tick"]):
        missing_tick = int(states["tick"][len(estimates)])
        missing_ns = missing_tick * initial["dt_ns"]
        deliveries = [row for row in observations if row["actual_delivery_time_ns"] == missing_ns]
        estimates.append({
            "time_ns": missing_ns, "target": None,
            "visual_update": bool(deliveries[-1].get("navigation_accepted")) if deliveries else None,
            "used_for_control": False, "onboard_metrics": None,
        })
    elapsed_ns = int(states["tick"][-1]) * initial["dt_ns"]
    events.append({"tick": int(states["tick"][-1]), "time_ns": elapsed_ns,
                   "phase": states["phase"][-1], "kind": "outcome",
                   "reason": final_outcome})
    trajectory = {key: np.asarray(value, dtype=np.float64) for key, value in states.items()}
    trajectory["tick"] = np.asarray(states["tick"], dtype=np.uint64)
    trajectory["phase"] = np.asarray(states["phase"], dtype=np.uint8)
    for key, value in commands.items():
        trajectory[key] = (np.asarray(value, dtype=np.bool_) if key == "control_update"
                           else np.asarray(value, dtype=np.float64).reshape(-1, 2, 3))
    initial["num_steps"] = int(trajectory["tick"][-1])
    initial["renderer"] = json_value(renderer.provenance)
    ratio = (elapsed_ns / baseline["elapsed_time_ns"] if baseline
             and baseline["elapsed_time_ns"] > 0 else None)
    summary = {
        "outcome": final_outcome, "failure_reason": failure_reason,
        "abort_reason": decision.abort_reason if decision else None,
        "elapsed_time_ns": elapsed_ns, "samples": len(trajectory["tick"]),
        "onboard_achieved_dwell_ns": decision.achieved_dwell_ns if decision else 0,
        "physical_achieved_dwell_ns": (
            elapsed_ns - physical_dwell.started_ns
            if final_outcome in ("success", "retreated")
            and physical_dwell.started_ns is not None else 0
        ),
        "paired_truth_baseline": baseline, "completion_time_ratio": ratio,
        "completion_within_1_5x": (ratio <= 1.5 if ratio is not None else None),
        "wall_seconds": perf_counter() - started,
        "renderer_performance": json_value(renderer.performance),
        "images_delivered": sum(row["actual_delivery_time_ns"] is not None
                                and row["image_path"] is not None for row in observations),
        "accepted_observations": sum(bool(row.get("navigation_accepted"))
                                     for row in observations),
        "numerical_failure": final_outcome == "numerical_failure",
    }
    summary.update(_physical_report(trajectory, initial))
    expected, gates = _qualification(
        case, summary, trajectory, observations, estimates, events, mission
    )
    gates["m3_qualification"] = calibration["qualification_passed"] is True
    summary.update({"expected_outcome": expected, "qualification_gates": gates,
                    "qualification_passed": all(gates.values())})
    write_json(output / "manifest.json", json_value(initial))
    np.savez(output / "trajectory.npz", **trajectory)
    write_json(output / "events.json", json_value(events))
    write_jsonl(output / "observations.jsonl", observations)
    write_jsonl(output / "estimates.jsonl", estimates)
    write_json(output / "summary.json", json_value(summary))
    return summary
