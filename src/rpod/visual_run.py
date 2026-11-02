"""Closed-loop visual experiments; truth stays in the outer physical environment."""

from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import copy
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .approach import (
    STATE_FIELDS, Dwell, make_approach_scenario, make_engine, simulate_approach, state_metrics,
)
from .camera import CameraModel
from .frames import port_relative_state, quaternion_matrices, quaternion_product, relative_state
from .lidar import LidarModel
from .observations import ChaserInputs, LidarPose, NavigationEstimate, PoseEstimate
from .recording import load_events, load_run, software_info, write_json
from .vision_records import json_value, new_output, own_navigation, sha256_file, write_jsonl
from .visual_agent import VisualAgent
from .visual_control import MissionSpec, VisualController
from .visual_env import VisualEnv, _hard_guard, _physical_valid, _target_torque

VISUAL_CASES = ("nominal", "disturbed", "hidden", "abort", "short-loss", "long-loss")
AUTONOMY_FILES = ("visual_control.py", "navigation.py", "visual_agent.py",
                  "visual_env.py", "visual_run.py", "lidar.py")
# The realistic Sun starts this far above the approach axis; it sets at up to the orbital
# rate (0.065 deg/s), and Earth's limb lies 19.8 deg below level at 400 km.
REALISTIC_SUN_ELEVATION_DEG = 30.0
M4_CONDITIONS = {
    "nominal": {}, "hidden": {}, "abort": {},
    "disturbed": {"chaser_disturbance": {
        "offset_lvlh_m": [0.5, 0, 0.25], "delta_v_lvlh_mps": [0.01, 0.01, -0.01],
        "rotation_axis_body": [1, 0, 0], "rotation_deg": 3,
    }},
    "short-loss": {"loss_schedule": {"missing_capture_s": [60], "black_capture_s": []}},
    "long-loss": {"loss_schedule": {"missing_capture_s": [],
                                    "black_capture_s": list(range(60, 66))}},
}
_INERTIAL_DESCRIPTION = (
    "Earth-centered fixed axes inherited from the M2 encounter; M4 tick zero uses "
    "the recorded handoff states and preserves those inertial axes"
)


def _record_manifest(initial):
    """Resolve descriptive M2 aliases on a copy, after all physical computation."""
    recorded = copy.deepcopy(initial)
    replacements = {
        "goals.retreat_gap_m": recorded["mission"]["retreat_gap_m"],
        "scheduled_abort_gap_m": recorded["mission"]["abort_gap_m"],
        "frames.inertial": _INERTIAL_DESCRIPTION,
    }
    changes = []
    for field, value in replacements.items():
        parts = field.split(".")
        parent = recorded
        for part in parts[:-1]:
            parent = parent[part]
        previous = parent[parts[-1]]
        if previous != value:
            changes.append({"field": field, "before": previous, "after": value})
        parent[parts[-1]] = value
    recorded["record_writer_implementation_sha256"] = sha256_file(Path(__file__))
    note = {
        "version": 1, "operation": "write_time_resolution",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive manifest fields only; execution-time autonomy hashes, "
                 "physical initialization, observations, commands, and qualification are preserved",
        "changes": changes,
        "abort_threshold_basis": "onboard estimated axial port gap",
    }
    archive = Path(__file__).resolve().parents[2] / "benchmarks/m4-qualified-source"
    index = archive / "sha256.json"
    if index.is_file():
        archived = json.loads(index.read_text(encoding="utf-8"))
        if archived["files_sha256"] == recorded["autonomy_implementation_sha256"]:
            note["qualified_source_archive"] = {
                "workspace_relative_path": "benchmarks/m4-qualified-source",
                "index_sha256": sha256_file(index),
            }
    recorded["record_metadata_correction"] = note
    return recorded


def correct_visual_manifest(directory: Path) -> dict:
    """Correct a completed record's three descriptive aliases without rerunning it."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if not (directory / "summary.json").is_file():
        raise ValueError("metadata correction requires a completed run with summary.json")
    original = json.loads(path.read_text(encoding="utf-8"))
    if original.get("schema_version") != 3 or original.get("visual_case") not in VISUAL_CASES:
        raise ValueError("metadata correction requires a schema-3 visual run")
    before_hash = sha256_file(path)
    existing = original.get("record_metadata_correction")
    if existing is not None:
        if (existing.get("version") != 1
                or original["goals"]["retreat_gap_m"] != original["mission"]["retreat_gap_m"]
                or original["scheduled_abort_gap_m"] != original["mission"]["abort_gap_m"]
                or original["frames"]["inertial"] != _INERTIAL_DESCRIPTION):
            raise ValueError("existing metadata correction is inconsistent")
        return {"changed": False, "manifest_sha256": before_hash}
    backup = directory / "manifest.before-metadata-correction.json"
    original_bytes = path.read_bytes()
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            raise ValueError("existing original-manifest backup differs from this record")
    else:
        with backup.open("xb") as destination:
            destination.write(original_bytes)
    recorded = _record_manifest(original)
    recorded["record_metadata_correction"].update({
        "operation": "post_run_metadata_correction",
        "original_manifest_file": backup.name,
        "original_manifest_sha256": before_hash,
    })
    write_json(path, recorded)
    return {
        "changed": True, "original_manifest_sha256": before_hash,
        "manifest_sha256": sha256_file(path),
        "qualified_runner_sha256": original["autonomy_implementation_sha256"]["visual_run.py"],
        "record_writer_implementation_sha256": recorded["record_writer_implementation_sha256"],
    }


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
    """One of the six fixed M4 qualification cases."""
    if case not in VISUAL_CASES:
        raise ValueError(f"unknown visual case: {case!r}")
    manifest = initial_state(f"visual-{case}", seed, **M4_CONDITIONS[case])
    manifest["visual_case"] = case
    return manifest


def initial_state(scenario, seed, *, error_scale=1.0, chaser_disturbance=None,
                  sun_angle_deg=0.0, image_degradation=None, loss_schedule=None,
                  lidar_noise_seed=None, scene="simple"):
    """Environment-only initialization; hidden offsets never enter MissionSpec.

    The seeded target errors use the M4 hidden-initialization distribution, scaled by
    `error_scale`. `sun_angle_deg` turns the Sun from the M4 lit face (`sunlight_direction`).
    A lidar noise seed adds the default lidar to the episode. `scene="realistic"` renders
    the detailed satellite and Earth instead of the simple marked cube.
    """
    if scene not in ("simple", "realistic"):
        raise ValueError(f"unknown render scene: {scene!r}")
    manifest = make_approach_scenario("approach")
    manifest.update(copy.deepcopy(_handoff()))
    positions = np.asarray(manifest["initial_positions_m"])
    velocities = np.asarray(manifest["initial_velocities_mps"])
    quaternions = np.asarray(manifest["initial_quaternions_xyzw"])
    _, _, basis = relative_state(positions, velocities)
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-1, 1, 3) * [0.2, 0.5, 0.2] * error_scale
    delta_v = rng.uniform(-0.02, 0.02, 3) * error_scale
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.deg2rad(2), np.deg2rad(2)) * error_scale
    perturbation = np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)]
    positions[0] += basis @ offset
    velocities[0] += basis @ delta_v
    quaternions[0] = quaternion_product(perturbation, quaternions[0])
    if chaser_disturbance is not None:
        positions[1] += basis @ chaser_disturbance["offset_lvlh_m"]
        velocities[1] += basis @ chaser_disturbance["delta_v_lvlh_mps"]
        half = np.deg2rad(chaser_disturbance["rotation_deg"]) / 2
        quaternions[1] = quaternion_product(quaternions[1], np.r_[
            np.asarray(chaser_disturbance["rotation_axis_body"]) * np.sin(half), np.cos(half)
        ])
    manifest.update({
        "schema_version": 3, "scenario": scenario,
        "seed": int(seed), "num_steps": 6000, "max_steps": 6000,
        "chaser_disturbance": chaser_disturbance, "sun_angle_deg": sun_angle_deg,
        "image_degradation": image_degradation,
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
        "loss_schedule": loss_schedule or {"missing_capture_s": [], "black_capture_s": []},
    })
    if lidar_noise_seed is not None:
        manifest["lidar"] = {"model": LidarModel().to_dict(), "noise_seed": list(lidar_noise_seed)}
        manifest["observation_model"] = ("Blender RGB and image-derived pose; simulated flash "
                                         "lidar and reflector-derived pose; exact own navigation")
    if scene != "simple":
        manifest["scene"] = scene
    return manifest


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


def camera_from_settings(settings):
    """Rebuild the camera from calibration or manifest settings, ignoring derived fields."""
    return CameraModel(**{key: settings[key] for key in (
        "width", "height", "hfov_deg", "mount_position_body_m", "rotation_bc", "camera_id"
    ) if key in settings})


def load_qualified_calibration(path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if calibration.get("qualification_passed") is not True:
        raise ValueError("M3 qualification must pass before a visual control run")
    return calibration


def run_visual(case, seed, output: Path, calibration_path: Path) -> dict:
    """Run one fixed M4 case only after M3 qualification, and apply its gates."""
    calibration = load_qualified_calibration(calibration_path)
    initial = visual_initial_state(case, seed)
    mission = MissionSpec(abort_gap_m=20.0 if case == "abort" else None)

    def qualify(summary, trajectory, observations, estimates, events):
        expected, gates = _qualification(
            case, summary, trajectory, observations, estimates, events, mission
        )
        gates["m3_qualification"] = calibration["qualification_passed"] is True
        return {"expected_outcome": expected, "qualification_gates": gates,
                "qualification_passed": all(gates.values())}

    return run_episode(initial, mission, calibration, calibration_path, output,
                       paired_baseline=case in ("nominal", "disturbed", "hidden"),
                       qualify=qualify)


def sunlight_direction(initial):
    """Inertial Sun incident-ray direction: the M4 lit face turned by `sun_angle_deg`.

    The turn is about chaser body Z, within the orbit plane. The realistic scene's Earth
    can eclipse the Sun, so there the Sun is first raised REALISTIC_SUN_ELEVATION_DEG above
    the approach axis toward body X, the handoff's local vertical, then turned about body X.
    It then stays clear of Earth's limb for the longest episode.
    """
    light_body = np.array([0.0, 1.0, 0.0])
    if initial.get("scene") == "realistic":
        elevation = np.deg2rad(REALISTIC_SUN_ELEVATION_DEG)
        light_body = np.array([-np.sin(elevation), np.cos(elevation), 0.0])
    half = np.deg2rad(initial["sun_angle_deg"]) / 2
    axis = [1, 0, 0] if initial.get("scene") == "realistic" else [0, 0, 1]
    light_body = quaternion_matrices([*np.multiply(axis, np.sin(half)), np.cos(half)]) @ light_body
    return quaternion_matrices(initial["initial_quaternions_xyzw"][1]) @ light_body


def run_episode(initial, mission, calibration, calibration_path, output,
                paired_baseline=False, qualify=None) -> dict:
    """Drive the reference client through the environment; save the actual delivered data.

    `qualify(summary, trajectory, observations, estimates, events)` may add summary fields.
    """
    camera = camera_from_settings(calibration.get("camera", {}))
    lidar = LidarModel(**initial["lidar"]["model"]) if initial.get("lidar") else None
    agent = VisualAgent(mission, camera, calibration, lidar)
    output = new_output(output)
    shutil.copyfile(calibration_path, output / "calibration.json")
    initial.update({
        "software": software_info(), "camera": camera.to_dict(),
        "autonomy_implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name)) for name in AUTONOMY_FILES
        },
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
    if lidar:
        initial["navigation_configuration"].update({
            "lidar_measurement": "weighted Kabsch reflector pose; covariance is the inverse "
                                 "linearized fit information in lidar axes, rotated to inertial",
            "failed_delivery_policy": "brake unless another capture accepted within 1 s",
        })
    initial["sunlight_direction_inertial"] = sunlight_direction(initial).tolist()
    baseline = paired_truth_baseline(initial, mission) if paired_baseline else None
    env = VisualEnv(camera)
    autonomy_seconds = []
    started = perf_counter()
    try:
        with env:
            observation = env.reset(initial, output)
            while observation is not None:
                tick_started = perf_counter()
                command = agent.act(observation)
                autonomy_seconds.append(perf_counter() - tick_started)
                observation = env.step(command)
    except (OSError, ValueError, RuntimeError) as error:
        if env.engine is None:
            raise
        env.fail(str(error))

    states = {key: list(value) for key, value in env.states.items()}
    ticks = len(states["tick"])
    unanswered = ticks - len(agent.phase)
    phase = agent.phase + [agent.controller.phase] * unanswered
    desired_gap = agent.desired_gap_m + [agent.controller.profile_start] * unanswered
    events = sorted(
        [{"tick": event["tick"], "time_ns": event["time_ns"], "phase": phase[event["tick"]],
          "kind": event["kind"], "reason": event["reason"]} for event in env.events]
        + agent.events,
        key=lambda event: event["tick"],
    )
    final_outcome = env.outcome
    phase[-1] = 6 if final_outcome in ("navigation_lost", "acquisition_failed") else 3
    terminal_ns = int(states["tick"][-1]) * initial["dt_ns"]
    observations = env.observations
    for row in observations:
        if row["actual_delivery_time_ns"] is not None:
            row["navigation_accepted"] = agent.accepted.get(row["capture_id"], False)
        elif row["scheduled_delivery_time_ns"] <= terminal_ns:
            row.update({
                "actual_delivery_time_ns": terminal_ns, "navigation_accepted": False,
                "delivery_failure": True,
                "pose_estimate": json_value(PoseEstimate(
                    row["capture_time_ns"], row["scheduled_delivery_time_ns"], False,
                    reason="pipeline_failure",
                )),
            })
    lidar_rows = env.lidar_rows
    for row in lidar_rows:
        if row["actual_delivery_time_ns"] is not None:
            row["navigation_accepted"] = agent.lidar_accepted.get(row["capture_id"], False)
        elif row["scheduled_delivery_time_ns"] <= terminal_ns:
            row.update({
                "actual_delivery_time_ns": terminal_ns, "navigation_accepted": False,
                "delivery_failure": True,
                "pose_estimate": json_value(LidarPose(
                    row["capture_time_ns"], row["scheduled_delivery_time_ns"], False,
                    reason="pipeline_failure",
                )),
            })
    estimates = list(agent.estimates)
    while len(estimates) < ticks:
        missing_tick = int(states["tick"][len(estimates)])
        missing_ns = missing_tick * initial["dt_ns"]
        deliveries = [row for row in observations + lidar_rows
                      if row["actual_delivery_time_ns"] == missing_ns]
        estimates.append({
            "time_ns": missing_ns, "target": None,
            "visual_update": bool(deliveries[-1].get("navigation_accepted")) if deliveries else None,
            "used_for_control": False, "onboard_metrics": None,
        })
    elapsed_ns = int(states["tick"][-1]) * initial["dt_ns"]
    events.append({"tick": int(states["tick"][-1]), "time_ns": elapsed_ns,
                   "phase": phase[-1], "kind": "outcome", "reason": final_outcome})
    states.update(phase=phase, desired_gap_m=desired_gap)
    trajectory = {key: np.asarray(value, dtype=np.float64) for key, value in states.items()}
    trajectory["tick"] = np.asarray(states["tick"], dtype=np.uint64)
    trajectory["phase"] = np.asarray(phase, dtype=np.uint8)
    steps = len(env.commands["requested_forces_body_n"])
    for key, value in {**env.commands, "control_update": agent.control_update[:steps]}.items():
        trajectory[key] = (np.asarray(value, dtype=np.bool_) if key == "control_update"
                           else np.asarray(value, dtype=np.float64).reshape(-1, 2, 3))
    initial["num_steps"] = int(trajectory["tick"][-1])
    initial["renderer"] = json_value(env.renderer.provenance)
    ratio = (elapsed_ns / baseline["elapsed_time_ns"] if baseline
             and baseline["elapsed_time_ns"] > 0 else None)
    decision = agent.decision
    summary = {
        "outcome": final_outcome, "failure_reason": env.failure_reason,
        "abort_reason": decision.abort_reason if decision else None,
        "elapsed_time_ns": elapsed_ns, "samples": len(trajectory["tick"]),
        "onboard_achieved_dwell_ns": decision.achieved_dwell_ns if decision else 0,
        "physical_achieved_dwell_ns": env.physical_dwell_ns,
        "paired_truth_baseline": baseline, "completion_time_ratio": ratio,
        "completion_within_1_5x": (ratio <= 1.5 if ratio is not None else None),
        "wall_seconds": perf_counter() - started,
        "renderer_performance": json_value(env.renderer.performance),
        "performance": {"perception_seconds": env.perception_seconds,
                        "autonomy_seconds": autonomy_seconds},
        "images_delivered": sum(row["actual_delivery_time_ns"] is not None
                                and row["image_path"] is not None for row in observations),
        "accepted_observations": sum(bool(row.get("navigation_accepted"))
                                     for row in observations),
        "numerical_failure": final_outcome == "numerical_failure",
    }
    if lidar:
        summary["performance"]["lidar_perception_seconds"] = env.lidar_seconds
        summary.update({
            "lidar_scans_delivered": sum(row["actual_delivery_time_ns"] is not None
                                         for row in lidar_rows),
            "lidar_accepted_observations": sum(bool(row.get("navigation_accepted"))
                                               for row in lidar_rows),
        })
    summary.update(_physical_report(trajectory, initial))
    if qualify is not None:
        summary.update(qualify(summary, trajectory, observations, estimates, events))
    write_json(output / "manifest.json", json_value(_record_manifest(initial)))
    np.savez(output / "trajectory.npz", **trajectory)
    write_json(output / "events.json", json_value(events))
    write_jsonl(output / "observations.jsonl", observations)
    if lidar:
        write_jsonl(output / "lidar.jsonl", lidar_rows)
    write_jsonl(output / "estimates.jsonl", estimates)
    write_json(output / "summary.json", json_value(summary))
    return summary
