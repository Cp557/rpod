"""Validate and replay saved visual runs without executing their runtime pipeline."""

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .vision_records import read_rgb, sha256_file


_STATES = (
    "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps",
)
_COMMANDS = (
    "requested_forces_body_n", "requested_torques_body_nm",
    "applied_forces_body_n", "applied_torques_body_nm",
)
_OWN = (
    "own_position_m", "own_velocity_mps", "own_quaternion_xyzw",
    "own_angular_velocity_body_radps",
)


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line, parse_constant=_reject_constant)
                for line in source if line.strip()]


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**64 - 1


def _array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid visual record {label}")
    return array


def _unit_quaternions(value, shape, label):
    array = _array(value, shape, label)
    if not np.allclose(np.linalg.norm(array, axis=-1), 1.0, atol=1e-10, rtol=0):
        raise ValueError(f"visual record {label} must contain unit quaternions")
    return array


def _rotation(value, label):
    rotation = _array(value, (3, 3), label)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError(f"visual record {label} must be a proper rotation")
    return rotation


def _checked_file(directory, relative, checksum):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("visual record paths must be relative")
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("visual record path escapes the run directory")
    if not isinstance(checksum, str) or sha256_file(path) != checksum:
        raise ValueError(f"visual record checksum mismatch: {relative}")
    return path


def _validate_manifest(directory, manifest):
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("unsupported visual record schema")
    if manifest["schema_version"] != 3:
        raise ValueError("unsupported visual record schema; expected version 3")
    if manifest["body_order"] != ["target", "chaser"] or not isinstance(
        manifest["scenario"], str
    ) or not manifest["scenario"].startswith("visual-"):
        raise ValueError("invalid visual record body order or scenario")
    steps, dt_ns = manifest["num_steps"], manifest["dt_ns"]
    if (
        not _integer(steps) or not _integer(dt_ns, 1)
        or steps * dt_ns > 2**64 - 1
        or not _integer(manifest["max_steps"])
        or steps > manifest["max_steps"]
        or not _integer(manifest["camera_period_ticks"], 1)
        or not _integer(manifest["delivery_delay_ticks"])
    ):
        raise ValueError("invalid visual record clock or sample count")
    for key in ("mu_m3_s2", "earth_radius_m"):
        if not np.isfinite(manifest[key]) or manifest[key] <= 0:
            raise ValueError(f"invalid visual record {key}")
    for key, shape, positive in (
        ("masses_kg", (2,), True),
        ("inertias_kg_m2", (2, 3), True),
        ("force_limits_n", (2,), False),
        ("torque_limits_nm", (2,), False),
        ("docking_ports_body_m", (2, 3), None),
    ):
        values = _array(manifest[key], shape, key)
        if positive is not None and np.any(values <= 0 if positive else values < 0):
            raise ValueError(f"invalid visual record {key}")
    camera = manifest["camera"]
    if not _integer(camera["width"], 1) or not _integer(camera["height"], 1):
        raise ValueError("invalid visual record camera dimensions")
    _array(camera["K"], (3, 3), "camera intrinsics")
    _array(camera["mount_position_body_m"], (3,), "camera mount")
    _rotation(camera["rotation_bc"], "camera mount rotation")
    calibration = _checked_file(
        directory, manifest["calibration_file"], manifest["calibration_sha256"]
    )
    qualification = _read_json(calibration)
    if not isinstance(qualification, dict) or qualification.get("qualification_passed") is not True:
        raise ValueError("visual record calibration was not qualified")
    lidar = manifest.get("lidar")
    if lidar is not None:
        model = lidar["model"]
        if (
            not _integer(model["width"], 1) or not _integer(model["height"], 1)
            or not _integer(model["period_ticks"], 1)
            or not _integer(model["offset_ticks"]) or model["offset_ticks"] >= model["period_ticks"]
            or not _integer(model["delivery_delay_ticks"])
        ):
            raise ValueError("invalid visual record lidar dimensions or timing")
        _array(model["mount_position_body_m"], (3,), "lidar mount")
        _rotation(model["rotation_bc"], "lidar mount rotation")


def _validate_trajectory(manifest, trajectory):
    steps = manifest["num_steps"]
    tick = trajectory["tick"]
    if (
        tick.dtype != np.dtype("uint64") or tick.shape != (steps + 1,)
        or not np.array_equal(tick, np.arange(steps + 1, dtype=np.uint64))
    ):
        raise ValueError("visual record ticks must be contiguous from zero")
    shapes = {key: (steps + 1, 2, 4 if key == "quaternions_xyzw" else 3) for key in _STATES}
    shapes.update({key: (steps + 1, 4 if key == "own_quaternion_xyzw" else 3) for key in _OWN})
    shapes["desired_gap_m"] = (steps + 1,)
    shapes.update({key: (steps, 2, 3) for key in _COMMANDS})
    for key, shape in shapes.items():
        array = trajectory[key]
        if array.dtype != np.dtype("float64"):
            raise ValueError(f"invalid visual record {key} dtype")
        _array(array, shape, key)
    for key, dtype, shape in (
        ("phase", np.uint8, (steps + 1,)),
        ("control_update", np.bool_, (steps,)),
    ):
        if trajectory[key].dtype != np.dtype(dtype) or trajectory[key].shape != shape:
            raise ValueError(f"invalid visual record {key}")
    if np.any(trajectory["phase"] > 6) or trajectory["phase"][-1] not in (3, 6):
        raise ValueError("invalid visual record supervisor phase")
    if np.any(np.linalg.norm(trajectory["positions_m"], axis=-1) == 0):
        raise ValueError("visual record contains a singular radius")
    _unit_quaternions(trajectory["quaternions_xyzw"], (steps + 1, 2, 4), "attitude")
    _unit_quaternions(trajectory["own_quaternion_xyzw"], (steps + 1, 4), "own attitude")
    for state, own in zip(_STATES, _OWN):
        initial = _array(manifest[f"initial_{state}"], trajectory[state][0].shape, state)
        if not np.allclose(initial, trajectory[state][0], atol=1e-12, rtol=0):
            raise ValueError(f"visual record initial {state} disagrees with manifest")
        if not np.array_equal(trajectory[state][:, 1], trajectory[own]):
            raise ValueError(f"visual record ideal own-navigation {own} disagrees with truth")
    for quantity, unit, limit in (
        ("forces", "n", "force_limits_n"), ("torques", "nm", "torque_limits_nm"),
    ):
        applied = trajectory[f"applied_{quantity}_body_{unit}"]
        if np.any(np.linalg.norm(applied, axis=-1) > np.asarray(manifest[limit]) + 1e-10):
            raise ValueError(f"visual record applied {quantity} exceed limits")


def _validate_events(events, manifest):
    if not isinstance(events, list) or not events:
        raise ValueError("visual records require an event list")
    previous = -1
    kinds = {"supervisor_transition", "physical_boundary", "safety_detection", "abort_request", "outcome"}
    for event in events:
        if (
            not isinstance(event, dict) or not _integer(event.get("tick"))
            or not previous <= event["tick"] <= manifest["num_steps"]
            or event.get("time_ns") != event["tick"] * manifest["dt_ns"]
            or not _integer(event.get("time_ns"))
            or type(event.get("phase")) is not int or not 0 <= event["phase"] <= 6
            or not isinstance(event.get("kind"), str) or event["kind"] not in kinds
            or not isinstance(event.get("reason"), str)
        ):
            raise ValueError("invalid visual record event")
        previous = event["tick"]
    if events[-1]["kind"] != "outcome" or events[-1]["tick"] != manifest["num_steps"]:
        raise ValueError("visual records must end with a timestamped outcome")


def _validate_observations(directory, manifest, rows):
    end_ns = manifest["num_steps"] * manifest["dt_ns"]
    period_ns = manifest["camera_period_ticks"] * manifest["dt_ns"]
    delay_ns = manifest["delivery_delay_ticks"] * manifest["dt_ns"]
    accepted = {}
    deliveries = {}
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict) or type(row.get("capture_id")) is not int
            or row["capture_id"] != index
            or not _integer(row.get("capture_time_ns"))
            or row["capture_time_ns"] != index * period_ns
            or row["capture_time_ns"] > end_ns
            or not _integer(row.get("scheduled_delivery_time_ns"))
            or row.get("scheduled_delivery_time_ns") != row["capture_time_ns"] + delay_ns
        ):
            raise ValueError("invalid visual record capture interval")
        image_path, checksum = row.get("image_path"), row.get("image_sha256")
        if image_path is None:
            if checksum is not None or row.get("fault") != "missing_capture":
                raise ValueError("missing visual record image must have a declared capture fault")
        else:
            path = _checked_file(directory, image_path, checksum)
            pixels = read_rgb(path)
            if pixels.shape != (manifest["camera"]["height"], manifest["camera"]["width"], 3):
                raise ValueError("visual record image dimensions disagree with camera")
        delivery = row.get("actual_delivery_time_ns")
        if delivery is None:
            if row["scheduled_delivery_time_ns"] <= end_ns:
                raise ValueError("visual record omits an image delivery before its end")
            if row.get("pose_estimate") is not None or row.get("navigation_accepted") is not None:
                raise ValueError("undelivered image cannot contain a consumed visual pose")
            continue
        if not _integer(delivery) or delivery != row["scheduled_delivery_time_ns"] or delivery > end_ns:
            raise ValueError("invalid visual record delivery interval")
        pose = row["pose_estimate"]
        if (
            not isinstance(pose, dict) or type(pose.get("valid")) is not bool
            or not _integer(pose.get("capture_time_ns"))
            or not _integer(pose.get("delivery_time_ns"))
            or pose.get("capture_time_ns") != row["capture_time_ns"]
            or pose.get("delivery_time_ns") != delivery
            or type(row.get("navigation_accepted")) is not bool
            or (row["navigation_accepted"] and not pose["valid"])
        ):
            raise ValueError("invalid delivered visual record pose")
        if pose["valid"]:
            _rotation(pose["rotation_ct"], "image-derived rotation")
            translation = _array(pose["translation_ct"], (3,), "image-derived position")
            if translation[2] <= 0 or image_path is None:
                raise ValueError("accepted image pose must lie in front of a saved camera image")
        corners = pose.get("image_corners")
        if corners is not None:
            corners = np.asarray(corners, dtype=np.float64)
            if corners.ndim != 2 or corners.shape[1] != 2 or not np.isfinite(corners).all():
                raise ValueError("invalid visual record detected corners")
        deliveries[delivery] = row["navigation_accepted"]
        if row["navigation_accepted"]:
            accepted[row["capture_time_ns"]] = delivery
    return accepted, deliveries


def _validate_lidar(directory, manifest, rows, accepted, deliveries):
    """Check scans and lidar poses; add lidar acceptances to the camera ones."""
    model, dt_ns = manifest["lidar"]["model"], manifest["dt_ns"]
    end_ns = manifest["num_steps"] * dt_ns
    for index, row in enumerate(rows):
        capture_ns = (index * model["period_ticks"] + model["offset_ticks"]) * dt_ns
        if (
            not isinstance(row, dict) or row.get("capture_id") != index
            or row.get("capture_time_ns") != capture_ns or capture_ns > end_ns
            or row.get("scheduled_delivery_time_ns")
            != capture_ns + model["delivery_delay_ticks"] * dt_ns
        ):
            raise ValueError("invalid visual record lidar capture interval")
        with np.load(_checked_file(directory, row.get("scan_path"), row.get("scan_sha256")),
                     allow_pickle=False) as scan:
            shape = (model["height"], model["width"])
            if (scan["ranges_m"].shape != shape or scan["retroreflective"].shape != shape
                    or scan["retroreflective"].dtype != np.dtype(np.bool_)):
                raise ValueError("visual record lidar scan dimensions disagree with the model")
        delivery = row.get("actual_delivery_time_ns")
        if delivery is None:
            if row["scheduled_delivery_time_ns"] <= end_ns:
                raise ValueError("visual record omits a lidar delivery before its end")
            continue
        pose = row["pose_estimate"]
        if (
            delivery != row["scheduled_delivery_time_ns"] or delivery > end_ns
            or not isinstance(pose, dict) or type(pose.get("valid")) is not bool
            or pose.get("capture_time_ns") != capture_ns
            or pose.get("delivery_time_ns") != delivery
            or type(row.get("navigation_accepted")) is not bool
            or (row["navigation_accepted"] and not pose["valid"])
        ):
            raise ValueError("invalid delivered visual record lidar pose")
        if pose["valid"]:
            _rotation(pose["rotation_lt"], "lidar-derived rotation")
            _array(pose["translation_lt"], (3,), "lidar-derived position")
            _array(pose["covariance"], (6, 6), "lidar-derived covariance")
        deliveries[delivery] = deliveries.get(delivery, False) or row["navigation_accepted"]
        if row["navigation_accepted"]:
            accepted[capture_ns] = delivery


def _validate_estimates(manifest, rows, accepted, deliveries):
    if len(rows) != manifest["num_steps"] + 1:
        raise ValueError("visual record must save an estimate decision for every physics tick")
    accepted_at = {delivery: capture for capture, delivery in accepted.items()}
    latest_capture = None
    for index, row in enumerate(rows):
        time_ns = index * manifest["dt_ns"]
        if time_ns in accepted_at:
            latest_capture = accepted_at[time_ns]
        delivered = deliveries.get(time_ns)
        # With lidar, another sensor's recent capture may cover a failed delivery.
        reported = (False, None) if delivered is False and "lidar" in manifest else (delivered,)
        if (
            not isinstance(row, dict) or row.get("time_ns") != time_ns
            or type(row.get("time_ns")) is not int
            or type(row.get("used_for_control")) is not bool
            or not any(row.get("visual_update") is value for value in reported)
        ):
            raise ValueError("invalid visual record estimate timing")
        target = row["target"]
        if target is None:
            continue
        if (
            not isinstance(target, dict) or target.get("time_ns") != time_ns
            or not _integer(target.get("time_ns"))
            or not _integer(target.get("capture_time_ns"))
            or target["capture_time_ns"] not in accepted
            or accepted[target["capture_time_ns"]] > time_ns
            or target["capture_time_ns"] != latest_capture
        ):
            raise ValueError("visual record estimate uses an unavailable observation")
        for key in ("position_m", "velocity_mps", "angular_velocity_inertial_radps"):
            _array(target[key], (3,), f"estimated {key}")
        if np.linalg.norm(target["position_m"]) == 0:
            raise ValueError("visual record estimated position has a singular radius")
        _unit_quaternions(target["quaternion_xyzw"], (4,), "estimated attitude")
        covariance = _array(target["covariance"], (12, 12), "estimated covariance")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0) or (
            np.linalg.eigvalsh(covariance).min() < -1e-9
        ):
            raise ValueError("visual record covariance must be symmetric positive semidefinite")
        age_ns = time_ns - target["capture_time_ns"]
        health = "fresh" if age_ns <= 1_200_000_000 else (
            "predicted" if age_ns <= 3_000_000_000 else "lost"
        )
        if target.get("health") != health:
            raise ValueError("visual record estimate health disagrees with observation age")


def _load_visual(directory):
    directory = Path(directory)
    try:
        manifest = _read_json(directory / "manifest.json")
        _validate_manifest(directory, manifest)
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            trajectory = {key: data[key] for key in (
                "tick", "phase", "desired_gap_m", "control_update", *_STATES, *_COMMANDS, *_OWN,
            )}
        _validate_trajectory(manifest, trajectory)
        events = _read_json(directory / "events.json")
        _validate_events(events, manifest)
        observations = _read_rows(directory / "observations.jsonl")
        accepted, deliveries = _validate_observations(directory, manifest, observations)
        lidar = []
        if "lidar" in manifest:
            lidar = _read_rows(directory / "lidar.jsonl")
            _validate_lidar(directory, manifest, lidar, accepted, deliveries)
        estimates = _read_rows(directory / "estimates.jsonl")
        _validate_estimates(manifest, estimates, accepted, deliveries)
    except (KeyError, TypeError, IndexError, OverflowError, OSError, BadZipFile, EOFError) as error:
        raise ValueError(f"invalid visual record: {error}") from error
    return manifest, trajectory, events, observations, estimates, lidar


def load_visual_run(directory):
    manifest, trajectory, *_ = _load_visual(directory)
    return manifest, trajectory


def load_visual_events(directory, manifest):
    events = _read_json(Path(directory) / "events.json")
    _validate_events(events, manifest)
    return events


def replay_visual_run(directory, save_only=False):
    from .visualization import export_replay, open_replay

    directory = Path(directory)
    manifest, trajectory, events, observations, estimates, lidar = _load_visual(directory)
    output = export_replay(
        manifest, trajectory, directory / "replay.rrd", events=events,
        observations=observations, estimates=estimates, image_root=directory, lidar=lidar,
    )
    if not save_only:
        open_replay(output)
    return output
