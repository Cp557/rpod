"""Rendered-image camera checks and the fixed M3 perception qualification suite."""

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .camera import CameraModel, camera_pose, project_points, relative_target_pose
from .frames import matrix_quaternion, quaternion_matrices
from .observations import ImageObservation, OwnNavigation
from .perception import estimate_pose
from .camera import earth_pose
from .recording import load_run, software_info, write_json
from .rendering import BlenderRenderer, realistic_scene
from .scenarios import ORBIT_RADIUS_M
from .vision_records import (
    degrade_image, json_value, new_output, read_rgb, sha256_file, write_jsonl, write_rgb,
)

RANGES_M = (4.5, 5.0, 10.0, 20.0, 30.0, 32.0)
RANGE_EDGES_CAMERA_M = (5.7, 8.45, 15.95, 25.95, 31.95, 1e9)
NEGATIVE_VARIANTS = ("empty", "occluded", "out_of_view", "unknown_ids")
# Realistic fixtures put the target 400 km above Earth. Along-track ("vbar"), Earth is below
# the image and the fixture Sun turns about the local vertical; from above ("rbar"), the
# camera looks straight down at Earth behind the target, with the Sun overhead.
EARTH_CENTERS_FIXTURE_M = {"vbar": (0.0, 0.0, -ORBIT_RADIUS_M), "rbar": (0.0, ORBIT_RADIUS_M, 0.0)}


def rotation(axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    half = math.radians(angle_deg) / 2
    return quaternion_matrices(np.r_[axis * math.sin(half), math.cos(half)])


def fixture_pose(gap, variant, camera):
    """Derive COM and camera from a specified actual port gap and body attitude."""
    angle = 1 if gap <= 5 else 3 if gap <= 10 else 5 if gap <= 20 else 8
    lateral = min(0.02 * gap, 0.5)
    variants = (
        ([1, 0, 0], 0, 0, 0),
        ([0, 0, 1], angle, lateral, 0),
        ([0, 0, 1], -angle, -lateral, 0),
        ([1, 0, 0], angle, 0, lateral),
        ([1, 0, 0], -angle, 0, -lateral),
        ([0, 1, 0], 5, 0, 0),
    )
    axis, degrees, x, z = variants[variant]
    body_rotation = rotation(axis, degrees)
    own_position = (np.array([0.0, -1.0, 0.0]) + [x, -gap, z]
                    - body_rotation @ [0, 0.5, 0])
    own = OwnNavigation(0, own_position, np.zeros(3), matrix_quaternion(body_rotation), np.zeros(3))
    r_camera, r_nc = camera_pose(own, camera)
    return r_nc.T, r_nc.T @ -r_camera, r_nc


def _pose_record(pose):
    return json_value(pose)


def _error(pose, rotation_ct, translation_ct):
    if not pose.valid:
        return {"translation_error_m": None, "rotation_error_deg": None,
                "translation_residual_c": None, "rotation_residual_c": None}
    residual_r = pose.rotation_ct @ rotation_ct.T
    vector, _ = cv2.Rodrigues(residual_r)
    translation = pose.translation_ct - translation_ct
    return {
        "translation_error_m": float(np.linalg.norm(translation)),
        "rotation_error_deg": math.degrees(float(np.linalg.norm(vector))),
        "translation_residual_c": translation.tolist(),
        "rotation_residual_c": vector.ravel().tolist(),
    }


def _performance(renderer, records, elapsed):
    durations = [record["estimation_seconds"] for record in records]
    return {
        "wall_seconds": elapsed,
        "image_count": len(records),
        "estimation_median_seconds": float(np.median(durations)) if durations else None,
        "estimation_p95_seconds": float(np.percentile(durations, 95)) if durations else None,
        "renderer": json_value(getattr(renderer, "performance", {})),
        "scope": "Blender rendering and image estimation; no real-time or compute-energy claim",
        **_image_transfer_performance(records),
    }


def _image_transfer_performance(records):
    durations = [row["image_transfer_seconds"] for row in records]
    return {
        "image_transfer_seconds": durations,
        "image_transfer_median_seconds": float(np.median(durations)) if durations else None,
        "image_transfer_p95_seconds": float(np.percentile(durations, 95)) if durations else None,
        "image_transfer_scope": (
            "Per-observation file read, PNG decode, and BGR-to-RGB conversion in Python; "
            "worker render timing includes PNG write"
        ),
    }


def _validate_m3_records(records):
    """Require each frozen fixture exactly once, including calibration and negatives."""
    expected = {}
    capture_id = 0
    for gap in RANGES_M:
        for variant in range(6):
            for light in range(3):
                identity = {
                    "gap_m": gap, "variant": variant, "light": light,
                    "partition": "calibration" if variant in (0, 5) else "test",
                    "positive": True, "degraded": False,
                }
                expected[capture_id] = identity
                expected[capture_id + 10_000] = {**identity, "degraded": True}
                capture_id += 1
    for variant in NEGATIVE_VARIANTS:
        for gap in RANGES_M:
            expected[capture_id] = {
                "gap_m": gap, "variant": variant, "partition": "test",
                "positive": False, "degraded": False, "target_present": variant != "empty",
            }
            capture_id += 1
    if len(records) != len(expected):
        raise ValueError(f"incomplete frozen M3 fixtures: expected {len(expected)} records")
    seen = set()
    for row in records:
        capture_id = row.get("capture_id")
        if type(capture_id) is not int or capture_id not in expected or capture_id in seen:
            raise ValueError("frozen M3 capture IDs must be complete and unique")
        seen.add(capture_id)
        for field, value in expected[capture_id].items():
            actual = row.get(field)
            if actual != value or (type(value) in (bool, int) and type(actual) is not type(value)):
                raise ValueError(f"frozen M3 fixture {capture_id} has inconsistent {field}")


def _qualify(records):
    _validate_m3_records(records)
    gates = {}
    bins = {}
    for degraded in (False, True):
        label = "degraded" if degraded else "clean"
        selected = [row for row in records if row["partition"] == "test"
                    and row["positive"] and row["degraded"] == degraded]
        coverage = sum(row["pose"]["valid"] for row in selected) / len(selected)
        gates[f"{label}_coverage"] = coverage >= (0.9 if degraded else 0.95)
        bins[label] = {"coverage": coverage, "by_gap": {}}
        for gap in RANGES_M:
            subset = [row for row in selected if row["gap_m"] == gap]
            valid = [row for row in subset if row["pose"]["valid"]]
            rate = len(valid) / len(subset)
            translation = (
                float(np.percentile([row["translation_error_m"] for row in valid], 95))
                if valid else None
            )
            angle = (
                float(np.percentile([row["rotation_error_deg"] for row in valid], 95))
                if valid else None
            )
            near = gap <= 10
            position_limit = (0.1 if near else 0.3) if degraded else (0.05 if near else 0.2)
            angle_limit = (2.0 if near else 3.0) if degraded else (1.0 if near else 2.0)
            passed = (
                rate >= 0.9 and translation is not None
                and translation <= position_limit and angle <= angle_limit
            )
            gates[f"{label}_{gap:g}m"] = passed
            bins[label]["by_gap"][str(gap)] = {
                "count": len(subset), "accepted": len(valid), "coverage": rate,
                "p95_translation_m": translation, "p95_rotation_deg": angle,
                "position_limit_m": position_limit, "rotation_limit_deg": angle_limit,
            }
    negatives = [row for row in records if not row["positive"]]
    gates["negative_rejections"] = not any(row["pose"]["valid"] for row in negatives)
    accepted = [row for row in records if row["positive"] and row["pose"]["valid"]]
    gates["no_gross_errors"] = all(
        row["translation_error_m"] <= 1 and row["rotation_error_deg"] <= 10
        for row in accepted
    )
    case_fields = (
        "capture_id", "image", "gap_m", "variant", "light", "partition", "degraded",
        "translation_error_m", "rotation_error_deg",
    )

    def case(row):
        return {**{key: row.get(key) for key in case_fields},
                "reason": row["pose"].get("reason", "")}

    return {"qualification_passed": all(gates.values()), "gates": gates,
            "metrics": bins, "negative_count": len(negatives),
            "negative_accepted": sum(row["pose"]["valid"] for row in negatives),
            "rejected_positive_cases": [
                case(row) for row in records if row["positive"] and not row["pose"]["valid"]
            ],
            "worst_accepted_cases": {
                metric: [
                    case(row)
                    for row in sorted(accepted, key=lambda row: row[field], reverse=True)[:5]
                ]
                for metric, field in (
                    ("translation", "translation_error_m"), ("rotation", "rotation_error_deg")
                )
            }}


def _calibration(records, report):
    from .perception import ESTIMATOR_CONFIG
    translation_variance, rotation_variance = [], []
    for gap in RANGES_M:
        subset = [row for row in records if row["partition"] == "calibration"
                  and row["gap_m"] == gap and row["pose"]["valid"]]
        if not subset:
            report["gates"][f"calibration_{gap:g}m"] = False
            report["qualification_passed"] = False
            translation_variance.append([0.002**2, 0.002**2, 0.01**2])
            rotation_variance.append([math.radians(0.25)**2] * 3)
            continue
        pos = np.asarray([row["translation_residual_c"] for row in subset])
        rot = np.asarray([row["rotation_residual_c"] for row in subset])
        translation_variance.append(
            np.maximum(np.mean(pos**2, axis=0), np.array([0.002, 0.002, 0.01])**2).tolist()
        )
        rotation_variance.append(
            np.maximum(np.mean(rot**2, axis=0), math.radians(0.25)**2).tolist()
        )
    return {"schema_version": 1, "range_bins": list(RANGE_EDGES_CAMERA_M),
            "translation_variance": translation_variance,
            "rotation_variance": rotation_variance,
            "qualification_passed": report["qualification_passed"],
            "camera": CameraModel().to_dict(), "estimator": ESTIMATOR_CONFIG,
            "source": (
                "M3 calibration partition only; per-range mean-square camera-frame "
                "residuals with floors"
            )}


def perception_eval(output, source=None, blender_path=None, scene="simple", earth="vbar"):
    """Render and score the M3 fixtures (or a recorded M2 run) in the chosen scene.

    In the realistic scene, `earth` places Earth for the fixtures (EARTH_CENTERS_FIXTURE_M).
    """
    if scene not in ("simple", "realistic") or earth not in EARTH_CENTERS_FIXTURE_M:
        raise ValueError(f"unknown render scene or Earth geometry: {scene!r}, {earth!r}")
    source_record = None
    if source is not None:
        source_record = load_run(source)
        if source_record[0]["schema_version"] != 2:
            raise ValueError("perception-eval --source requires an M2 schema-2 run")
    output = new_output(output)
    (output / "images").mkdir()
    (output / "truth").mkdir()
    camera = CameraModel()
    scene_spec = realistic_scene() if scene == "realistic" else None
    scene_fields = {"scene": scene, "earth_geometry": earth} if scene_spec else {}

    def fixture_earth(rotation_ct, translation_ct, r_nc):
        """Fixture targets keep the fixture's inertial axes, so Earth shares their rotation."""
        if scene_spec is None:
            return None
        return rotation_ct, translation_ct + r_nc.T @ np.asarray(EARTH_CENTERS_FIXTURE_M[earth])

    started = time.perf_counter()
    geometry_report = (
        camera_check(output / "camera-check", blender_path, scene_spec) if source is None else None
    )
    records, truths = [], []

    def observe(path, capture_id, rotation_ct, translation_ct, metadata):
        transfer_started = time.perf_counter()
        image = read_rgb(path)
        image_transfer_seconds = time.perf_counter() - transfer_started
        observation = ImageObservation(
            capture_id,
            int(metadata.get("capture_time_ns", capture_id * 1_000_000_000)),
            int(metadata.get("capture_time_ns", capture_id * 1_000_000_000)),
            image,
        )
        start = time.perf_counter()
        pose = estimate_pose(observation, camera)
        estimation_seconds = time.perf_counter() - start
        record = {**metadata, "capture_id": capture_id, "image": str(path.relative_to(output)),
                  "image_sha256": sha256_file(path), "pose": _pose_record(pose),
                  "estimation_seconds": estimation_seconds,
                  "image_transfer_seconds": image_transfer_seconds,
                  **_error(pose, rotation_ct, translation_ct)}
        records.append(record)
        truths.append({"capture_id": capture_id, "rotation_ct": rotation_ct.tolist(),
                       "translation_ct": translation_ct.tolist(), **metadata,
                       **_error(pose, rotation_ct, translation_ct)})

    with BlenderRenderer(output, blender_path=blender_path, camera=camera,
                         scene=scene_spec) as renderer:
        if source is None:
            capture_id = 0
            for gap in RANGES_M:
                print(f"Rendering M3 fixtures at {gap:g} m", flush=True)
                for variant in range(6):
                    rotation_ct, translation_ct, r_nc = fixture_pose(gap, variant, camera)
                    for light_index, degrees in enumerate((-30, 0, 30)):
                        path = output / "images" / f"{capture_id:04d}.png"
                        light_n = rotation([0, 0, 1], degrees) @ [0, 1, 0]
                        renderer.render(
                            rotation_ct, translation_ct, path,
                            light_direction_c=r_nc.T @ light_n,
                            earth=fixture_earth(rotation_ct, translation_ct, r_nc),
                        )
                        metadata = {"gap_m": gap, "variant": variant, "light": light_index,
                                    "partition": "calibration" if variant in (0, 5) else "test",
                                    "positive": True, "degraded": False}
                        observe(path, capture_id, rotation_ct, translation_ct, metadata)
                        degraded_path = output / "images" / f"{capture_id:04d}-degraded.png"
                        write_rgb(degraded_path, degrade_image(read_rgb(path), 31_000 + capture_id))
                        observe(degraded_path, capture_id + 10_000, rotation_ct, translation_ct,
                                {**metadata, "degraded": True})
                        capture_id += 1
            for variant in NEGATIVE_VARIANTS:
                for index in range(6):
                    gap = RANGES_M[index]
                    r_ct, t_ct, r_nc = fixture_pose(gap, 0, camera)
                    path = output / "images" / f"negative-{variant}-{index}.png"
                    renderer.render(
                        r_ct, t_ct, path, light_direction_c=r_nc.T @ [0, 1, 0],
                        variant=variant, earth=fixture_earth(r_ct, t_ct, r_nc),
                    )
                    label_t = t_ct.copy()
                    if variant == "out_of_view":
                        label_t[0] += 4 * abs(label_t[2]) + 10
                    observe(path, capture_id, r_ct, label_t, {"gap_m": gap, "variant": variant,
                            "partition": "test", "positive": False, "degraded": False,
                            "target_present": variant != "empty"})
                    capture_id += 1
            report = _qualify(records)
            report["gates"]["camera_geometry"] = geometry_report["qualification_passed"]
            report["qualification_passed"] = all(report["gates"].values())
            calibration = {**_calibration(records, report), **scene_fields}
            write_json(output / "calibration.json", calibration)
        else:
            manifest, trajectory = source_record
            from .frames import port_relative_state

            port_r, _, _ = port_relative_state(
                trajectory["positions_m"], trajectory["velocities_mps"],
                trajectory["quaternions_xyzw"], trajectory["angular_velocities_body_radps"],
                manifest["docking_ports_body_m"],
            )
            eligible = np.flatnonzero((-port_r[:, 1] >= 5) & (-port_r[:, 1] <= 30))
            if not len(eligible):
                raise ValueError("source run contains no 5–30 m port-gap samples")
            indexes = eligible[
                np.unique(np.linspace(0, len(eligible)-1, min(24, len(eligible)), dtype=int))
            ]
            sun_n = quaternion_matrices(trajectory["quaternions_xyzw"][indexes[0], 1]) @ [0, 1, 0]
            for capture_id, index in enumerate(indexes):
                own = OwnNavigation(
                    int(trajectory["tick"][index]) * manifest["dt_ns"],
                    trajectory["positions_m"][index, 1], trajectory["velocities_mps"][index, 1],
                    trajectory["quaternions_xyzw"][index, 1],
                    trajectory["angular_velocities_body_radps"][index, 1],
                )
                r_ct, t_ct = relative_target_pose(own, trajectory["positions_m"][index, 0],
                                                 trajectory["quaternions_xyzw"][index, 0], camera)
                _, r_nc = camera_pose(own, camera)
                path = output / "images" / f"{capture_id:04d}.png"
                renderer.render(r_ct, t_ct, path, light_direction_c=r_nc.T @ sun_n,
                                earth=earth_pose(own, camera) if scene_spec else None)
                observe(path, capture_id, r_ct, t_ct, {
                    "gap_m": -float(port_r[index, 1]), "capture_time_ns": own.time_ns,
                    "partition": "source", "positive": True, "degraded": False,
                })
            report = {"source": str(Path(source).resolve()), "samples": len(records),
                      "accepted": sum(row["pose"]["valid"] for row in records)}
        performance = _performance(renderer, records, time.perf_counter()-started)
        provenance = getattr(renderer, "provenance", {})
    manifest = {"artifact_type": "perception_dataset", "schema_version": 1,
                "camera": camera.to_dict(), "software": software_info(),
                "opencv": cv2.__version__, "rendering": json_value(provenance),
                "fixture_spec": {
                    "ranges_m": list(RANGES_M), "pose_variants": 6,
                    "light_angles_deg": [-30, 0, 30], "calibration_variants": [0, 5],
                    "degradation": {"gaussian_sigma_dn": 2, "blur_sigma_px": 0.5,
                                    "noise_seed": "31000 + clean capture_id"},
                },
                "implementation_sha256": {
                    name: sha256_file(Path(__file__).with_name(name))
                    for name in ("camera.py", "perception.py", "blender_worker.py")
                },
                "suite": "m3" if source is None else "recorded_m2", "frame_count": len(records),
                "source_hashes": None if source is None else {
                    name: sha256_file(Path(source) / name)
                    for name in ("manifest.json", "trajectory.npz")
                }, **scene_fields}
    if scene_spec and source is None:
        manifest["fixture_spec"]["earth_center_fixture_m"] = EARTH_CENTERS_FIXTURE_M[earth]
    write_json(output / "dataset.json", manifest)
    public_keys = (
        "capture_id", "image", "image_sha256", "pose", "estimation_seconds",
        "image_transfer_seconds",
    )
    write_jsonl(output / "observations.jsonl", [
        {key: row[key] for key in public_keys} for row in records
    ])
    write_jsonl(output / "truth" / "poses.jsonl", truths)
    write_json(output / "report.json", report)
    write_json(output / "performance.json", performance)
    return report


def camera_check(output, blender_path=None, scene=None):
    output = new_output(output)
    started = time.perf_counter()
    camera = CameraModel()
    points = np.array([[-0.6, -0.1, 0.5], [0.55, 0.15, 0.45], [-0.5, 0.2, -0.5],
                       [0.6, -0.15, -0.4], [0.05, -0.3, 0.05], [0.1, 0.3, 0.7]])
    r_ct = rotation([0, 0, 1], 12) @ rotation([1, 0, 0], 8)
    t_ct = np.array([0.05, -0.1, 3.5])
    expected = project_points(points, r_ct, t_ct, camera)
    with BlenderRenderer(output, blender_path=blender_path, camera=camera,
                         scene=scene) as renderer:
        result = renderer.render(r_ct, t_ct, output / "calibration.png", calibration_points=points)
        predicted = np.asarray(result["projected_pixels"])
        projection_error = np.linalg.norm(predicted-expected, axis=-1)
        image = read_rgb(output / "calibration.png").astype(float)
        colors = np.asarray(result["calibration_colors"][:len(points)], dtype=float)
        if colors.max() <= 1:
            colors *= 255
        errors, centroids = [], []
        for index, point in enumerate(expected):
            # A bounded ROI around each independently projected dot avoids background color ties.
            x, y = point
            x0, x1 = max(0, int(x)-20), min(camera.width, int(x)+21)
            y0, y1 = max(0, int(y)-20), min(camera.height, int(y)+21)
            crop = image[y0:y1, x0:x1]
            direction = colors[index] / np.linalg.norm(colors[index])
            magnitude = np.linalg.norm(crop, axis=-1)
            similarity = np.sum(crop*direction, axis=-1) / np.maximum(magnitude, 1)
            weights = np.where((similarity > 0.995) & (magnitude > 50), magnitude, 0)
            if not weights.sum():
                raise ValueError(f"rendered calibration dot {index} not found")
            yy, xx = np.indices(weights.shape)
            centroid = np.array([
                x0+(weights*xx).sum()/weights.sum(),
                y0+(weights*yy).sum()/weights.sum(),
            ])
            centroids.append(centroid.tolist())
            errors.append(float(np.linalg.norm(centroid-point)))
        report = {"projection_max_error_px": float(projection_error.max()),
                  "raster_max_error_px": max(errors), "raster_errors_px": errors,
                  "expected_pixels": expected.tolist(), "blender_pixels": predicted.tolist(),
                  "rendered_centroids": centroids, "camera": camera.to_dict(),
                  "performance": _performance(renderer, [], time.perf_counter() - started)}
        report["qualification_passed"] = (
            report["projection_max_error_px"] <= 0.05 and report["raster_max_error_px"] <= 0.5
        )
    write_json(output / "report.json", report)
    return report
