"""Natural-feature camera pose for the unmarked target: keypoints, data, inference and PnP.

- `select_keypoints` picks 20 mesh vertices by farthest-point sampling: 12 over the whole
  satellite and 8 around the docking face, which stays in view at the 5 m hold point.
- `render_dataset` renders the unmarked target at random poses, ranges, Sun and Earth with
  `UnmarkedRenderer`, labelling each keypoint's pixel and its visibility by a mesh ray cast.
  The held-out `test` split uses conditions training never sees: Sun phase angles of 70-100
  degrees and 3 DN / 0.75 px degradation.
- `KeypointEstimator` runs the ONNX heatmap network (trained by `tools/keypoints`) through
  OpenCV DNN, refines each peak to sub-pixel, and solves PnP with RANSAC and LM refinement,
  rejecting poses whose predicted uncertainty is large.
- `calibrate_keypoints` turns validation errors into the M3 range-binned calibration that
  `VisualNavigation` already uses for camera measurement covariance.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .camera import earth_pose, project_points, relative_target_pose
from .frames import quaternion_matrices
from .observations import OwnNavigation, PoseEstimate
from .vision_records import degrade_image, read_rgb, sha256_file

KEYPOINTS_PATH = Path("models/m11-keypoints.json")
INPUT_SIZE = 512
HEATMAP_STRIDE = 4
# Heatmap pixel h sits at network-input pixel HEATMAP_STRIDE * h + HEATMAP_OFFSET; the
# training tool labels with the same convention.
HEATMAP_OFFSET = (HEATMAP_STRIDE - 1) / 2
CONFIDENCE = 0.3
MINIMUM_KEYPOINTS = 6
RANSAC_PX = 6.0
RMS_LIMIT_PX = 4.0
# Largest predicted pose uncertainty accepted: root-trace of the translation and rotation-vector
# covariance that the inliers' reprojection residuals and Jacobian imply.
POSE_SIGMA_M, POSE_SIGMA_DEG = 0.25, 1.0
VISIBILITY_TOLERANCE_M = 0.03
EARTH_RADIUS_M, ORBIT_RADIUS_M = 6_378_137.0, 6_778_137.0
# Earth's angular radius from orbit: a Sun direction inside it eclipses the target.
ECLIPSE_DEG = math.degrees(math.asin(EARTH_RADIUS_M / ORBIT_RADIUS_M)) + 2.0
SPLITS = {"train": (0.0, 70.0, None), "val": (0.0, 70.0, (2.0, 0.5)),
          "test": (70.0, 100.0, (3.0, 0.75))}


def _farthest(points, count, start):
    chosen = [start]
    distance = np.linalg.norm(points - points[start], axis=1)
    for _ in range(count - 1):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return points[chosen]


def select_keypoints(vertices):
    """12 farthest-point vertices over the satellite and 8 within 1.2 m of the docking axis
    on its face (y < -0.9 m)."""
    vertices = np.unique(np.round(vertices, 6), axis=0)
    spread = _farthest(vertices, 12, int(np.argmax(np.linalg.norm(vertices, axis=1))))
    face = vertices[(vertices[:, 1] < -0.9) & (np.abs(vertices[:, 0]) < 1.2)
                    & (np.abs(vertices[:, 2]) < 1.2)]
    near = _farthest(face, 8, int(np.argmax(np.linalg.norm(face[:, [0, 2]], axis=1))))
    return np.vstack([spread, near])


def keypoint_labels(points_body, rotation_ct, translation_ct, camera, mesh):
    """Pixel of every keypoint, and whether it is in the image and not hidden by the mesh."""
    pixels = project_points(points_body, rotation_ct, translation_ct, camera)
    center = -np.asarray(rotation_ct).T @ np.asarray(translation_ct)
    rays = points_body - center
    distance = np.linalg.norm(rays, axis=1)
    hits = mesh.cast_rays(np.repeat(center[None], len(rays), axis=0),
                          np.ascontiguousarray(rays / distance[:, None]), 1e4)
    inside = ((pixels >= 0) & (pixels <= [camera.width - 1, camera.height - 1])).all(axis=1)
    depth = (np.asarray(rotation_ct) @ points_body.T).T[:, 2] + translation_ct[2]
    unoccluded = np.nan_to_num(hits, nan=np.inf) >= distance - VISIBILITY_TOLERANCE_M
    return pixels, inside & unoccluded & (depth > 0)


def _random_rotation(rng):
    quaternion = rng.normal(size=4)
    return quaternion_matrices(quaternion / np.linalg.norm(quaternion))


def _sample(rng, camera, split):
    """Own state, target pose, Sun ray and Earth pose for one image of `split`. A geometry
    whose Sun phase window is all eclipsed (the camera looks along nadir) is redrawn."""
    low, high, _ = SPLITS[split]
    while True:
        position = _random_rotation(rng)[:, 0] * ORBIT_RADIUS_M
        quaternion = rng.normal(size=4)
        own = OwnNavigation(int(rng.integers(0, 6 * 3600)) * 1_000_000_000, position,
                            np.zeros(3), quaternion / np.linalg.norm(quaternion), np.zeros(3))
        distance = rng.uniform(5.5, 35.0)
        lateral = rng.uniform(-0.15, 0.15, 2) * distance
        translation_ct = np.array([lateral[0], lateral[1], distance])
        rotation_ct = _random_rotation(rng)
        rotation_nc = quaternion_matrices(own.quaternion_xyzw) @ np.asarray(camera.rotation_bc)
        target_n = position + rotation_nc @ translation_ct
        nadir_c = rotation_nc.T @ (-target_n / np.linalg.norm(target_n))
        toward_camera = -translation_ct / np.linalg.norm(translation_ct)
        for _ in range(1000):
            sun = rng.normal(size=3)
            sun /= np.linalg.norm(sun)
            phase = math.degrees(math.acos(np.clip(sun @ toward_camera, -1, 1)))
            eclipsed = math.degrees(math.acos(np.clip(sun @ nadir_c, -1, 1))) < ECLIPSE_DEG
            if low <= phase < high and not eclipsed:
                return own, rotation_ct, translation_ct, -sun, phase


def render_dataset(output, counts, seed, camera, mesh, keypoints, renderer_factory):
    """Render `counts` images per split; write PNGs and one labels.jsonl per split."""
    output = Path(output)
    with renderer_factory(output / "renderer") as renderer:
        for split, count in counts.items():
            folder = output / split
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng([seed, list(SPLITS).index(split)])
            with (folder / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as labels:
                for index in range(count):
                    own, rotation_ct, translation_ct, light, phase = _sample(rng, camera, split)
                    path = folder / f"{index:06d}.png"
                    renderer.render(rotation_ct, translation_ct, path, light_direction_c=light,
                                    earth=earth_pose(own, camera))
                    pixels, visible = keypoint_labels(keypoints, rotation_ct, translation_ct,
                                                      camera, mesh)
                    labels.write(json.dumps({
                        "image": path.name, "rotation_ct": rotation_ct.tolist(),
                        "translation_ct": translation_ct.tolist(), "phase_deg": phase,
                        "keypoints_px": pixels.tolist(), "visible": visible.tolist(),
                    }) + "\n")
                    if index % 500 == 0:
                        print(f"{split} {index}/{count}", flush=True)


def load_keypoints(path=KEYPOINTS_PATH):
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["points_body_m"])


def _peaks(heatmaps):
    """Sub-pixel peak (u, v) in heatmap pixels and peak value for each channel."""
    count, height, width = heatmaps.shape
    flat = heatmaps.reshape(count, -1).argmax(axis=1)
    rows, columns = np.divmod(flat, width)
    peaks = np.zeros((count, 2))
    for index, (row, column) in enumerate(zip(rows, columns)):
        offsets = []
        for axis, (position, size) in enumerate(((column, width), (row, height))):
            if 0 < position < size - 1:
                before = heatmaps[index, row - (axis == 1), column - (axis == 0)]
                after = heatmaps[index, row + (axis == 1), column + (axis == 0)]
                center = heatmaps[index, row, column]
                curvature = before - 2 * center + after
                offsets.append(0.5 * (before - after) / curvature if curvature < 0 else 0.0)
            else:
                offsets.append(0.0)
        peaks[index] = [column + offsets[0], row + offsets[1]]
    return peaks, heatmaps.reshape(count, -1).max(axis=1)


class KeypointEstimator:
    """Heatmap network plus PnP, returning a `PoseEstimate` like marker perception."""

    def __init__(self, model_path, keypoints, camera):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.keypoints, self.camera = np.asarray(keypoints, dtype=np.float64), camera
        self.matrix = camera.K

    def detect(self, pixels):
        """Keypoint pixels in the full image and their confidences."""
        image = cv2.resize(pixels, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(image.astype(np.float32) / 255.0, 1.0)
        blob = (blob - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) \
            / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        self.net.setInput(blob)
        heatmaps = self.net.forward()[0]
        peaks, confidence = _peaks(heatmaps)
        scale = np.array([self.camera.width / INPUT_SIZE, self.camera.height / INPUT_SIZE])
        network = peaks * HEATMAP_STRIDE + HEATMAP_OFFSET
        return (network + 0.5) * scale - 0.5, confidence

    def estimate(self, image):
        capture, delivery = image.capture_time_ns, image.delivery_time_ns
        if image.pixels is None:
            return PoseEstimate(capture, delivery, False, reason="missing_image")
        pixels, confidence = self.detect(image.pixels)
        found = confidence >= CONFIDENCE
        if found.sum() < MINIMUM_KEYPOINTS:
            return PoseEstimate(capture, delivery, False, reason="insufficient_keypoints",
                                inlier_count=int(found.sum()))
        objects = np.ascontiguousarray(self.keypoints[found])
        points = np.ascontiguousarray(pixels[found])
        cv2.setRNGSeed(0)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects, points, self.matrix, None, iterationsCount=200,
            reprojectionError=RANSAC_PX, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < MINIMUM_KEYPOINTS:
            return PoseEstimate(capture, delivery, False, reason="insufficient_inliers",
                                inlier_count=0 if inliers is None else len(inliers))
        inliers = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(objects[inliers], points[inliers], self.matrix, None,
                                          rvec, tvec)
        rotation = cv2.Rodrigues(rvec)[0]
        translation = tvec[:, 0]
        projected = project_points(objects[inliers], rotation, translation, self.camera)
        rms = float(np.sqrt(np.mean(np.sum((projected - points[inliers]) ** 2, axis=1))))
        if not np.isfinite(rotation).all() or translation[2] <= 0:
            return PoseEstimate(capture, delivery, False, reason="nonpositive_geometry")
        if rms > RMS_LIMIT_PX:
            return PoseEstimate(capture, delivery, False, reason="reprojection_error",
                                inlier_count=len(inliers), reprojection_rms_px=rms)
        jacobian = cv2.projectPoints(objects[inliers], rvec, tvec, self.matrix, None)[1][:, :6]
        variance = np.sum((projected - points[inliers]) ** 2) / (2 * len(inliers) - 6)
        covariance = variance * np.linalg.inv(jacobian.T @ jacobian)
        if (np.sqrt(np.trace(covariance[3:, 3:])) > POSE_SIGMA_M
                or math.degrees(np.sqrt(np.trace(covariance[:3, :3]))) > POSE_SIGMA_DEG):
            return PoseEstimate(capture, delivery, False, reason="uncertain_pose",
                                inlier_count=len(inliers), reprojection_rms_px=rms)
        return PoseEstimate(capture, delivery, True, rotation, translation,
                            inlier_count=len(inliers), reprojection_rms_px=rms)


def evaluate(dataset, split, estimator, seed=0):
    """Per-image pose errors (camera-axis translation, rotation vector) against labels."""
    _, _, degradation = SPLITS[split]
    rows = []
    for index, line in enumerate((Path(dataset) / split / "labels.jsonl").open(encoding="utf-8")):
        label = json.loads(line)
        pixels = read_rgb(Path(dataset) / split / label["image"])
        if degradation is not None and (split == "test" or index % 2 == 0):
            pixels = degrade_image(pixels, [seed, index], *degradation)
        pose = estimator.estimate(_Image(pixels))
        truth_r, truth_t = np.asarray(label["rotation_ct"]), np.asarray(label["translation_ct"])
        row = {"range_m": float(truth_t[2]), "phase_deg": label["phase_deg"], "valid": pose.valid,
               "reason": pose.reason}
        if pose.valid:
            error = cv2.Rodrigues(pose.rotation_ct @ truth_r.T)[0][:, 0]
            row.update(translation_error_m=(pose.translation_ct - truth_t).tolist(),
                       rotation_error_rad=error.tolist())
        rows.append(row)
    return rows


class _Image:
    capture_time_ns = delivery_time_ns = 0

    def __init__(self, pixels):
        self.pixels = pixels


def calibrate_keypoints(rows, range_bins, camera_settings, sources):
    """M3-format calibration: per-range-bin mean-square errors in camera axes, from `rows`.
    Like M3, it qualifies only if no accepted pose is off by more than 1 m or 10 degrees."""
    valid = [row for row in rows if row["valid"]]
    ranges = np.array([row["range_m"] for row in valid])
    translation = np.array([row["translation_error_m"] for row in valid])
    rotation = np.array([row["rotation_error_rad"] for row in valid])
    bins = np.searchsorted(range_bins, ranges)
    translation_variance, rotation_variance = [], []
    for index in range(len(range_bins)):
        chosen = bins == index
        source = chosen if chosen.sum() >= 30 else np.ones(len(valid), bool)
        translation_variance.append(np.mean(translation[source] ** 2, axis=0).tolist())
        rotation_variance.append(np.mean(rotation[source] ** 2, axis=0).tolist())
    gross = int(np.sum((np.linalg.norm(translation, axis=1) > 1)
                       | (np.degrees(np.linalg.norm(rotation, axis=1)) > 10)))
    return {"schema_version": 1, "range_bins": list(range_bins),
            "translation_variance": translation_variance, "rotation_variance": rotation_variance,
            "qualification_passed": gross == 0, "gross_errors": gross,
            "scene": "realistic", "camera": camera_settings,
            "estimator": {"kind": "keypoint heatmaps and PnP", **sources},
            "source": "validation renders of the unmarked target", "samples": len(valid),
            "attempts": len(rows), "coverage": len(valid) / len(rows)}


def error_summary(rows, range_bins):
    """Coverage and pose-error median, P95 and maximum per camera range bin."""

    def stats(values):
        return ({"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                 "max": float(np.max(values))} if values else None)

    summary, lower = [], 0.0
    for upper in range_bins:
        chosen = [row for row in rows if lower < row["range_m"] <= upper]
        valid = [row for row in chosen if row["valid"]]
        summary.append({
            "range_m": [lower, upper], "images": len(chosen),
            "coverage": len(valid) / len(chosen) if chosen else None,
            "translation_m": stats([float(np.linalg.norm(row["translation_error_m"]))
                                    for row in valid]),
            "rotation_deg": stats([math.degrees(np.linalg.norm(row["rotation_error_rad"]))
                                   for row in valid])})
        lower = upper
    return summary


def model_provenance(model_path, keypoints_path):
    return {"model": Path(model_path).as_posix(), "model_sha256": sha256_file(model_path),
            "keypoints_sha256": sha256_file(keypoints_path)}
