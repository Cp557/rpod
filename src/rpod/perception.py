"""Image-only pose estimates for the declared cooperative marker target."""

from functools import lru_cache

import numpy as np

from .camera import CameraModel, MARKER_CENTERS, marker_corners, project_points
from .observations import ImageObservation, PoseEstimate


ESTIMATOR_CONFIG = {
    "dictionary": "DICT_4X4_50",
    "corner_refinement": "CORNER_REFINE_APRILTAG",
    "detector_upscale": 2,
    "resize_interpolation": "INTER_CUBIC",
    "inverse_resize_coordinates": "(pixel + 0.5) / scale - 0.5",
    "minimum_marker_edge_px": 20.0,
    "ransac_method": "SOLVEPNP_EPNP",
    "ransac_iterations": 200,
    "ransac_reprojection_error_px": 2.0,
    "ransac_confidence": 0.999,
    "ransac_seed": 0,
    "refinement": "solvePnPRefineLM on RANSAC inliers",
    "minimum_inliers": 8,
    "minimum_inliers_per_depth_group": 4,
    "maximum_reprojection_rms_px": 1.0,
}


@lru_cache(maxsize=1)
def _detector():
    import cv2

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def estimate_pose(image: ImageObservation, camera: CameraModel | None = None) -> PoseEstimate:
    """Detect pixels and solve target-body-to-camera pose; no truth state is an input."""
    import cv2

    camera = camera or CameraModel()

    def invalid(reason, ids=(), inlier_count=0, rms=None, corners=None):
        return PoseEstimate(
            image.capture_time_ns, image.delivery_time_ns, False, reason=reason,
            marker_ids=tuple(ids), inlier_count=inlier_count,
            reprojection_rms_px=rms, image_corners=corners,
        )

    if image.pixels is None:
        return invalid("missing_image")
    pixels = np.asarray(image.pixels)
    if pixels.dtype != np.uint8 or pixels.shape not in (
        (camera.height, camera.width), (camera.height, camera.width, 3),
    ):
        return invalid("invalid_image")
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
    scale = ESTIMATOR_CONFIG["detector_upscale"]
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    detected, ids, _ = _detector().detectMarkers(enlarged)
    if ids is None:
        return invalid("insufficient_markers")
    objects, corners, corner_ids = [], [], []
    accepted_ids = []
    for points, marker_id in zip(detected, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id not in MARKER_CENTERS:
            continue
        points = (points.reshape(4, 2).astype(np.float64) + 0.5) / scale - 0.5
        if np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).min() < ESTIMATOR_CONFIG["minimum_marker_edge_px"]:
            continue
        if marker_id in accepted_ids:
            return invalid("duplicate_marker_ids")
        accepted_ids.append(marker_id)
        objects.extend(marker_corners(marker_id))
        corners.extend(points)
        corner_ids.extend([marker_id] * 4)
    if len(accepted_ids) < 2:
        return invalid("insufficient_markers", accepted_ids)
    if len({MARKER_CENTERS[key][1] for key in accepted_ids}) < 2:
        return invalid("one_depth_group", accepted_ids)
    objects = np.ascontiguousarray(objects, dtype=np.float64)
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    cv2.setRNGSeed(ESTIMATOR_CONFIG["ransac_seed"])
    try:
        solved, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            objects, corners, camera.K, np.zeros(5),
            iterationsCount=ESTIMATOR_CONFIG["ransac_iterations"],
            reprojectionError=ESTIMATOR_CONFIG["ransac_reprojection_error_px"],
            confidence=ESTIMATOR_CONFIG["ransac_confidence"], flags=cv2.SOLVEPNP_EPNP,
        )
        if not solved or inliers is None or len(inliers) < ESTIMATOR_CONFIG["minimum_inliers"]:
            return invalid("insufficient_inliers", accepted_ids)
        indexes = inliers.reshape(-1)
        inlier_ids = np.asarray(corner_ids)[indexes]
        depths = np.array([MARKER_CENTERS[int(key)][1] for key in inlier_ids])
        if len(np.unique(inlier_ids)) < 2 or any(
            np.count_nonzero(depths == depth) < ESTIMATOR_CONFIG["minimum_inliers_per_depth_group"]
            for depth in (-1.0, -1.3)
        ):
            return invalid("insufficient_depth_support", accepted_ids, len(indexes))
        rotation_vector, translation = cv2.solvePnPRefineLM(
            objects[indexes], corners[indexes], camera.K, np.zeros(5),
            rotation_vector, translation,
        )
        rotation = cv2.Rodrigues(rotation_vector)[0]
        translation = translation.reshape(3)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return invalid("nonfinite_pose", accepted_ids, len(indexes))
        camera_points = objects[indexes] @ rotation.T + translation
        face_normal = rotation @ np.array([0.0, -1.0, 0.0])
        if np.any(camera_points[:, 2] <= 0) or face_normal @ translation >= 0:
            return invalid("nonpositive_geometry", accepted_ids, len(indexes))
        projected = project_points(objects[indexes], rotation, translation, camera)
        rms = float(np.sqrt(np.mean(np.sum((projected - corners[indexes])**2, axis=1))))
        if rms > ESTIMATOR_CONFIG["maximum_reprojection_rms_px"]:
            return invalid("reprojection_error", accepted_ids, len(indexes), rms, corners[indexes])
    except (cv2.error, ValueError):
        return invalid("pnp_failed", accepted_ids)
    return PoseEstimate(
        image.capture_time_ns, image.delivery_time_ns, True,
        rotation_ct=rotation, translation_ct=translation,
        marker_ids=tuple(sorted(int(key) for key in np.unique(inlier_ids))),
        inlier_count=len(indexes), reprojection_rms_px=rms, image_corners=corners[indexes],
    )
