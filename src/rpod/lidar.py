"""Simulated flash lidar, reflector-derived target pose, and its navigation measurement.

The lidar sees the same 2 m cube and marker plates that Blender renders, plus four
retroreflectors on the docking face. The reflectors exist only in this model; camera
images do not show them.
"""

from dataclasses import asdict, dataclass
from itertools import permutations
import math

import cv2
import numpy as np

from .camera import MARKER_CENTERS, MARKER_QUIET_ZONE_M, MARKER_SIDE_M, camera_pose
from .frames import matrix_quaternion
from .navigation import VisualNavigation, _Measurement, _skew
from .observations import LidarPose

TARGET_HALF_SIZE_M = 1.0
REFLECTOR_RADIUS_M = 0.06
REFLECTOR_STANDOFF_M = 0.002
REFLECTOR_ACCEPTANCE_DEG = 45.0
# On the -Y docking face, clear of the marker pads, with distinct pairwise spacings.
REFLECTORS_BODY_M = np.array([
    [0.0, -1.0, 0.85], [0.8, -1.0, 0.0], [0.0, -1.0, -0.45], [-0.6, -1.0, 0.0],
])
REFLECTOR_RMS_LIMIT_M = 0.05
CENTROID_SIGMA_FLOOR_M = 0.002


@dataclass(frozen=True)
class LidarModel:
    """Pinhole flash lidar with the camera's axes: +X right, +Y down, +Z forward."""

    width: int = 256
    height: int = 256
    hfov_deg: float = 30.0
    mount_position_body_m: tuple = (-0.2, 0.55, 0.2)
    rotation_bc: tuple = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    range_sigma_m: float = 0.02
    max_range_m: float = 100.0
    period_ticks: int = 10
    offset_ticks: int = 5
    delivery_delay_ticks: int = 2

    def directions(self) -> np.ndarray:
        """Unit ray through each pixel center, shape (height, width, 3)."""
        focal = self.width / (2 * math.tan(math.radians(self.hfov_deg) / 2))
        u, v = np.meshgrid((np.arange(self.width) - (self.width - 1) / 2) / focal,
                           (np.arange(self.height) - (self.height - 1) / 2) / focal)
        rays = np.stack((u, v, np.ones_like(u)), axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def to_dict(self) -> dict:
        return asdict(self)


def scan_target(rotation_lt, translation_lt, lidar, rng):
    """Ray-cast the target pose; float32 ranges (NaN without a return) and retro flags.

    Range noise is Gaussian on every return. A reflector is retroreflective only when
    the lidar lies within its acceptance cone; otherwise it returns like the face.
    """
    rays = lidar.directions().reshape(-1, 3)
    origin = -np.asarray(rotation_lt).T @ np.asarray(translation_lt)
    direction = rays @ np.asarray(rotation_lt)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = (-TARGET_HALF_SIZE_M - origin) / direction
        upper = (TARGET_HALF_SIZE_M - origin) / direction
        near = np.minimum(lower, upper).max(axis=1)
        far = np.maximum(lower, upper).min(axis=1)
        best = np.where((near <= far) & (near > 0), near, np.inf)
        retro = np.zeros(len(rays), dtype=bool)

        def face_hits(y, inside):
            """Hits on a -Y facing plane at body y, nearer than any earlier surface."""
            distance = (y - origin[1]) / direction[:, 1]
            points = origin + distance[:, None] * direction
            hit = (distance > 0) & (distance < best) & inside(points)
            best[hit] = distance[hit]
            return hit

        half = MARKER_SIDE_M / 2 + MARKER_QUIET_ZONE_M
        for x, y, z in MARKER_CENTERS.values():
            face_hits(y, lambda points: (np.abs(points[:, 0] - x) <= half)
                      & (np.abs(points[:, 2] - z) <= half))
        to_lidar = origin - REFLECTORS_BODY_M
        bright = (-to_lidar[:, 1] / np.linalg.norm(to_lidar, axis=1)
                  >= math.cos(math.radians(REFLECTOR_ACCEPTANCE_DEG)))
        for (x, y, z), shines in zip(REFLECTORS_BODY_M, bright):
            hit = face_hits(y - REFLECTOR_STANDOFF_M, lambda points: (
                (points[:, 0] - x)**2 + (points[:, 2] - z)**2 <= REFLECTOR_RADIUS_M**2))
            retro[hit] = shines
    returned = best <= lidar.max_range_m
    noisy = best + rng.normal(0.0, lidar.range_sigma_m, len(rays))
    ranges = np.where(returned, noisy, np.nan).astype(np.float32)
    shape = (lidar.height, lidar.width)
    return ranges.reshape(shape), (retro & returned).reshape(shape)


def _kabsch(model, measured, weights):
    """Weighted proper rotation and translation taking model points onto measured points."""
    share = weights / weights.sum()
    model_center, measured_center = share @ model, share @ measured
    spread = (model - model_center).T @ ((measured - measured_center) * share[:, None])
    u, _, vt = np.linalg.svd(spread)
    handed = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, handed]) @ u.T
    translation = measured_center - rotation @ model_center
    residual = measured - (model @ rotation.T + translation)
    return rotation, translation, float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))), model


def estimate_lidar_pose(scan, lidar) -> LidarPose:
    """Fit reflector-return centroids to the known layout; covariance from the fit geometry.

    Each centroid's variance combines range noise and pixel quantization over its hits.
    The pose covariance is the inverse of the linearized information for a left
    attitude perturbation, in lidar axes.
    """
    capture, delivery = scan.capture_time_ns, scan.delivery_time_ns
    points = lidar.directions() * scan.ranges_m[..., None]
    count, labels = cv2.connectedComponents(scan.retroreflective.astype(np.uint8), connectivity=8)
    pixel_rad = math.radians(lidar.hfov_deg) / lidar.width
    centroids, weights = [], []
    for label in range(1, count):
        hits = points[labels == label]
        centroid = hits.mean(axis=0)
        quantization = pixel_rad * np.linalg.norm(centroid)
        variance = ((lidar.range_sigma_m**2 + quantization**2 / 12) / len(hits)
                    + CENTROID_SIGMA_FLOOR_M**2)
        centroids.append(centroid)
        weights.append(1 / variance)
    found = len(centroids)
    if not 3 <= found <= len(REFLECTORS_BODY_M):
        return LidarPose(capture, delivery, False, reason="reflector_count", reflector_count=found)
    measured, weights = np.array(centroids), np.array(weights)
    rotation, translation, rms, model = min(
        (_kabsch(REFLECTORS_BODY_M[list(order)], measured, weights)
         for order in permutations(range(len(REFLECTORS_BODY_M)), found)),
        key=lambda fit: fit[2],
    )
    if rms > REFLECTOR_RMS_LIMIT_M:
        return LidarPose(capture, delivery, False, reason="reflector_mismatch",
                         reflector_count=found, rms_m=rms)
    information = np.zeros((6, 6))
    for point, weight in zip(model @ rotation.T, weights):
        jacobian = np.hstack((np.eye(3), -_skew(point)))
        information += weight * jacobian.T @ jacobian
    covariance = np.linalg.inv(information)
    return LidarPose(capture, delivery, True, rotation, translation, (covariance + covariance.T) / 2,
                     reflector_count=found, rms_m=rms)


class LidarNavigation(VisualNavigation):
    """The M4 filter, also accepting lidar poses with their own covariance."""

    def __init__(self, camera, calibration, lidar):
        self.lidar = lidar
        super().__init__(camera=camera, calibration=calibration)

    def _measurement(self, pose, own):
        if not isinstance(pose, LidarPose):
            return super()._measurement(pose, own)
        rotation = np.asarray(pose.rotation_lt, dtype=np.float64)
        translation = np.asarray(pose.translation_lt, dtype=np.float64)
        covariance = np.asarray(pose.covariance, dtype=np.float64)
        if (
            rotation.shape != (3, 3) or translation.shape != (3,) or covariance.shape != (6, 6)
            or not all(np.isfinite(array).all() for array in (rotation, translation, covariance))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
            or not np.allclose(covariance, covariance.T)
            or np.linalg.eigvalsh(covariance).min() <= 0
        ):
            raise ValueError("invalid lidar pose")
        position, rotation_nl = camera_pose(own, self.lidar)
        frame = np.kron(np.eye(2), rotation_nl)
        return _Measurement(
            int(pose.capture_time_ns), position + rotation_nl @ translation,
            matrix_quaternion(rotation_nl @ rotation), frame @ covariance @ frame.T,
        )
