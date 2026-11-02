"""Calibrated chaser camera and the known, explicitly marked target asset."""

from dataclasses import dataclass
import math

import numpy as np

from .frames import quaternion_matrices
from .observations import OwnNavigation


MARKER_SIDE_M = 0.55
MARKER_QUIET_ZONE_M = 0.08
MARKER_DICTIONARY = "DICT_4X4_50"
MARKER_CENTERS = {
    0: (-0.55, -1.0, 0.55),
    1: (0.55, -1.3, 0.55),
    2: (-0.55, -1.3, -0.55),
    3: (0.55, -1.0, -0.55),
}
EARTH_ROTATION_DEG = -37.8
EARTH_RATE_RADPS = 7.2921159e-5


@dataclass(frozen=True)
class CameraModel:
    width: int = 1024
    height: int = 1024
    hfov_deg: float = 35.0
    mount_position_body_m: tuple = (0.2, 0.55, 0.2)
    rotation_bc: tuple = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    camera_id: str = "chaser"

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) <= 0:
            raise ValueError("camera dimensions must be positive integers")
        if not math.isfinite(self.hfov_deg) or not 0 < self.hfov_deg < 179:
            raise ValueError("camera horizontal field of view must lie between 0 and 179 degrees")
        mount = np.asarray(self.mount_position_body_m, dtype=np.float64)
        rotation = np.asarray(self.rotation_bc, dtype=np.float64)
        if mount.shape != (3,) or not np.isfinite(mount).all():
            raise ValueError("camera mount must be a finite three-vector")
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-12, rtol=0)):
            raise ValueError("camera rotation_bc must be a proper rotation")

    @property
    def K(self) -> np.ndarray:
        """Square-pixel pinhole intrinsics; top-left pixel center is (0, 0)."""
        focal = self.width / (2 * math.tan(math.radians(self.hfov_deg) / 2))
        return np.array([
            [focal, 0, (self.width - 1) / 2],
            [0, focal, (self.height - 1) / 2],
            [0, 0, 1],
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height, "hfov_deg": self.hfov_deg,
            "mount_position_body_m": list(self.mount_position_body_m),
            "rotation_bc": np.asarray(self.rotation_bc).tolist(),
            "camera_id": self.camera_id, "K": self.K.tolist(),
            "distortion_coefficients": [0.0] * 5,
            "axes": "+X right, +Y down, +Z forward",
            "pixel_convention": "top-left pixel center (0, 0)",
        }


def marker_corners(marker_id: int) -> np.ndarray:
    """Target-body corners in detected top-left, top-right, bottom-right, bottom-left order."""
    if marker_id not in MARKER_CENTERS:
        raise ValueError(f"unknown target marker id: {marker_id}")
    half = MARKER_SIDE_M / 2
    return np.asarray(MARKER_CENTERS[marker_id], dtype=np.float64) + np.array([
        [-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half],
    ])


def camera_pose(own: OwnNavigation, camera: CameraModel | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return camera origin and optical-camera-to-inertial rotation from chaser navigation."""
    camera = camera or CameraModel()
    rotation_nb = quaternion_matrices(own.quaternion_xyzw)
    position = np.asarray(own.position_m, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all() or rotation_nb.shape != (3, 3):
        raise ValueError("own navigation must describe one finite spacecraft pose")
    return (
        position + rotation_nb @ np.asarray(camera.mount_position_body_m),
        rotation_nb @ np.asarray(camera.rotation_bc),
    )


def relative_target_pose(
    own: OwnNavigation,
    target_position_n: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    camera: CameraModel | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rendering/evaluation truth: target-body-to-camera rotation and target COM position."""
    position_nc, rotation_nc = camera_pose(own, camera)
    position_nt = np.asarray(target_position_n, dtype=np.float64)
    rotation_nt = quaternion_matrices(target_quaternion_xyzw)
    if position_nt.shape != (3,) or not np.isfinite(position_nt).all() or rotation_nt.shape != (3, 3):
        raise ValueError("target must describe one finite spacecraft pose")
    return rotation_nc.T @ rotation_nt, rotation_nc.T @ (position_nt - position_nc)


def earth_rotation_ne(time_s: float) -> np.ndarray:
    """Earth-body-to-inertial rotation: a turn about inertial +Z at Earth's sidereal rate.

    The simulation has no epoch, so the angle at t = 0 is chosen: the M2 handoff's longitude
    then lies over 70 deg E, and the approaches pass over southern India.
    """
    angle = np.radians(EARTH_ROTATION_DEG) + EARTH_RATE_RADPS * time_s
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def earth_pose(own: OwnNavigation, camera: CameraModel | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rendering truth: Earth-body-to-camera rotation and Earth's center in camera axes.

    The inertial frame is Earth-centered; Earth turns in it at the capture time.
    """
    position_nc, rotation_nc = camera_pose(own, camera)
    return rotation_nc.T @ earth_rotation_ne(own.time_ns / 1e9), rotation_nc.T @ -position_nc


def project_points(
    points_body: np.ndarray,
    rotation_ct: np.ndarray,
    translation_ct: np.ndarray,
    camera: CameraModel | None = None,
) -> np.ndarray:
    """Project finite target-body points to OpenCV pixel centers, without distortion."""
    camera = camera or CameraModel()
    points = np.asarray(points_body, dtype=np.float64)
    rotation = np.asarray(rotation_ct, dtype=np.float64)
    translation = np.asarray(translation_ct, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("projection requires points (N, 3), rotation (3, 3), translation (3,)")
    camera_points = points @ rotation.T + translation
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
        raise ValueError("projected points must be finite and in front of the camera")
    homogeneous = camera_points @ camera.K.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]
