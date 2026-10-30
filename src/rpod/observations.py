"""Explicit observation contracts for visual navigation and chaser autonomy."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OwnNavigation:
    time_ns: int
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_xyzw: np.ndarray
    angular_velocity_body_radps: np.ndarray


@dataclass(frozen=True)
class ImageObservation:
    capture_id: int
    capture_time_ns: int
    delivery_time_ns: int
    pixels: np.ndarray | None
    camera_id: str = "chaser"


@dataclass(frozen=True)
class PoseEstimate:
    capture_time_ns: int
    delivery_time_ns: int
    valid: bool
    rotation_ct: np.ndarray | None = None
    translation_ct: np.ndarray | None = None
    reason: str = ""
    marker_ids: tuple[int, ...] = ()
    inlier_count: int = 0
    reprojection_rms_px: float | None = None
    image_corners: np.ndarray | None = None


@dataclass(frozen=True)
class NavigationEstimate:
    time_ns: int
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_xyzw: np.ndarray
    angular_velocity_inertial_radps: np.ndarray
    covariance: np.ndarray
    capture_time_ns: int
    health: str


@dataclass(frozen=True)
class ChaserInputs:
    time_ns: int
    own: OwnNavigation
    target: NavigationEstimate | None
    visual_update: bool | None = None


@dataclass(frozen=True)
class Delivery:
    """A delayed camera frame, its image-derived pose, and own navigation at capture."""

    image: ImageObservation
    pose: PoseEstimate
    own_at_capture: OwnNavigation


@dataclass(frozen=True)
class LidarScan:
    """A flash-lidar range image: NaN without a return, plus retroreflector-bright pixels."""

    capture_id: int
    capture_time_ns: int
    delivery_time_ns: int
    ranges_m: np.ndarray
    retroreflective: np.ndarray


@dataclass(frozen=True)
class LidarPose:
    """Target-body-to-lidar pose from reflector returns; covariance orders [position, attitude]."""

    capture_time_ns: int
    delivery_time_ns: int
    valid: bool
    rotation_lt: np.ndarray | None = None
    translation_lt: np.ndarray | None = None
    covariance: np.ndarray | None = None
    reason: str = ""
    reflector_count: int = 0
    rms_m: float | None = None


@dataclass(frozen=True)
class LidarDelivery:
    """A delayed lidar scan, its reflector-derived pose, and own navigation at capture."""

    scan: LidarScan
    pose: LidarPose
    own_at_capture: OwnNavigation


@dataclass(frozen=True)
class EnvObservation:
    """Everything a visual-environment client receives at one physics tick."""

    time_ns: int
    own: OwnNavigation
    deliveries: tuple[Delivery, ...] = ()
    lidar: tuple[LidarDelivery, ...] = ()


CLAIMS = ("success", "retreated", "navigation_lost", "acquisition_failed")


@dataclass(frozen=True)
class ChaserCommand:
    """Chaser body wrench held for one tick; a claim ends the episode without stepping."""

    force_body_n: np.ndarray
    torque_body_nm: np.ndarray
    claim: str | None = None
