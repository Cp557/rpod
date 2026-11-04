"""Natural-feature flash lidar: mesh ray casts and point-to-plane ICP against the target mesh.

`scan_mesh` replaces `lidar.scan_target` for the unmarked target: the same pinhole rays, range
noise and output, cast against the rendered mesh, with no retroreflectors.

`IcpTracker` estimates the target pose from each scan as the lidar's own processing unit. It
predicts from its previous solution, moving the translation with the shift of the measured
point cloud's centroid and the attitude at the last measured rate, or searches attitude
hypotheses when it has no solution; it never sees truth or the client filter. Poses use the `LidarPose` convention: measured points are
`R_lt p_body + t_lt`, perturbed as `exp(dtheta) R_lt` and `t_lt + dt`. The covariance comes
from a per-range calibration against truth, not from the fit, which is optimistic.
"""

from collections import namedtuple
import json
import math
from pathlib import Path

import numpy as np

from .frames import matrix_quaternion
from .navigation import _skew
from .observations import LidarPose

POINTS = 2000
ACQUISITION_POINTS = 600
ITERATIONS = 60
ACQUISITION_ITERATIONS = 12
REFINED_HYPOTHESES = 6
INLIER_M = 0.1
HUBER_M = 0.05
MINIMUM_POINTS = 150
MINIMUM_INLIER_FRACTION = 0.7
RMS_LIMIT_M = 0.06
# Range bins with fewer calibration samples use the variance pooled over all bins.
MINIMUM_CALIBRATION_SAMPLES = 30
# A second distinct solution must fit this much worse, or acquisition is ambiguous.
AMBIGUITY_RATIO = 1.25
DISTINCT_DEG = 15.0


def scan_mesh(rotation_lt, translation_lt, lidar, mesh, rng):
    """Ray-cast the target mesh; float32 ranges (NaN without a return), no reflectors."""
    rays = lidar.directions().reshape(-1, 3)
    rotation = np.asarray(rotation_lt, dtype=np.float64)
    origin = -rotation.T @ np.asarray(translation_lt, dtype=np.float64)
    directions = np.ascontiguousarray(rays @ rotation)
    ranges = mesh.cast_rays(np.repeat(origin[None], len(rays), axis=0), directions,
                            lidar.max_range_m)
    noisy = ranges + rng.normal(0.0, lidar.range_sigma_m, len(rays))
    shape = (lidar.height, lidar.width)
    return (np.where(np.isfinite(ranges), noisy, np.nan).astype(np.float32).reshape(shape),
            np.zeros(shape, dtype=bool))


def rotation_vector(rotation):
    """Rotation vector of a proper rotation matrix."""
    angle = math.acos(min(1.0, max(-1.0, (np.trace(rotation) - 1) / 2)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return angle * axis / np.linalg.norm(axis)


def exponential(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis = _skew(vector / angle)
    return np.eye(3) + math.sin(angle) * axis + (1 - math.cos(angle)) * axis @ axis


def icp(points_l, mesh, rotation, translation, iterations):
    """Point-to-plane Gauss-Newton with Huber weights; returns the fit and its residuals."""
    for _ in range(iterations):
        body = (points_l - translation) @ rotation
        closest, normals, _ = mesh.closest_points(np.ascontiguousarray(body))
        model_l = closest @ rotation.T
        normals_l = normals @ rotation.T
        residual = np.sum(normals_l * (points_l - model_l - translation), axis=1)
        weight = np.minimum(1.0, HUBER_M / np.maximum(np.abs(residual), 1e-12))
        jacobian = np.hstack([normals_l, np.cross(model_l, normals_l)])
        hessian = jacobian.T @ (jacobian * weight[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(6), jacobian.T @ (weight * residual))
        rotation = exponential(step[3:]) @ rotation
        translation = translation + step[:3]
        if np.linalg.norm(step[:3]) < 1e-5 and np.linalg.norm(step[3:]) < 1e-6:
            break
    body = (points_l - translation) @ rotation
    _, _, distance = mesh.closest_points(np.ascontiguousarray(body))
    return rotation, translation, distance


def _fit_quality(distance):
    inliers = distance < INLIER_M
    rms = float(np.sqrt(np.mean(distance[inliers] ** 2))) if inliers.any() else math.inf
    return rms, float(inliers.mean())


def _cost(distance):
    """Truncated quadratic: rewards both a tight fit and coverage."""
    return float(np.mean(np.minimum(distance, INLIER_M) ** 2))


def attitude_hypotheses():
    """72 rotations about 30 degrees apart: 12 icosahedron directions for body +Y, each with
    6 roll angles about it."""
    golden = (1 + math.sqrt(5)) / 2
    vertices = [(0, s1, s2 * golden) for s1 in (-1, 1) for s2 in (-1, 1)]
    vertices = np.array([v[k:] + v[:k] for v in vertices for k in range(3)], dtype=np.float64)
    hypotheses = []
    for axis in vertices / np.linalg.norm(vertices, axis=1, keepdims=True):
        helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 0, 1.0])
        x = np.cross(helper, axis)
        x /= np.linalg.norm(x)
        base = np.stack([x, axis, np.cross(x, axis)], axis=1)  # body Y -> axis
        for roll in np.arange(6) * math.pi / 3:
            hypotheses.append(base @ exponential(np.array([0, roll, 0])))
    return hypotheses


# The tracker's last solution: pose, measured cloud centroid, attitude rate (lidar axes,
# rad/s) and capture time.
Track = namedtuple("Track", "rotation translation centroid rate time_ns")


class IcpTracker:
    """Lidar pose processing for the unmarked target, keeping its own last solution."""

    def __init__(self, lidar, mesh, calibration, seed=0):
        self.lidar, self.mesh, self.calibration = lidar, mesh, calibration
        self.seed = [int(value) for value in np.atleast_1d(seed)]
        self.previous = None

    def reset(self):
        self.previous = None

    def _covariance(self, distance_m):
        bins = self.calibration["range_bins_m"]
        index = min(int(np.searchsorted(bins, distance_m)), len(bins) - 1)
        return np.diag(self.calibration["variance"][index])

    def estimate(self, scan):
        capture, delivery = scan.capture_time_ns, scan.delivery_time_ns
        points = (self.lidar.directions() * scan.ranges_m[..., None]).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)].astype(np.float64)
        if len(points) < MINIMUM_POINTS:
            self.previous = None
            return LidarPose(capture, delivery, False, reason="too_few_returns")
        rng = np.random.default_rng([*self.seed, scan.capture_id])
        sample = points[rng.choice(len(points), min(POINTS, len(points)), replace=False)]
        centroid, previous = points.mean(axis=0), self.previous
        if previous is None:
            found = self._acquire(sample[:ACQUISITION_POINTS])
            if isinstance(found, LidarPose):
                return LidarPose(capture, delivery, False, reason=found.reason)
            rotation, translation = found
        else:
            elapsed = (capture - previous.time_ns) / 1e9
            rotation = exponential(previous.rate * elapsed) @ previous.rotation
            translation = previous.translation + centroid - previous.centroid
        rotation, translation, distance = icp(sample, self.mesh, rotation, translation,
                                              ITERATIONS)
        rms, fraction = _fit_quality(distance)
        if rms > RMS_LIMIT_M or fraction < MINIMUM_INLIER_FRACTION:
            self.previous = None
            return LidarPose(capture, delivery, False, reason="icp_mismatch", rms_m=rms)
        rate = (rotation_vector(rotation @ previous.rotation.T) / elapsed
                if previous is not None and elapsed > 0 else np.zeros(3))
        self.previous = Track(rotation, translation, centroid, rate, capture)
        covariance = self._covariance(float(np.linalg.norm(translation)))
        return LidarPose(capture, delivery, True, rotation, translation, covariance, rms_m=rms)

    def _acquire(self, sample):
        """Coarse ICP from every attitude hypothesis, full ICP from the best few, and the best
        refined fit unless a distinct one fits nearly as well."""
        center = sample.mean(axis=0)
        translation = center + 0.5 * center / np.linalg.norm(center)
        coarse = sample[:ACQUISITION_POINTS]
        fits = []
        for hypothesis in attitude_hypotheses():
            rotation, fitted, distance = icp(coarse, self.mesh, hypothesis, translation,
                                             ACQUISITION_ITERATIONS)
            fits.append((_cost(distance), rotation, fitted))
        fits.sort(key=lambda fit: fit[0])
        refined = []
        for _, rotation, fitted in fits[:REFINED_HYPOTHESES]:
            rotation, fitted, distance = icp(sample, self.mesh, rotation, fitted, ITERATIONS)
            refined.append((_cost(distance), rotation, fitted))
        refined.sort(key=lambda fit: fit[0])
        best = refined[0]
        for cost, rotation, _ in refined[1:]:
            if math.degrees(np.linalg.norm(rotation_vector(rotation @ best[1].T))) > DISTINCT_DEG:
                if cost < AMBIGUITY_RATIO * best[0]:
                    return LidarPose(0, 0, False, reason="ambiguous_acquisition")
                break
        return best[1], best[2]


def calibrate(lidar, mesh, ranges_m, samples, seed, range_bins_m):
    """Per-range-bin variances of tracked ICP poses against truth, in lidar axes.

    Each sample draws a random target attitude and a line-of-sight position at a range in
    `ranges_m`, scans it, and tracks from truth perturbed by up to 3 degrees and 0.15 m, as
    after a 1 s step. Errors use the `LidarPose` convention.
    """
    rng = np.random.default_rng(seed)
    errors = {index: [] for index in range(len(range_bins_m))}
    for _ in range(samples):
        axis = rng.normal(size=3)
        rotation = exponential(axis / np.linalg.norm(axis) * rng.uniform(0, math.pi))
        distance = rng.uniform(*ranges_m)
        direction = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.0])
        translation = distance * direction / np.linalg.norm(direction)
        ranges, _ = scan_mesh(rotation, translation, lidar, mesh, rng)
        scan = _Scan(int(rng.integers(1 << 30)), ranges)
        tracker = IcpTracker(lidar, mesh, {"range_bins_m": range_bins_m,
                                           "variance": [[1.0] * 6] * len(range_bins_m)})
        nudge = rng.normal(size=3)
        points = (lidar.directions() * ranges[..., None]).reshape(-1, 3)
        tracker.previous = Track(
            exponential(nudge / np.linalg.norm(nudge) * math.radians(rng.uniform(0, 3)))
            @ rotation, translation + rng.uniform(-0.15, 0.15, 3),
            points[np.isfinite(points).all(axis=1)].mean(axis=0), np.zeros(3), scan.capture_time_ns)
        pose = tracker.estimate(scan)
        index = min(int(np.searchsorted(range_bins_m, distance)), len(range_bins_m) - 1)
        if pose.valid:
            errors[index].append(np.r_[pose.translation_lt - translation,
                                       rotation_vector(pose.rotation_lt @ rotation.T)])
    pooled = np.array([row for rows in errors.values() for row in rows])
    variance, counts = [], []
    for index in range(len(range_bins_m)):
        rows = np.array(errors[index])
        counts.append(len(rows))
        sparse = len(rows) < MINIMUM_CALIBRATION_SAMPLES
        variance.append(np.mean((pooled if sparse else rows) ** 2, axis=0).tolist())
    return {"artifact_type": "natural_lidar_calibration", "range_bins_m": list(range_bins_m),
            "variance": variance, "samples": counts, "seed": seed,
            "parameterization": "lidar axes [t_lt error, left attitude error of R_lt]"}


class _Scan:
    def __init__(self, capture_id, ranges_m):
        self.capture_id, self.capture_time_ns, self.delivery_time_ns = capture_id, 0, 0
        self.ranges_m = ranges_m


def load_calibration(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pose_quaternion(rotation_lt):
    return matrix_quaternion(rotation_lt)
