"""Flash-lidar sensor model, reflector-derived pose, and its navigation measurement."""

import unittest

import cv2
import numpy as np

from rpod.frames import matrix_quaternion
from rpod.lidar import LidarModel, LidarNavigation, estimate_lidar_pose, scan_target
from rpod.navigation import _rotation_residual
from rpod.observations import LidarPose, LidarScan, OwnNavigation


def nominal_pose(lidar, gap_m, rotation=np.eye(3)):
    """Target-to-lidar pose with the ports on one line and `rotation` of the target body."""
    rotation_bc = np.asarray(lidar.rotation_bc)
    target_body = np.array([0.0, 0.5 + gap_m + 1.0, 0.0]) - lidar.mount_position_body_m
    return rotation_bc.T @ rotation, rotation_bc.T @ target_body


def scan(lidar, rotation_lt, translation_lt, seed):
    ranges, retro = scan_target(rotation_lt, translation_lt, lidar, np.random.default_rng(seed))
    return LidarScan(0, 500_000_000, 700_000_000, ranges, retro)


def small_rotation(rng, sigma_deg=2.0):
    vector = rng.normal(size=3) * np.deg2rad(sigma_deg)
    return cv2.Rodrigues(vector)[0]


class LidarSensor(unittest.TestCase):
    def test_scan_sees_four_reflectors_only_within_their_acceptance_cone(self):
        lidar = LidarModel()
        result = scan(lidar, *nominal_pose(lidar, 10), seed=1)
        blobs = cv2.connectedComponents(result.retroreflective.astype(np.uint8), connectivity=8)[0]
        self.assertEqual(blobs - 1, 4)
        # The reflectors sit 2 mm proud of the docking face, 10 m - 0.05 m from the lidar.
        reflector_ranges = result.ranges_m[result.retroreflective]
        self.assertAlmostEqual(float(np.median(reflector_ranges)), 9.95, delta=0.03)
        yaw = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(60)]))[0]
        oblique = scan(lidar, *nominal_pose(lidar, 10, yaw), seed=1)
        self.assertGreater(np.isfinite(oblique.ranges_m).sum(), 0)
        self.assertFalse(oblique.retroreflective.any())
        distant = scan(lidar, *nominal_pose(lidar, 150), seed=1)
        self.assertFalse(np.isfinite(distant.ranges_m).any())

    def test_pose_matches_truth_within_its_covariance(self):
        lidar, rng = LidarModel(), np.random.default_rng(3)
        for gap in (5, 30):
            nees = []
            for seed in range(20):
                rotation_lt, translation_lt = nominal_pose(lidar, gap, small_rotation(rng))
                pose = estimate_lidar_pose(scan(lidar, rotation_lt, translation_lt, seed), lidar)
                self.assertTrue(pose.valid, pose.reason)
                error = np.r_[pose.translation_lt - translation_lt, _rotation_residual(
                    matrix_quaternion(pose.rotation_lt), matrix_quaternion(rotation_lt))]
                nees.append(error @ np.linalg.solve(pose.covariance, error))
            # Six degrees of freedom: a consistent or conservative covariance averages <= 6.
            self.assertLess(np.mean(nees), 6.0, gap)

    def test_too_few_reflectors_give_no_pose(self):
        lidar = LidarModel()
        full = scan(lidar, *nominal_pose(lidar, 10), seed=1)
        labels = cv2.connectedComponents(full.retroreflective.astype(np.uint8), connectivity=8)[1]
        partial = LidarScan(0, 0, 0, full.ranges_m, full.retroreflective & (labels <= 2))
        pose = estimate_lidar_pose(partial, lidar)
        self.assertEqual((pose.valid, pose.reason, pose.reflector_count),
                         (False, "reflector_count", 2))


class LidarMeasurement(unittest.TestCase):
    def test_filter_places_the_target_from_a_lidar_pose_in_inertial_axes(self):
        lidar = LidarModel()
        full = scan(lidar, *nominal_pose(lidar, 10), seed=1)
        # Delivered at capture, so the first estimate is the measurement itself.
        pose = estimate_lidar_pose(LidarScan(0, 500_000_000, 500_000_000, full.ranges_m,
                                             full.retroreflective), lidar)
        own = OwnNavigation(500_000_000, np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]),
                            np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
        navigator = LidarNavigation(None, None, lidar)
        self.assertTrue(navigator.observe(pose, own))
        estimate = navigator.predict(500_000_000)
        # Identity attitude: the target COM is 0.5 m + 10 m gap + 1 m along inertial +Y.
        np.testing.assert_allclose(estimate.position_m, own.position_m + [0.0, 11.5, 0.0],
                                   atol=0.03)
        self.assertEqual(estimate.health, "fresh")
        singular = LidarPose(1_500_000_000, 1_700_000_000, True, pose.rotation_lt,
                             pose.translation_lt, np.zeros((6, 6)))
        later = OwnNavigation(1_500_000_000, own.position_m, own.velocity_mps,
                              own.quaternion_xyzw, np.zeros(3))
        self.assertFalse(navigator.observe(singular, later))


if __name__ == "__main__":
    unittest.main()
