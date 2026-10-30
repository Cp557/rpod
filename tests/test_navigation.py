"""Visual-filter math and timing checks using synthetic pose observations, not images."""

import math
import unittest

import numpy as np

from rpod.camera import CameraModel
from rpod.frames import matrix_quaternion, quaternion_matrices
from rpod.navigation import VisualNavigation, _FilterState, _propagate
from rpod.observations import OwnNavigation, PoseEstimate
from rpod.scenarios import MU_M3_S2, ORBIT_RADIUS_M


CAMERA = CameraModel(
    mount_position_body_m=(0.0, 0.0, 0.0),
    rotation_bc=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)


def fixture(capture_ns, delivery_ns):
    time_s = capture_ns / 1e9
    rate = math.sqrt(MU_M3_S2 / ORBIT_RADIUS_M**3)
    angle = rate * time_s
    position = ORBIT_RADIUS_M * np.array([math.cos(angle), math.sin(angle), 0.0])
    velocity = ORBIT_RADIUS_M * rate * np.array([-math.sin(angle), math.cos(angle), 0.0])
    own = OwnNavigation(
        capture_ns, position - [0.0, 0.0, 20.0], velocity,
        np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3),
    )
    pose = PoseEstimate(
        capture_ns, delivery_ns, True,
        rotation_ct=np.eye(3), translation_ct=np.array([0.0, 0.0, 20.0]),
    )
    return pose, own


class NavigationTests(unittest.TestCase):
    def test_inertial_angular_rate_multiplies_on_the_left(self):
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        rate = math.sqrt(MU_M3_S2 / ORBIT_RADIUS_M**3)
        state = _FilterState(
            0,
            np.array([ORBIT_RADIUS_M, 0.0, 0.0]),
            np.array([0.0, ORBIT_RADIUS_M * rate, 0.0]),
            matrix_quaternion(rotation_x),
            np.array([0.0, 0.0, 0.2]),
            np.eye(12) * 1e-4,
        )
        for tick in range(1, 11):
            state = _propagate(state, tick * 100_000_000, 4e-8)
        sin, cos = math.sin(0.2), math.cos(0.2)
        rotation_z = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(quaternion_matrices(state.quaternion), rotation_z @ rotation_x,
                                   atol=1e-13, rtol=0)
        expected_position = ORBIT_RADIUS_M * np.array([math.cos(rate), math.sin(rate), 0.0])
        np.testing.assert_allclose(state.position, expected_position, atol=1e-8, rtol=0)
        self.assertTrue(np.isfinite(state.covariance).all())
        self.assertGreater(np.linalg.eigvalsh(state.covariance).min(), 0.0)

    def test_delayed_and_out_of_order_updates_match_on_time_posterior(self):
        on_time = VisualNavigation(CAMERA)
        for second in range(3):
            pose, own = fixture(second * 1_000_000_000, second * 1_000_000_000)
            self.assertTrue(on_time.observe(pose, own))
        expected = on_time.predict(2_500_000_000)
        for order in ((1, 2), (2, 1)):
            delayed = VisualNavigation(CAMERA)
            self.assertTrue(delayed.observe(*fixture(0, 200_000_000)))
            deliveries = {
                12 if order[0] == 1 else 23: 1,
                22: 2,
            }
            for tick in range(3, 26):
                time_ns = tick * 100_000_000
                delayed.predict(time_ns)
                if tick in deliveries:
                    capture_ns = deliveries[tick] * 1_000_000_000
                    self.assertTrue(delayed.observe(*fixture(capture_ns, time_ns)))
            actual = delayed.predict(2_500_000_000)
            np.testing.assert_allclose(actual.position_m, expected.position_m, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                actual.velocity_mps, expected.velocity_mps, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual.quaternion_xyzw, expected.quaternion_xyzw,
                                       atol=1e-13, rtol=0)

    def test_availability_invalid_observations_health_and_owned_results(self):
        navigation = VisualNavigation(CAMERA)
        pose, own = fixture(0, 200_000_000)
        self.assertFalse(navigation.observe(PoseEstimate(0, 200_000_000, False), own))
        self.assertIsNone(navigation.predict(0))
        future, future_own = fixture(1_000_000_000, 500_000_000)
        self.assertFalse(navigation.observe(future, future_own))
        self.assertTrue(navigation.observe(pose, own))
        self.assertIsNone(navigation.predict(100_000_000))
        initial = navigation.predict(200_000_000)
        self.assertEqual(initial.health, "fresh")
        initial.covariance[:] = -1.0
        self.assertTrue((np.diag(navigation.predict(200_000_000).covariance) > 0).all())
        self.assertEqual(navigation.predict(1_200_000_000).health, "fresh")
        self.assertEqual(navigation.predict(1_300_000_000).health, "predicted")
        self.assertEqual(navigation.predict(3_000_000_000).health, "predicted")
        self.assertEqual(navigation.predict(3_100_000_000).health, "lost")
        navigation.reset()
        self.assertIsNone(navigation.predict(3_100_000_000))

    def test_camera_axis_calibration_is_rotated_into_inertial_covariance(self):
        camera = CameraModel(
            mount_position_body_m=(0.0, 0.0, 0.0),
            rotation_bc=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        )
        calibration = {
            "range_bins": [10.0, 100.0],
            "translation_variance": [[1e-4, 4e-4, 1e-3], [1e-2, 4e-2, 0.1]],
            "rotation_variance": [[0.0] * 3, [1e-4, 4e-4, 9e-4]],
        }
        navigation = VisualNavigation(camera, calibration)
        self.assertTrue(navigation.observe(*fixture(0, 0)))
        estimate = navigation.predict(0)
        np.testing.assert_allclose(np.diag(estimate.covariance)[:3], [0.04, 0.01, 0.1],
                                   atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.diag(estimate.covariance)[6:9], [4e-4, 1e-4, 9e-4],
                                   atol=1e-15, rtol=0)

    def test_gated_outlier_does_not_change_state_or_freshness(self):
        navigation = VisualNavigation(CAMERA)
        self.assertTrue(navigation.observe(*fixture(0, 200_000_000)))
        before = navigation.predict(1_200_000_000)
        pose, own = fixture(1_000_000_000, 1_200_000_000)
        outlier = PoseEstimate(
            pose.capture_time_ns, pose.delivery_time_ns, True,
            rotation_ct=pose.rotation_ct, translation_ct=np.array([100.0, 0.0, 20.0]),
        )
        self.assertFalse(navigation.observe(outlier, own))
        after = navigation.predict(1_200_000_000)
        np.testing.assert_array_equal(after.position_m, before.position_m)
        np.testing.assert_array_equal(after.covariance, before.covariance)
        self.assertEqual(after.capture_time_ns, before.capture_time_ns)


if __name__ == "__main__":
    unittest.main()
