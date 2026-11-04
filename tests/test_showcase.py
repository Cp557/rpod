"""Showcase chase-camera framing contracts (no Blender)."""

import unittest

import numpy as np

from rpod.approach import make_approach_scenario
from rpod.camera import project_points
from rpod.frames import relative_state
from rpod.showcase import VIEW, chase_camera, close_camera, plume_throttles


class ChaseCameraFraming(unittest.TestCase):
    def test_both_spacecraft_in_frame_from_the_sunlit_side(self):
        scenario = make_approach_scenario("approach")
        target = np.asarray(scenario["initial_positions_m"][0])
        basis = relative_state(np.asarray(scenario["initial_positions_m"]),
                               np.asarray(scenario["initial_velocities_mps"]))[2]
        for separation, sun_normal in ((30.0, -0.25), (6.5, -0.25), (30.0, 0.45)):
            with self.subTest(separation=separation, sun_normal=sun_normal):
                chaser = target - separation * basis[:, 1]
                toward_sun = basis @ [0.0, -np.sqrt(1 - sun_normal**2), sun_normal]
                position, rotation_nc = chase_camera(target, chaser, basis, -toward_sun)
                self.assertGreater((position - target) @ basis[:, 2] * sun_normal, 0)
                np.testing.assert_allclose(rotation_nc.T @ rotation_nc, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation_nc), 1, places=12)
                centers = np.stack((target, chaser)) - position
                pixels = project_points(centers @ rotation_nc, np.eye(3), np.zeros(3), VIEW)
                margin = 0.1 * np.array([VIEW.width, VIEW.height])
                self.assertTrue(np.all(pixels > margin), pixels)
                self.assertTrue(np.all(pixels < [VIEW.width, VIEW.height] - margin), pixels)
                earth_c = rotation_nc.T @ -position
                self.assertGreater(earth_c[1], 0)

    def test_close_shot_frames_chaser_before_the_target_port_at_any_range(self):
        scenario = make_approach_scenario("approach")
        target = np.asarray(scenario["initial_positions_m"][0])
        basis = relative_state(np.asarray(scenario["initial_positions_m"]),
                               np.asarray(scenario["initial_velocities_mps"]))[2]
        port = target - basis[:, 1]
        for gap in (30.0, 1.0):
            with self.subTest(gap=gap):
                chaser = port - (gap + 0.5) * basis[:, 1]
                position, rotation_nc = close_camera(chaser, port, basis, 1.0)
                np.testing.assert_allclose(rotation_nc.T @ rotation_nc, np.eye(3), atol=1e-12)
                points_c = (np.stack((chaser, port)) - position) @ rotation_nc
                self.assertTrue(np.all(points_c[:, 2] > 0))
                self.assertLess(np.linalg.norm(points_c[0]), np.linalg.norm(points_c[1]))
                pixels = project_points(points_c, np.eye(3), np.zeros(3), VIEW)
                margin = 0.1 * np.array([VIEW.width, VIEW.height])
                self.assertTrue(np.all(pixels > margin), pixels)
                self.assertTrue(np.all(pixels < [VIEW.width, VIEW.height] - margin), pixels)
                self.assertGreater((rotation_nc.T @ -position)[1], 0)

    def test_plumes_fire_against_the_applied_force(self):
        throttles = plume_throttles([1.0, -2.0, 0.0], 2.0)
        # +x force: exhaust along -x; -y force: exhaust along +y, saturated.
        self.assertEqual(throttles, [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
