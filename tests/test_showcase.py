"""Showcase chase-camera framing contracts (no Blender)."""

import unittest

import numpy as np

from rpod.approach import make_approach_scenario
from rpod.camera import project_points
from rpod.frames import relative_state
from rpod.showcase import (VIEW, chase_camera, close_camera, frame_ticks, nozzle_throttles,
                           opening_camera, plume_throttles)
from rpod.thrusters import layout


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

    def test_opening_move_joins_the_wide_and_close_shots_with_both_in_frame(self):
        scenario = make_approach_scenario("approach")
        target = np.asarray(scenario["initial_positions_m"][0])
        basis = relative_state(np.asarray(scenario["initial_positions_m"]),
                               np.asarray(scenario["initial_velocities_mps"]))[2]
        port = target - basis[:, 1]
        chaser = port - 25.0 * basis[:, 1]
        sunlight = -(basis @ [0.0, -np.sqrt(1 - 0.25**2), -0.25])
        # A tumbling target's body axes, turned 80 degrees from LVLH about along-track.
        cosine, sine = np.cos(np.radians(80.0)), np.sin(np.radians(80.0))
        frame = basis @ np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
        poses = [opening_camera(target, chaser, port, basis, frame, sunlight, 1.0, fraction)
                 for fraction in np.linspace(-0.1, 1.1, 145)]
        for got, want in ((poses[0], chase_camera(target, chaser, basis, sunlight)),
                          (poses[-1], close_camera(chaser, port, frame, 1.0))):
            np.testing.assert_allclose(got[0], want[0], atol=1e-6)
            np.testing.assert_allclose(got[1], want[1], atol=1e-9)
        margin = 0.05 * np.array([VIEW.width, VIEW.height])
        # No cut: each 1/120 of the move shifts the camera far less than the shots are apart.
        self.assertGreater(np.linalg.norm(poses[0][0] - poses[-1][0]), 30.0)
        for (position, rotation_nc), (following, turned) in zip(poses, poses[1:] + poses[-1:]):
            np.testing.assert_allclose(rotation_nc.T @ rotation_nc, np.eye(3), atol=1e-12)
            self.assertLess(np.linalg.norm(following - position), 3.0)
            self.assertLess(np.degrees(np.arccos(np.clip(
                (np.trace(rotation_nc.T @ turned) - 1) / 2, -1, 1))), 5.0)
            points_c = (np.stack((target, chaser)) - position) @ rotation_nc
            self.assertTrue(np.all(points_c[:, 2] > 0))
            pixels = project_points(points_c, np.eye(3), np.zeros(3), VIEW)
            self.assertTrue(np.all(pixels > margin), pixels)
            self.assertTrue(np.all(pixels < [VIEW.width, VIEW.height] - margin), pixels)

    def test_plumes_fire_against_the_applied_force(self):
        throttles = plume_throttles([1.0, -2.0, 0.0], 2.0)
        # +x force: exhaust along -x; -y force: exhaust along +y, saturated.
        self.assertEqual(throttles, [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        # Each nozzle takes its exhaust axis' throttle, and fires against the force.
        nozzles = np.array(nozzle_throttles(throttles))
        forces = layout()[1]
        self.assertTrue(np.all(forces[nozzles > 0] @ [1.0, -2.0, 0.0] > 0))
        self.assertEqual(sorted(set(nozzles)), [0.0, 0.5, 1.0])

    def test_docking_runs_are_filmed_slower_around_contact(self):
        ticks = frame_ticks(3600, 0.1, terminal_s=200.0, latch_tick=3400)
        steps = dict(zip(ticks, np.diff(ticks)))
        self.assertEqual((ticks[0], ticks[-1]), (0, 3599))
        self.assertEqual(steps[0], 6)
        self.assertTrue(all(steps[tick] == 3 for tick in ticks if 2010 <= tick < 3270))
        self.assertTrue(all(steps[tick] == 1 for tick in ticks if 3280 <= tick < 3599))
        self.assertEqual(frame_ticks(101, 0.1), list(range(0, 101, 3)) + [100])


if __name__ == "__main__":
    unittest.main()
