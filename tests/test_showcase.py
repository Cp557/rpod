"""Showcase chase-camera framing contracts (no Blender)."""

import unittest

import numpy as np

from rpod.approach import make_approach_scenario
from rpod.camera import project_points
from rpod.frames import relative_state
from rpod.showcase import (VIEW, _overlay, _sun_side, chase_camera, frame_ticks,
                           nozzle_throttles, plume_throttles)
from rpod.thrusters import layout


class ChaseCameraFraming(unittest.TestCase):
    def test_both_spacecraft_in_frame_clear_of_the_overlay_from_behind_the_chaser(self):
        scenario = make_approach_scenario("approach")
        target = np.asarray(scenario["initial_positions_m"][0])
        basis = relative_state(np.asarray(scenario["initial_positions_m"]),
                               np.asarray(scenario["initial_velocities_mps"]))[2]
        # Pixels the telemetry panel and the sensor inset cover.
        covered = _overlay(np.zeros((VIEW.height, VIEW.width, 3), np.uint8),
                           np.full((8, 8, 3), 255, np.uint8), "run", [("row", "value")] * 6,
                           "camera").any(axis=-1)
        # Straight behind at 30 m, the hold and docked; then swung 35 deg off the approach
        # axis, as a tumbling target's port carries the chaser around.
        for separation, sun_normal, swing_deg in ((30.0, -0.25, 0.0), (6.5, -0.25, 0.0),
                                                  (1.5, -0.25, 0.0), (30.0, 0.45, 0.0),
                                                  (25.0, 0.25, 35.0)):
            with self.subTest(separation=separation, sun_normal=sun_normal, swing=swing_deg):
                swing = np.radians(swing_deg)
                chaser = target + separation * basis @ [
                    0.0, -np.cos(swing), np.sign(sun_normal) * np.sin(swing)]
                toward_sun = basis @ [0.0, -np.sqrt(1 - sun_normal**2), sun_normal]
                position, rotation_nc = chase_camera(target, chaser, basis,
                                                     _sun_side(basis, -toward_sun))
                self.assertGreater((position - target) @ basis[:, 2] * sun_normal, 0)
                # Behind the chaser, within 45 deg of the approach axis, so it looks past the
                # chaser at the target's docking face (the old side view sat 64-75 deg off it).
                view, approach = position - target, chaser - target
                self.assertGreater(view @ approach / np.linalg.norm(view) / np.linalg.norm(approach),
                                   np.cos(np.radians(45)))
                np.testing.assert_allclose(rotation_nc.T @ rotation_nc, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation_nc), 1, places=12)
                centers = np.stack((target, chaser)) - position
                pixels = project_points(centers @ rotation_nc, np.eye(3), np.zeros(3), VIEW)
                margin = 0.1 * np.array([VIEW.width, VIEW.height])
                self.assertTrue(np.all(pixels > margin), pixels)
                self.assertTrue(np.all(pixels < [VIEW.width, VIEW.height] - margin), pixels)
                column, row = np.round(pixels).astype(int).T
                self.assertFalse(covered[row, column].any(), pixels)
                self.assertGreater((rotation_nc.T @ -position)[1], 0)

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
        self.assertTrue(all(steps[tick] == 3 for tick in ticks if 2000 <= tick < 3080))
        self.assertTrue(all(steps[tick] == 1 for tick in ticks if 3280 <= tick < 3599))
        # Each slow-down eases in, one tick per frame at a time.
        self.assertTrue(np.all(np.isin(np.diff(np.diff(ticks)), (-1, 0))))
        self.assertEqual(frame_ticks(101, 0.1), list(range(0, 101, 3)) + [100])


if __name__ == "__main__":
    unittest.main()
