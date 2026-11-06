"""M13 pieces: the latched stack, the capture envelope, the docking tracker and controller."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from rpod.approach import make_engine
from rpod.docking import (CAPTURE, M4_TRANSLATION_SPECTRAL_DENSITY, MINIMUM_RANGE_M, TERMINAL,
                          UNINFORMATIVE_ATTITUDE_RAD, DockingController, DockingIcpTracker,
                          DockingMechanism, DockingNavigation, StackEngine, docking_lidar,
                          inside_envelope)
from rpod.frames import quaternion_matrices
from rpod.lidar import LidarModel
from rpod.m11_suite import m11_specs
from rpod.m13_suite import m13_specs
from rpod.natural_lidar import Track, exponential, scan_mesh
from rpod.navigation import _FilterState
from rpod.observations import ChaserInputs, NavigationEstimate, OwnNavigation
from rpod.tumble import TumbleController, tumble_initial_state
from rpod.visual_control import ControlDecision, MissionSpec

from test_m11 import asymmetric_mesh
from test_visual_env import near_handoff


def tumbling_pair():
    with patch("rpod.visual_run._handoff", return_value=near_handoff()):
        initial = tumble_initial_state("visual-m11-check", 2001, rate_deg_s=1.0,
                                       rate_axis=[1.0, 0.5, -0.3], lidar_noise_seed=[11, 0, 4])
    return initial, make_engine(initial).snapshot()


def momenta(snapshot, initial):
    """Linear momentum and angular momentum about the pair's center of mass."""
    masses, inertias = np.asarray(initial["masses_kg"]), np.asarray(initial["inertias_kg_m2"])
    positions, velocities = np.asarray(snapshot.positions_m), np.asarray(snapshot.velocities_mps)
    rotations = quaternion_matrices(np.asarray(snapshot.quaternions_xyzw))
    rates = np.asarray(snapshot.angular_velocities_body_radps)
    center, velocity = masses @ positions / masses.sum(), masses @ velocities / masses.sum()
    angular = sum(rotations[b] @ (inertias[b] * rates[b])
                  + masses[b] * np.cross(positions[b] - center, velocities[b] - velocity)
                  for b in range(2))
    return masses @ velocities, angular, center


class M13(unittest.TestCase):
    def test_the_latch_conserves_momentum_and_the_stack_stays_rigid(self):
        initial, snapshot = tumbling_pair()
        linear, angular, _ = momenta(snapshot, initial)
        stack = StackEngine(snapshot, initial)
        latched = stack.snapshot()
        np.testing.assert_allclose(np.asarray(latched.positions_m), snapshot.positions_m,
                                   atol=1e-9)
        after_linear, after_angular, _ = momenta(latched, initial)
        np.testing.assert_allclose(after_linear, linear, atol=1e-6)
        np.testing.assert_allclose(after_angular, angular, atol=1e-8)
        rotations = quaternion_matrices(np.asarray(latched.quaternions_xyzw))
        relative = rotations[0].T @ rotations[1]
        offset = rotations[0].T @ (np.asarray(latched.positions_m[1]) - latched.positions_m[0])
        for _ in range(300):
            coasting = stack.step(np.zeros((2, 3)), np.zeros((2, 3)))
        rotations = quaternion_matrices(np.asarray(coasting.quaternions_xyzw))
        np.testing.assert_allclose(rotations[0].T @ rotations[1], relative, atol=1e-12)
        np.testing.assert_allclose(rotations[0].T @ (np.asarray(coasting.positions_m[1])
                                                     - coasting.positions_m[0]), offset, atol=1e-8)
        np.testing.assert_allclose(momenta(coasting, initial)[1], angular, atol=1e-7)
        self.assertEqual(coasting.tick, snapshot.tick + 300)

    def test_the_chasers_wrench_moves_the_stack_about_its_center(self):
        initial, snapshot = tumbling_pair()
        pushed, coasting = StackEngine(snapshot, initial), StackEngine(snapshot, initial)
        before = pushed.snapshot()
        _, _, center = momenta(before, initial)
        chaser = quaternion_matrices(np.asarray(before.quaternions_xyzw[1]))
        force, torque = np.array([0.0, 1.5, 0.0]), np.array([0.1, 0.0, -0.2])
        pushed.step(np.array([np.zeros(3), force]), np.array([np.zeros(3), torque]))
        coasting.step(np.zeros((2, 3)), np.zeros((2, 3)))
        # Against a coasting copy, gravity cancels: one 0.1 s tick adds the force's impulse to
        # linear momentum, and the torque and the force's moment to angular momentum.
        linear, angular, _ = (np.subtract(a, b) for a, b in zip(
            momenta(pushed.snapshot(), initial), momenta(coasting.snapshot(), initial)))
        # The wrench is held in body axes while the stack turns about 0.1 degrees in the tick.
        np.testing.assert_allclose(linear, chaser @ force * 0.1, atol=5e-4)
        arm = np.asarray(before.positions_m[1]) - center
        np.testing.assert_allclose(angular, (chaser @ torque + np.cross(arm, chaser @ force))
                                   * 0.1, atol=5e-4)

    def test_the_capture_envelope_bounds_every_contact_condition(self):
        low, high = CAPTURE["closing_mps"]
        nominal = {"lateral_m": 0.01, "alignment_deg": 0.5, "closing_mps": (low + high) / 2,
                   "lateral_speed_mps": 0.002, "relative_rate_radps": 0.0005}
        self.assertTrue(inside_envelope(nominal))
        for key, value in (("lateral_m", 0.051), ("alignment_deg", 3.1),
                           ("closing_mps", 0.005), ("closing_mps", 0.07),
                           ("lateral_speed_mps", 0.03), ("relative_rate_radps", 0.004)):
            self.assertFalse(inside_envelope({**nominal, key: value}), key)

    def test_the_docking_tracker_reports_nothing_inside_its_minimum_range(self):
        lidar = LidarModel()
        tracker = DockingIcpTracker(lidar, asymmetric_mesh(),
                                    {"range_bins_m": [1e9], "variance": [[1e-4] * 6]})
        track = Track(np.eye(3), np.array([0, 0, MINIMUM_RANGE_M - 0.01]), np.zeros(3),
                      np.zeros(3), 0)
        tracker.previous = track
        scan = type("Scan", (), {"capture_id": 1, "capture_time_ns": 10**9,
                                 "delivery_time_ns": 10**9,
                                 "ranges_m": np.ones((lidar.height, lidar.width))})
        self.assertEqual(tracker.estimate(scan).reason, "below_minimum_range")
        self.assertIs(tracker.previous, track)

    def test_close_in_the_tracker_holds_attitude_and_reports_it_as_uninformative(self):
        lidar, mesh = docking_lidar(), asymmetric_mesh()
        rotation, translation = np.eye(3), np.array([0.05, -0.03, 3.0])
        ranges, _ = scan_mesh(rotation, translation, lidar, mesh, np.random.default_rng(5))
        points = (lidar.directions() * ranges[..., None]).reshape(-1, 3)
        held = exponential(np.radians([0.3, -0.2, 0.5])) @ rotation
        tracker = DockingIcpTracker(lidar, mesh, {"range_bins_m": [1e9], "variance": [[1e-4] * 6]})
        tracker.previous = Track(held, translation + [0.02, -0.01, 0.03],
                                 points[np.isfinite(points).all(axis=1)].mean(axis=0),
                                 np.zeros(3), 0)
        scan = type("Scan", (), {"capture_id": 1, "capture_time_ns": 10**9,
                                 "delivery_time_ns": 10**9, "ranges_m": ranges})
        pose = tracker.estimate(scan)
        self.assertTrue(pose.valid, pose.reason)
        np.testing.assert_array_equal(pose.rotation_lt, held)
        np.testing.assert_allclose(pose.covariance[3:, 3:],
                                   np.eye(3) * UNINFORMATIVE_ATTITUDE_RAD ** 2)
        self.assertLess(np.linalg.norm(pose.translation_lt - translation), 0.03)

    def test_the_docking_filter_drops_translation_noise_near_the_target(self):
        navigation = DockingNavigation.__new__(DockingNavigation)
        navigation._high_noise_intervals = []
        navigation.inertia = np.array([100.0, 120.0, 140.0])
        state = _FilterState(0, np.array([6.9e6, 0, 0]), np.array([0, 7.6e3, 0]),
                             np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros((12, 12)))
        covariances = {}
        for distance in (np.inf, 5.0):
            navigation.distance_m = distance
            covariances[distance] = navigation._path(replace(state, covariance=np.zeros((12, 12))),
                                                     10**9)[-1].covariance
        # Over 1 s, a white-acceleration density q adds q to the velocity variance.
        removed = covariances[np.inf][3:6, 3:6] - covariances[5.0][3:6, 3:6]
        np.testing.assert_allclose(np.diag(removed), M4_TRANSLATION_SPECTRAL_DENSITY
                                   - navigation.translation_spectral_density, rtol=1e-6)

    def test_the_controller_starts_the_terminal_segment_then_detumbles_once_latched(self):
        mission, mechanism = MissionSpec(), DockingMechanism()
        controller = DockingController(mission, mechanism)
        at_hold = ControlDecision(np.zeros(3), np.zeros(3), 3, 5.0, 5.0, False,
                                  ({"reason": "onboard_completion"},), "success", None, False,
                                  {"gap": 5.02}, 0)
        own = OwnNavigation(0, np.array([6.9e6, 0, 0]), np.array([0, 7.6e3, 0]),
                            np.array([0, 0, 0, 1.0]), np.array([0.01, -0.004, 0.002]))
        target = NavigationEstimate(0, own.position_m - [0, 6.5, 0], own.velocity_mps,
                                    np.array([0, 0, 0, 1.0]), np.zeros(3), np.eye(12), 0, "fresh")
        with patch.object(TumbleController, "update", return_value=at_hold):
            decision = controller.update(ChaserInputs(0, own, target))
        self.assertEqual((decision.outcome, decision.phase, decision.events[0]["reason"]),
                         (None, 1, "terminal_approach"))
        self.assertEqual((controller.goal, controller.profile_start), (TERMINAL["final_gap_m"], 5.02))
        self.assertEqual(controller.mission.position_kp_s2, TERMINAL["position_kp_s2"])
        mechanism.latched = True
        tumbling = controller.update(ChaserInputs(10**8, replace(own, time_ns=10**8), target))
        self.assertEqual((tumbling.phase, tumbling.events[0]["reason"]), (0, "captured"))
        self.assertAlmostEqual(np.linalg.norm(tumbling.torque_body_nm), 0.5)
        self.assertLess(tumbling.torque_body_nm @ own.angular_velocity_body_radps, 0)
        still = replace(own, angular_velocity_body_radps=np.zeros(3))
        outcomes = [controller.update(ChaserInputs(tick * 10**8, replace(still, time_ns=tick
                                                                         * 10**8), target))
                    .outcome for tick in range(2, 104)]
        self.assertEqual(outcomes.index("success"), 100)

    def test_m13_flies_every_m11_operational_episode(self):
        specs = m13_specs()
        operational = [spec for spec in m11_specs() if spec["category"] == "operational"]
        self.assertEqual(len(specs), 28)
        self.assertEqual([(spec["init_seed"], spec["tumble"]) for spec in specs],
                         [(spec["init_seed"], spec["tumble"]) for spec in operational])


if __name__ == "__main__":
    unittest.main()
