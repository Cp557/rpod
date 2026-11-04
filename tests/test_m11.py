"""M11 pieces: port-frame metrics, tumbling guidance, mesh lidar ICP, keypoint helpers."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from rpod import fleet
from rpod._core import TargetMesh
from rpod.approach import make_engine, state_metrics
from rpod.frames import matrix_quaternion, quaternion_matrices
from rpod.keypoints import _peaks, _sample, calibrate_keypoints, keypoint_labels
from rpod.lidar import LidarModel
from rpod.m11_suite import m11_specs
from rpod.natural_lidar import IcpTracker, Track, exponential, rotation_vector, scan_mesh
from rpod.navigation import _FilterState
from rpod.camera import CameraModel
from rpod.tumble import (tumble_initial_state, tumble_metrics, tumble_truth_baseline,
                         torque_free_propagate)
from rpod.tumble_fleet import TumbleFleetController, TumbleScoring
from rpod.visual_control import MissionSpec

from test_fleet import random_states, snapshot
from test_visual_env import near_handoff


def box(center, half):
    """Triangles of an axis-aligned box."""
    corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float)
    faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
             [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return np.asarray(center) + corners * half, np.array(faces)


def asymmetric_mesh():
    """A bus with a wing and a boom, so no rotation maps it onto itself."""
    parts = [box((0, 0, 0), (1.0, 1.0, 1.0)), box((3.0, 0.2, 0), (2.0, 0.05, 0.8)),
             box((-0.4, 0.5, -1.8), (0.15, 0.15, 0.8))]
    vertices, triangles, offset = [], [], 0
    for part_vertices, part_triangles in parts:
        vertices.append(part_vertices)
        triangles.append(part_triangles + offset)
        offset += len(part_vertices)
    return TargetMesh(np.vstack(vertices), np.vstack(triangles).astype(np.uint32))


class M11(unittest.TestCase):
    def test_target_frame_metrics_equal_lvlh_metrics_when_the_body_holds_lvlh(self):
        state = random_states(1, 3, max_angle_deg=0.0)[0]
        ports = MissionSpec().docking_ports_body_m
        lvlh = state_metrics(snapshot(state), {"docking_ports_body_m": ports})
        state[0, 6:10] = matrix_quaternion(lvlh["basis"])
        rate = np.linalg.norm(np.cross(state[0, :3], state[0, 3:6])) / (state[0, :3] @ state[0, :3])
        state[0, 10:] = [0, 0, rate]
        expected = state_metrics(snapshot(state), {"docking_ports_body_m": ports})
        actual = tumble_metrics(state[:, :3], state[:, 3:6], state[:, 6:10], state[:, 10:], ports)
        for key in ("rho", "rho_dot", "port_r", "port_v"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-9)
        self.assertAlmostEqual(float(actual["gap"]), expected["gap"], places=9)

    def test_synchronized_start_and_truth_guidance_dock_with_a_tumbling_target(self):
        with patch("rpod.visual_run._handoff", return_value=near_handoff()):
            initial = tumble_initial_state("visual-m11-check", 2001, rate_deg_s=1.0,
                                           rate_axis=[1.0, 0.5, -0.3], lidar_noise_seed=[11, 0, 4])
        state = np.concatenate([np.asarray(initial[key]) for key in (
            "initial_positions_m", "initial_velocities_mps", "initial_quaternions_xyzw",
            "initial_angular_velocities_body_radps")], axis=-1)
        start = tumble_metrics(state[:, :3], state[:, 3:6], state[:, 6:10], state[:, 10:],
                               initial["docking_ports_body_m"])
        np.testing.assert_allclose(start["port_r"], [0, -30, 0], atol=1e-6)
        np.testing.assert_allclose(start["port_v"], 0, atol=1e-9)
        result = tumble_truth_baseline(initial, MissionSpec())
        self.assertEqual(result["outcome"], "success")

    def test_torque_free_prediction_follows_the_simulated_tumble(self):
        with patch("rpod.visual_run._handoff", return_value=near_handoff()):
            initial = tumble_initial_state("visual-m11-check", 2001, rate_deg_s=1.0,
                                           rate_axis=[1.0, 0.5, -0.3], lidar_noise_seed=[11, 0, 4])
        engine = make_engine(initial)
        quaternion = np.asarray(initial["initial_quaternions_xyzw"][0], dtype=float)
        rate = quaternion_matrices(quaternion) @ initial["initial_angular_velocities_body_radps"][0]
        state = _FilterState(0, np.asarray(initial["initial_positions_m"][0], dtype=float),
                             np.asarray(initial["initial_velocities_mps"][0], dtype=float),
                             quaternion, rate, np.zeros((12, 12)))
        # 300 s of free precession at 1 deg/s, over which a constant rate drifts 42 degrees.
        for tick in range(1, 3001):
            truth = engine.step(np.zeros((2, 3)), np.zeros((2, 3)))
            state = torque_free_propagate(state, tick * initial["dt_ns"], 0.0,
                                          np.asarray(initial["inertias_kg_m2"][0]))
        rotation = quaternion_matrices(np.asarray(truth.quaternions_xyzw[0]))
        error = rotation_vector(rotation.T @ quaternion_matrices(state.quaternion))
        self.assertLess(np.linalg.norm(error), 1e-6)
        np.testing.assert_allclose(
            state.angular_velocity, rotation @ truth.angular_velocities_body_radps[0], atol=1e-8)

    def test_tumbling_fleet_reproduces_the_scalar_oracle(self):
        with patch("rpod.visual_run._handoff", return_value=near_handoff()):
            initials = [tumble_initial_state("visual-m11-check", seed, rate_deg_s=rate,
                                             rate_axis=axis, lidar_noise_seed=[11, 0, 4])
                        for seed, rate, axis in ((2001, 0.4, [0, 0, 1.0]),
                                                 (2002, 1.0, [1.0, 0.5, -0.3]))]
        mission = MissionSpec()
        expected = [tumble_truth_baseline(initial, mission) for initial in initials]
        result = fleet.run_fleet(initials, mission,
                                 fleet.TruthNavigation(2, initials[0]["dt_ns"]),
                                 TumbleFleetController(mission, 2), scoring=TumbleScoring(mission))
        for index, oracle in enumerate(expected):
            self.assertEqual(fleet.OUTCOMES[result["outcome"][index]], oracle["outcome"])
            self.assertEqual(result["end_tick"][index] * initials[0]["dt_ns"],
                             oracle["elapsed_time_ns"])

    def test_icp_tracks_a_turning_drifting_target_from_a_nearby_prior(self):
        mesh, lidar = asymmetric_mesh(), LidarModel()
        calibration = {"range_bins_m": [1e9], "variance": [[1e-4] * 3 + [1e-5] * 3]}
        tracker = IcpTracker(lidar, mesh, calibration, seed=3)
        rng = np.random.default_rng(4)
        rotation = exponential(np.array([0.4, -0.9, 0.3]))
        ranges, _ = scan_mesh(rotation, np.array([0, -0.2, 15.0]), lidar, mesh, rng)
        points = (lidar.directions() * ranges[..., None]).reshape(-1, 3)
        tracker.previous = Track(exponential(np.array([0.02, 0.0, -0.03])) @ rotation,
                                 np.array([0.1, -0.1, 15.1]),
                                 points[np.isfinite(points).all(axis=1)].mean(axis=0),
                                 np.zeros(3), -10**9)
        for second in range(3):
            turned = exponential(np.array([0, 0, math.radians(1.0) * second])) @ rotation
            translation = np.array([0.1 * second, -0.2, 15.0])
            ranges, retro = scan_mesh(turned, translation, lidar, mesh, rng)
            self.assertFalse(retro.any())
            scan = type("Scan", (), {"capture_id": second, "capture_time_ns": second * 10**9,
                                     "delivery_time_ns": second * 10**9, "ranges_m": ranges})
            pose = tracker.estimate(scan)
            # Locked on, not accuracy: flat box faces let ICP slide; QuikSCAT's accuracy is
            # measured by the lidar calibration.
            self.assertTrue(pose.valid, pose.reason)
            self.assertLess(np.linalg.norm(pose.translation_lt - translation), 0.1)
            self.assertLess(math.degrees(np.linalg.norm(rotation_vector(
                pose.rotation_lt @ turned.T))), 2.0)

    def test_keypoint_labels_hide_points_behind_the_mesh_and_peaks_are_subpixel(self):
        mesh, camera = asymmetric_mesh(), CameraModel()
        rotation_ct = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # camera on body -Y
        front, back = [0.5, -1.0, 0.5], [0.5, 1.0, 0.5]
        _, visible = keypoint_labels(np.array([front, back]), rotation_ct, np.array([0, 0, 10.0]),
                                     camera, mesh)
        self.assertEqual(visible.tolist(), [True, False])
        grid = np.arange(32, dtype=float)
        heatmap = np.exp(-((grid[None, :] - 12.3) ** 2 + (grid[:, None] - 7.6) ** 2) / 4.5)
        peaks, confidence = _peaks(heatmap[None])
        np.testing.assert_allclose(peaks[0], [12.3, 7.6], atol=0.05)
        self.assertGreater(confidence[0], 0.9)

    def test_dataset_sampling_records_true_phase_and_survives_nadir_views(self):
        # These 1,500 draws include three views so close to nadir that the Earth eclipses
        # (nearly) the whole 70 degree phase window; they must be redrawn, not spin forever.
        rng, camera = np.random.default_rng([11, 0]), CameraModel()
        for _ in range(1500):
            _, _, translation_ct, light, phase = _sample(rng, camera, "train")
            toward_camera = -translation_ct / np.linalg.norm(translation_ct)
            self.assertAlmostEqual(phase, math.degrees(math.acos(-light @ toward_camera)), 9)
            self.assertLess(phase, 70.0)

    def test_keypoint_calibration_qualifies_only_without_gross_errors(self):
        rows = [{"range_m": 10.0 + index % 20, "valid": True,
                 "translation_error_m": [0.01, -0.01, 0.05], "rotation_error_rad": [0.002, 0, 0]}
                for index in range(60)] + [{"range_m": 12.0, "valid": False}]
        calibration = calibrate_keypoints(rows, [8.45, 15.95, 1e9], {}, {})
        self.assertTrue(calibration["qualification_passed"])
        self.assertAlmostEqual(calibration["coverage"], 60 / 61)
        np.testing.assert_allclose(calibration["translation_variance"][1], [1e-4, 1e-4, 2.5e-3])
        rows[0] = {**rows[0], "rotation_error_rad": [0.2, 0, 0]}  # 11.5 degrees
        calibration = calibrate_keypoints(rows, [8.45, 15.95, 1e9], {}, {})
        self.assertEqual((calibration["qualification_passed"], calibration["gross_errors"]),
                         (False, 1))

    def test_m11_specs_add_a_seeded_tumble_to_every_m5_episode(self):
        specs = m11_specs()
        self.assertEqual(len(specs), 40)
        self.assertEqual(specs, m11_specs())
        rates = [spec["tumble"]["rate_deg_s"] for spec in specs]
        self.assertTrue(all(0.2 <= rate <= 1.0 for rate in rates))
        self.assertTrue(all(abs(np.linalg.norm(spec["tumble"]["axis_body"]) - 1) < 1e-12
                            for spec in specs))


if __name__ == "__main__":
    unittest.main()
