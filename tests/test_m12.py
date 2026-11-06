"""M12 pieces: lidar fault injection, the coasting tracker, the avoidance burn, drift checks."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod.approach import make_engine
from rpod.camera import CameraModel
from rpod.lidar import LidarModel
from rpod.lidar_faults import (CAM_DRIFT_MPS, CAM_FORCE_N, CAM_SEPARATION_MPS,
                               CamTumbleController, CoastingIcpTracker, LidarFaultEnv, cam_burn,
                               drift_clearance, hull_clearance)
from rpod.m12_suite import FAULTS, TRIGGER_GAPS_M, _fault_response, m12_specs
from rpod.m11_suite import m11_specs
from rpod.natural_lidar import IcpTracker, Track
from rpod.observations import ChaserCommand, ChaserInputs, NavigationEstimate, OwnNavigation
from rpod.tumble import TumbleController, tumble_initial_state
from rpod.visual_control import ControlDecision, MissionSpec
from rpod.visual_run import sunlight_direction

from test_m11 import asymmetric_mesh
from test_visual_env import near_handoff

CALIBRATION = {"range_bins_m": [1e9], "variance": [[1e-4] * 3 + [1e-5] * 3]}


def tumbling_initial():
    with patch("rpod.visual_run._handoff", return_value=near_handoff()):
        initial = tumble_initial_state("visual-m11-check", 2001, rate_deg_s=0.5,
                                       rate_axis=[0, 0, 1.0], lidar_noise_seed=[11, 0, 4])
    initial["mission"] = {"final_gap_m": 5.0, "retreat_gap_m": 30.0, "dwell_ns": 10**10}
    initial["sunlight_direction_inertial"] = sunlight_direction(initial).tolist()
    return initial


def state_rows(engine_snapshot):
    return np.concatenate([np.asarray(getattr(engine_snapshot, field)) for field in (
        "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps")],
        axis=-1)


class M12(unittest.TestCase):
    def run_scans(self, fault, coast=False):
        """Three scans with `fault` triggered at the first; rows, delivered poses, ranges."""
        initial = {**tumbling_initial(), "lidar_fault": fault}
        env = LidarFaultEnv(CameraModel(), asymmetric_mesh(), CALIBRATION, coast_empty_scans=coast)
        with tempfile.TemporaryDirectory() as directory, env:
            env.reset(initial, directory)
            poses = []
            for _ in range(28):
                observation = env.step(ChaserCommand(np.zeros(3), np.zeros(3)))
                poses += [delivery.pose for delivery in observation.lidar]
            ranges = [np.load(Path(directory) / row["scan_path"])["ranges_m"]
                      for row in env.lidar_rows]
        return env.lidar_rows, poses, ranges

    def test_faults_hit_only_their_scans_and_leave_nominal_noise_alone(self):
        nominal = {"kind": "dropped", "scans": 0, "trigger_gap_m": 100.0, "bias_m": 0.0}
        rows, poses, clean = self.run_scans(nominal)
        self.assertEqual([row.get("fault") for row in rows], [None, None, None])
        rows, poses, blank = self.run_scans({**nominal, "kind": "blank", "scans": 1})
        self.assertEqual([row.get("fault") for row in rows], ["blank", None, None])
        self.assertEqual((rows[0]["returns"], poses[0].reason), (0, "too_few_returns"))
        np.testing.assert_array_equal(blank[1], clean[1])
        rows, poses, biased = self.run_scans({**nominal, "kind": "biased", "scans": 2,
                                              "bias_m": 0.25})
        self.assertEqual([row.get("fault") for row in rows], ["biased", "biased", None])
        np.testing.assert_allclose(biased[0], clean[0] + 0.25)
        np.testing.assert_array_equal(biased[2], clean[2])
        rows, poses, _ = self.run_scans({**nominal, "kind": "dropped", "scans": 1})
        self.assertEqual(rows[0]["fault"], "dropped")
        self.assertEqual([pose.reason for pose in poses][:1], ["dropped_scan"])

    def test_the_coasting_tracker_keeps_its_track_only_through_empty_scans(self):
        lidar, track = LidarModel(), Track(np.eye(3), np.array([0, 0, 15.0]), np.zeros(3),
                                           np.zeros(3), 0)
        empty = type("Scan", (), {"capture_id": 1, "capture_time_ns": 10**9,
                                  "delivery_time_ns": 10**9,
                                  "ranges_m": np.full((lidar.height, lidar.width), np.nan)})
        for tracker_class, kept in ((IcpTracker, False), (CoastingIcpTracker, True)):
            tracker = tracker_class(lidar, asymmetric_mesh(), CALIBRATION)
            tracker.previous = track
            self.assertEqual(tracker.estimate(empty).reason, "too_few_returns")
            self.assertEqual(tracker.previous is track, kept)

    def test_the_avoidance_burn_leaves_the_planned_relative_velocity(self):
        rng = np.random.default_rng(3)
        position = np.array([6.9e6, 1e5, 2e4])
        velocity = np.array([-100.0, 7.6e3, 50.0])
        target = NavigationEstimate(0, position + rng.normal(size=3) * 4, velocity
                                    + rng.normal(size=3) * 0.1, np.array([0, 0, 0, 1.0]),
                                    np.zeros(3), np.eye(12), 0, "lost")
        own = OwnNavigation(0, position, velocity, np.array([0, 0, 0, 1.0]), np.zeros(3))
        mission = MissionSpec()
        direction, end_ns = cam_burn(ChaserInputs(0, own, target), mission)
        seconds = end_ns / 1e9
        relative = velocity - target.velocity_mps + CAM_FORCE_N / mission.masses_kg[1] \
            * seconds * direction
        away = (position - target.position_m) / np.linalg.norm(position - target.position_m)
        along = np.cross(np.cross(position, velocity) / np.linalg.norm(np.cross(position,
                                                                                velocity)),
                         position / np.linalg.norm(position))
        np.testing.assert_allclose(relative @ away, CAM_SEPARATION_MPS
                                   + abs(away @ along) * CAM_DRIFT_MPS, atol=1e-6)
        np.testing.assert_allclose(abs(relative @ along), CAM_DRIFT_MPS
                                   + abs(away @ along) * CAM_SEPARATION_MPS, atol=1e-6)

    def test_the_controller_burns_then_declares_the_loss(self):
        mission = MissionSpec()
        lost = ControlDecision(np.zeros(3), np.zeros(3), 6, 5.0, 5.0, False,
                               ({"reason": "navigation_lost"},), "navigation_lost", None, False,
                               None, 0)
        position, velocity = np.array([6.9e6, 0, 0]), np.array([0, 7.6e3, 0])
        own = OwnNavigation(0, position, velocity, np.array([0, 0, 0, 1.0]), np.zeros(3))
        target = NavigationEstimate(0, position - [0, 7.0, 0], velocity, np.array([0, 0, 0, 1.0]),
                                    np.zeros(3), np.eye(12), 0, "lost")
        controller = CamTumbleController(mission)
        with patch.object(TumbleController, "update", return_value=lost):
            decisions = [controller.update(ChaserInputs(tick * 10**8, replace(own, time_ns=tick
                                                                              * 10**8), target))
                         for tick in range(400)]
        burning = [decision for decision in decisions if decision.outcome is None]
        self.assertEqual(decisions[0].events, lost.events)
        self.assertTrue(all(np.linalg.norm(decision.force_body_n) == CAM_FORCE_N
                            for decision in burning))
        self.assertEqual(decisions[len(burning)].outcome, "navigation_lost")
        _, end_ns = cam_burn(ChaserInputs(0, own, target), mission)
        self.assertEqual(len(burning), math.ceil(end_ns / 10**8))

    def test_drift_clearance_finds_a_collision_and_matches_exact_checks(self):
        initial, mesh = tumbling_initial(), asymmetric_mesh()
        start = state_rows(make_engine(initial).snapshot())
        toward = start.copy()
        line = toward[0, :3] - toward[1, :3]
        toward[1, 3:6] = toward[0, 3:6] + 0.5 * line / np.linalg.norm(line)
        result = drift_clearance(initial, np.stack([start, toward]), mesh, seconds=120.0)
        self.assertGreater(result["minimum_clearance_m"][0], 20.0)
        self.assertLess(result["minimum_clearance_m"][1], 0.05)
        # Exact hull checks every tick of the colliding coast agree with the bounded search.
        engine = make_engine({**initial, "initial_velocities_mps": toward[:, 3:6].tolist()})
        exact = []
        for _ in range(1201):
            snapshot = engine.snapshot()
            exact.append(hull_clearance(mesh, np.asarray(snapshot.positions_m)[None],
                                        np.asarray(snapshot.quaternions_xyzw)[None])[0])
            engine.step(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertAlmostEqual(result["minimum_clearance_m"][1], min(exact), places=9)

    def test_m12_specs_put_every_fault_at_every_gap_on_m11_operational_episodes(self):
        specs, operational = m12_specs(), [spec for spec in m11_specs()
                                           if spec["category"] == "operational"]
        self.assertEqual(len({spec["id"] for spec in specs}), len(FAULTS) * len(TRIGGER_GAPS_M))
        cells = {(spec["lidar_fault"]["kind"], spec["lidar_fault"]["scans"],
                  spec["lidar_fault"]["trigger_gap_m"]) for spec in specs}
        self.assertEqual(cells, {(kind, scans, gap) for (kind, scans) in FAULTS
                                 for gap in TRIGGER_GAPS_M})
        for spec, base in zip(specs, operational):
            self.assertEqual((spec["init_seed"], spec["tumble"], spec["conditions"]),
                             (base["init_seed"], base["tumble"], base["conditions"]))

    def test_fault_response_times_the_brake_reacquisition_and_loss(self):
        lidar = [{"capture_id": 9, "capture_time_ns": 9_500_000_000,
                  "actual_delivery_time_ns": 9_700_000_000, "fault": "dropped",
                  "navigation_accepted": False}]
        events = [{"kind": "supervisor_transition", "reason": "visual_brake", "tick": 97,
                   "time_ns": 9_700_000_000},
                  {"kind": "supervisor_transition", "reason": "navigation_lost", "tick": 116,
                   "time_ns": 11_600_000_000},
                  {"kind": "outcome", "reason": "navigation_lost", "tick": 175,
                   "time_ns": 17_500_000_000}]
        estimates = [{"time_ns": tick * 10**8, "target": {"capture_time_ns": 8_500_000_000}}
                     for tick in range(176)]
        response = _fault_response(lidar, events, estimates, "navigation_lost")
        self.assertEqual((response["brake_latency_s"], response["reacquisition_s"],
                          response["loss_time_s"], response["avoidance_burn_s"]),
                         (0.0, None, 11.6, 5.9))
        self.assertAlmostEqual(response["estimate_age_at_loss_s"], 3.1)


if __name__ == "__main__":
    unittest.main()
