"""Focused checks for port kinematics, supervised approach, and controlled records."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod.approach import Dwell, make_approach_scenario, simulate_approach
from rpod.frames import port_relative_state, quaternion_matrices
from rpod.recording import load_events, load_run, replay_run, run_scenario


class ControlledApproach(unittest.TestCase):
    def assert_command_contract(self, manifest, trajectory, events):
        samples = len(trajectory["tick"])
        np.testing.assert_array_equal(
            trajectory["tick"], np.arange(samples, dtype=np.uint64)
        )
        for field in (
            "positions_m", "velocities_mps", "quaternions_xyzw",
            "angular_velocities_body_radps",
        ):
            self.assertTrue(np.isfinite(trajectory[field]).all(), field)
        np.testing.assert_allclose(
            np.linalg.norm(trajectory["quaternions_xyzw"], axis=-1),
            1.0, rtol=0, atol=1e-12,
        )
        updates = trajectory["control_update"]
        self.assertEqual(updates.shape, (samples - 1,))
        self.assertTrue(updates[0])
        for quantity, units, limit in (
            ("forces", "n", "force_limits_n"),
            ("torques", "nm", "torque_limits_nm"),
        ):
            for prefix in ("requested", "applied"):
                field = f"{prefix}_{quantity}_body_{units}"
                self.assertEqual(trajectory[field].shape, (samples - 1, 2, 3))
                self.assertTrue(np.isfinite(trajectory[field]).all(), field)
            applied = trajectory[f"applied_{quantity}_body_{units}"]
            self.assertTrue(
                np.all(np.linalg.norm(applied, axis=-1)
                       <= np.asarray(manifest[limit]) + 1e-12)
            )
            requested = trajectory[f"requested_{quantity}_body_{units}"]
            np.testing.assert_array_equal(
                requested[1:][~updates[1:]], requested[:-1][~updates[1:]]
            )
        for event in events:
            self.assertEqual(event["time_ns"], event["tick"] * manifest["dt_ns"])
            if event["reason"] in ("terminal_transfer_started", "retreat_started"):
                self.assertTrue(updates[event["tick"]])

    def assert_goal_dwell(self, manifest, trajectory, goal, end_tick):
        acceptance = manifest["acceptance"]
        dwell_ticks = acceptance["dwell_ns"] // manifest["dt_ns"]
        self.assertGreaterEqual(end_tick, dwell_ticks)
        selection = slice(end_tick - dwell_ticks, end_tick + 1)
        self.assertEqual(
            int(trajectory["tick"][end_tick]
                - trajectory["tick"][end_tick - dwell_ticks]) * manifest["dt_ns"],
            acceptance["dwell_ns"],
        )
        positions = trajectory["positions_m"][selection]
        velocities = trajectory["velocities_mps"][selection]
        quaternions = trajectory["quaternions_xyzw"][selection]
        body_rates = trajectory["angular_velocities_body_radps"][selection]
        port_r, port_v, _ = port_relative_state(
            positions, velocities, quaternions, body_rates,
            manifest["docking_ports_body_m"],
        )
        self.assertLessEqual(
            np.linalg.norm(port_r - [0, -goal, 0], axis=-1).max(),
            acceptance["position_error_m"],
        )
        self.assertLessEqual(
            np.linalg.norm(port_v, axis=-1).max(), acceptance["speed_mps"]
        )
        rotations = quaternion_matrices(quaternions)
        relative = np.swapaxes(rotations[:, 0], -1, -2) @ rotations[:, 1]
        angle = np.degrees(np.arccos(np.clip(
            (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1
        )))
        self.assertLessEqual(angle.max(), acceptance["alignment_deg"])
        inertial_rates = np.einsum("tbij,tbj->tbi", rotations, body_rates)
        self.assertLessEqual(
            np.linalg.norm(inertial_rates[:, 1] - inertial_rates[:, 0], axis=-1).max(),
            acceptance["relative_rate_radps"],
        )
        np.testing.assert_array_equal(trajectory["desired_gap_m"][selection], goal)

    def test_offset_ports_match_independent_rotating_frame_finite_difference(self):
        axes = np.array([[1.0, 2.0, -1.0], [-2.0, 1.0, 3.0]])
        axes /= np.linalg.norm(axes, axis=-1)[:, None]
        spin_rates = np.array([0.3, -0.4])
        offsets = np.array([[0.7, -1.0, 0.2], [-0.4, 0.5, 0.3]])
        chaser_velocity = np.array([1.0, -2.0, 0.5])

        def sample(time):
            angle = 0.2 * time
            cosine, sine = math.cos(angle), math.sin(angle)
            basis = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            positions = np.array([
                [1000 * cosine, 1000 * sine, 0],
                np.array([1010.0, -30.0, 7.0]) + chaser_velocity * time,
            ])
            velocities = np.array([
                [-200 * sine, 200 * cosine, 0], chaser_velocity,
            ])
            rotations, quaternions = [], []
            for axis, initial, rate in zip(axes, [0.4, -0.2], spin_rates):
                theta = initial + rate * time
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotations.append(
                    math.cos(theta) * np.eye(3)
                    + (1 - math.cos(theta)) * np.outer(axis, axis)
                    + math.sin(theta) * skew
                )
                quaternions.append(np.r_[axis * math.sin(theta / 2),
                                         math.cos(theta / 2)])
            rotations = np.asarray(rotations)
            port_positions = positions + np.einsum("bij,bj->bi", rotations, offsets)
            expected = basis.T @ (port_positions[1] - port_positions[0])
            return positions, velocities, quaternions, rotations, expected

        time, delta = 0.7, 1e-4
        positions, velocities, quaternions, rotations, expected = sample(time)
        body_rates = axes * spin_rates[:, None]
        port_r, port_v, _ = port_relative_state(
            positions, velocities, quaternions, body_rates, offsets
        )
        expected_velocity = (sample(time + delta)[-1] - sample(time - delta)[-1])
        expected_velocity /= 2 * delta
        np.testing.assert_allclose(port_r, expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(port_v, expected_velocity, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            quaternion_matrices(quaternions), rotations, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            quaternion_matrices(-np.asarray(quaternions)), rotations,
            rtol=0, atol=1e-14,
        )

    def test_nominal_and_disturbed_dwell_commands_and_saved_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("approach", "approach-disturbed"):
                with self.subTest(scenario=name):
                    output = Path(temporary) / name
                    outcome = run_scenario(name, output)
                    manifest, trajectory = load_run(output)
                    events = load_events(output, manifest)
                    self.assertEqual(outcome["outcome"], "success")
                    self.assertIsNone(outcome["abort_reason"])
                    self.assert_command_contract(manifest, trajectory, events)
                    phases = trajectory["phase"]
                    np.testing.assert_array_equal(
                        phases[np.r_[True, phases[1:] != phases[:-1]]], [0, 1, 3]
                    )
                    hold = next(event for event in events
                                if event["reason"] == "hold_dwell_satisfied")
                    self.assert_goal_dwell(
                        manifest, trajectory, manifest["goals"]["hold_gap_m"],
                        hold["tick"],
                    )
                    self.assert_goal_dwell(
                        manifest, trajectory, manifest["goals"]["final_gap_m"],
                        int(trajectory["tick"][-1]),
                    )
                    with (
                        patch("rpod.approach.make_engine",
                              side_effect=AssertionError("replay ran physics")),
                        patch("rpod.visualization.open_replay") as viewer,
                    ):
                        replay = replay_run(output, save_only=True)
                        viewer.assert_not_called()
                    self.assertGreater(replay.stat().st_size, 0)
                    if name == "approach":
                        for field in ("quaternions_xyzw",
                                      "angular_velocities_body_radps",
                                      "applied_forces_body_n"):
                            broken = {key: array.copy()
                                      for key, array in trajectory.items()}
                            if field == "quaternions_xyzw":
                                broken[field][5, 1] *= 2
                            elif field == "angular_velocities_body_radps":
                                broken[field][5, 1, 0] = np.nan
                            else:
                                broken[field] = broken[field][:-1]
                            np.savez(output / "trajectory.npz", **broken)
                            with self.subTest(corrupt_field=field):
                                with self.assertRaises(ValueError):
                                    load_run(output)
                        broken_events = [dict(event) for event in events]
                        broken_events[0]["time_ns"] += 1
                        (output / "events.json").write_text(
                            json.dumps(broken_events), encoding="utf-8"
                        )
                        with self.assertRaisesRegex(ValueError, "event"):
                            load_events(output, manifest)
                        broken_events[0]["time_ns"] = events[0]["time_ns"]
                        broken_events[0]["kind"] = ["outcome"]
                        (output / "events.json").write_text(
                            json.dumps(broken_events), encoding="utf-8"
                        )
                        with self.assertRaisesRegex(ValueError, "event"):
                            load_events(output, manifest)

    def test_requested_retreat_updates_at_crossing_and_retains_cause(self):
        manifest = make_approach_scenario("approach-retreat")
        trajectory, events, outcome = simulate_approach(manifest)
        self.assertEqual(outcome["outcome"], "retreated")
        requests = [event for event in events if event["kind"] == "abort_request"]
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(outcome["abort_reason"], request["reason"])
        self.assertIn(request["reason"], events[-1]["reason"])
        self.assert_command_contract(manifest, trajectory, events)
        phases = trajectory["phase"]
        np.testing.assert_array_equal(
            phases[np.r_[True, phases[1:] != phases[:-1]]], [0, 1, 2, 3]
        )
        port_r, _, _ = port_relative_state(
            trajectory["positions_m"], trajectory["velocities_mps"],
            trajectory["quaternions_xyzw"],
            trajectory["angular_velocities_body_radps"],
            manifest["docking_ports_body_m"],
        )
        approach_started = next(event["tick"] for event in events
                                if event["reason"] == "terminal_transfer_started")
        gap = -port_r[:, 1]
        eligible = ((trajectory["tick"] >= approach_started)
                    & (gap <= manifest["scheduled_abort_gap_m"]))
        self.assertEqual(request["tick"], int(np.flatnonzero(eligible)[0]))
        self.assertTrue(trajectory["control_update"][request["tick"]])
        self.assertAlmostEqual(
            trajectory["desired_gap_m"][request["tick"]],
            gap[request["tick"]], delta=1e-10,
        )
        self.assert_goal_dwell(
            manifest, trajectory, manifest["goals"]["retreat_gap_m"],
            int(trajectory["tick"][-1]),
        )
        for separation, steps, boundary in (
            (None, 1, None),
            (3.52, 2, "minimum_axial_port_gap"),
            (2.5, 1, "minimum_com_separation"),
        ):
            with self.subTest(initial_com_separation=separation):
                unsafe = make_approach_scenario("approach")
                unsafe["num_steps"] = steps
                unsafe["initial_velocities_mps"][1] = (
                    np.asarray(unsafe["initial_velocities_mps"][0]) + [0, 0.6, 0]
                ).tolist()
                if separation is not None:
                    unsafe["initial_positions_m"][1] = (
                        np.asarray(unsafe["initial_positions_m"][0])
                        + [0, -separation, 0]
                    ).tolist()
                record, detections, result = simulate_approach(unsafe)
                self.assertEqual(result["outcome"], "failed")
                self.assertEqual(result["achieved_final_dwell_ns"], 0)
                if boundary == "minimum_com_separation":
                    self.assertEqual(result["abort_reason"], boundary)
                    self.assertEqual(int(record["tick"][-1]), 0)
                else:
                    self.assertEqual(result["abort_reason"], "closing_speed_limit")
                    soft = [event for event in detections
                            if event["kind"] == "safety_detection"]
                    self.assertEqual(len(soft), 1)
                    self.assertEqual((soft[0]["tick"], soft[0]["reason"]),
                                     (0, "closing_speed_limit"))
                    self.assertEqual(record["phase"][0], 2)
                    self.assertTrue(record["control_update"][0])
                    self.assertEqual(int(record["tick"][-1]), 1)
                hard = [event for event in detections
                        if event["kind"] == "physical_boundary"]
                if boundary is None:
                    self.assertEqual(hard, [])
                else:
                    self.assertEqual(len(hard), 1)
                    self.assertEqual(hard[0]["reason"], boundary)
                    if boundary == "minimum_axial_port_gap":
                        self.assertEqual((hard[0]["tick"], hard[0]["phase"]), (1, 2))

    def test_dwell_starts_at_first_valid_time_and_restarts_after_invalid(self):
        duration, tick = 10_000_000_000, 100_000_000
        dwell = Dwell(duration)
        start = 7 * tick
        self.assertFalse(dwell.update(start, True))
        self.assertFalse(dwell.update(start + duration - tick, True))
        self.assertTrue(dwell.update(start + duration, True))
        self.assertFalse(dwell.update(start + duration + tick, False))
        restarted = start + duration + 2 * tick
        self.assertFalse(dwell.update(restarted, True))
        self.assertFalse(dwell.update(restarted + duration - tick, True))
        self.assertTrue(dwell.update(restarted + duration, True))


if __name__ == "__main__":
    unittest.main()
