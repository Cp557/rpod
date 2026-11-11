"""M14 pieces: thruster allocation, pulse modulation, fault detection and the suite's scoring."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from rpod._core import RigidBodyEngine
from rpod.approach import make_approach_scenario
from rpod.m14_suite import FAULT_GROUPS, TRIGGER_GAPS_M, _thruster_check, m14_specs
from rpod.thrusters import (COUNT, MINIMUM_ON_TIME_S, ThrusterEngine, Thrusters, allocate,
                            axis_authority, hardware_matrix, modulate, nominal_matrix,
                            wrench_bounds)

DT_S = 0.1


def least_duty(matrix, wrench, slack=1e-9):
    """The least total duty producing `wrench` (to within `slack` per component), from
    OpenCV's simplex solver; exact equality pairs are too tight for it."""
    count = matrix.shape[1]
    constraints = np.vstack([np.hstack([matrix, (wrench + slack)[:, None]]),
                             np.hstack([-matrix, (slack - wrench)[:, None]]),
                             np.hstack([np.eye(count), np.ones((count, 1))])])
    status, solution = cv2.solveLP(-np.ones(count), constraints, 1e-9)
    assert status in (cv2.SOLVELP_SINGLE, cv2.SOLVELP_MULTI), status
    return float(np.sum(solution))


def fly(thrusters, ticks, seed, fault_tick=None):
    """Hold-like requests on the real engine: small random forces, rate damping."""
    manifest = make_approach_scenario("approach")
    force_limit, torque_limit = wrench_bounds()
    engine = ThrusterEngine(RigidBodyEngine(
        manifest["mu_m3_s2"], manifest["dt_ns"], manifest["initial_positions_m"],
        manifest["initial_velocities_mps"], manifest["initial_quaternions_xyzw"],
        manifest["initial_angular_velocities_body_radps"], manifest["masses_kg"],
        manifest["inertias_kg_m2"], [2.0, force_limit], [0.5, torque_limit]), thrusters, True)
    rng, force, torque = np.random.default_rng(seed), np.zeros(3), np.zeros(3)
    snapshots = []
    for tick in range(ticks):
        if tick == fault_tick:
            thrusters.fail(tick)
        if tick % 10 == 2:
            rate = np.asarray(engine.snapshot().angular_velocities_body_radps[1])
            force = rng.normal(size=3) * (1.0 if rng.random() < 0.1 else 0.05)
            torque = -np.array([20.0, 25.0, 30.0]) * 0.8 * rate + rng.normal(size=3) * 0.02
        snapshots.append(engine.step(np.array([np.zeros(3), force]),
                                     np.array([np.zeros(3), torque])))
    return snapshots


class M14(unittest.TestCase):
    def test_allocation_is_exact_and_least_propellant_inside_m13s_limits(self):
        matrix, rng = nominal_matrix(), np.random.default_rng(1)
        for _ in range(20):
            force, torque = rng.normal(size=3), rng.normal(size=3)
            wrench = np.r_[force / np.linalg.norm(force) * rng.uniform(0, 2),
                           torque / np.linalg.norm(torque) * rng.uniform(0, 0.5)]
            duty, scale = allocate(matrix, wrench)
            self.assertEqual(scale, 1.0)
            np.testing.assert_allclose(matrix @ duty, wrench, atol=1e-9)
            self.assertTrue(np.all((duty >= 0) & (duty <= 1)))
            self.assertAlmostEqual(duty.sum(), least_duty(matrix, wrench), places=6)

    def test_a_request_beyond_authority_is_scaled_along_its_direction(self):
        reduced = np.delete(nominal_matrix(), 0, axis=1)  # thruster 0 pushes +X
        wrench = np.r_[2.0, 0, 0, 0, 0, 0]
        duty, scale = allocate(reduced, wrench)
        self.assertAlmostEqual(scale, axis_authority(reduced)[0, 0] / 2.0, places=6)
        self.assertLess(scale, 1.0)
        np.testing.assert_allclose(reduced @ duty, scale * wrench, atol=1e-9)

    def test_any_single_thruster_loss_keeps_six_axis_control(self):
        np.testing.assert_allclose(axis_authority(nominal_matrix()),
                                   [[2, 2], [4, 4], [2, 2], [1.44, 1.44], [1.44, 1.44],
                                    [1.44, 1.44]], atol=1e-9)
        for thruster in range(COUNT):
            remaining = axis_authority(np.delete(nominal_matrix(), thruster, axis=1))
            self.assertGreaterEqual(remaining.min(), 0.72 - 1e-9)

    def test_the_bounds_cover_every_thruster_combination(self):
        force_limit, torque_limit = wrench_bounds()
        for stream in ([1, 0, 6], [1, 1, 6], [2, 5, 6]):
            hardware = hardware_matrix(stream)
            for on in (np.ones(COUNT), np.random.default_rng(stream).integers(0, 2, COUNT)):
                self.assertLessEqual(np.linalg.norm(hardware[:3] @ on), force_limit)
                self.assertLessEqual(np.linalg.norm(hardware[3:] @ on), torque_limit)
            self.assertFalse(np.allclose(hardware, nominal_matrix()))

    def test_pulse_modulation_conserves_impulse_above_the_minimum_on_time(self):
        duty = np.array([0.0, 0.03, 0.08, 0.15, 1.0])
        accumulated, total = np.zeros(5), np.zeros(5)
        for _ in range(100):
            on, accumulated = modulate(duty, accumulated, DT_S)
            fired = on[on > 0]
            self.assertTrue(np.all((fired >= MINIMUM_ON_TIME_S - 1e-15) & (fired <= DT_S)))
            total += on
        np.testing.assert_allclose(total + accumulated, duty * DT_S * 100, atol=1e-12)
        self.assertTrue(np.all(accumulated < MINIMUM_ON_TIME_S))
        on, accumulated = modulate(np.zeros(5), accumulated, DT_S)
        self.assertFalse(np.any(on) or np.any(accumulated))

    def test_the_engine_applies_the_hardwares_output_and_nothing_is_isolated(self):
        thrusters = Thrusters([3, 0, 6], [3, 0, 7], 100.0, [20, 25, 30], DT_S)
        snapshots = fly(thrusters, 600, seed=3)
        records = thrusters.records()
        produced = records["actual_on_s"] @ thrusters.hardware.T / DT_S
        applied = np.array([np.r_[snapshot.applied_forces_body_n[1],
                                  snapshot.applied_torques_body_nm[1]]
                            for snapshot in snapshots])
        np.testing.assert_allclose(applied, produced, atol=1e-12)
        self.assertEqual(thrusters.isolations, [])
        self.assertLess(records["statistic"].max(), 10.0)

    def test_each_fault_isolates_its_thruster_and_not_the_opposite_nozzle(self):
        # Thrusters 14 and 15 share a pod and push +Y and -Y: exactly opposite wrenches.
        for kind in ("stuck_on", "stuck_off"):
            for thruster in (0, 14, 15):
                with self.subTest(kind=kind, thruster=thruster):
                    thrusters = Thrusters([4, thruster, 6], [4, thruster, 7], 100.0,
                                          [20, 25, 30], DT_S, {"kind": kind,
                                                               "candidates": [thruster]})
                    fly(thrusters, 500, seed=thruster, fault_tick=200)
                    self.assertEqual(thrusters.failed_thruster, thruster)
                    commanded = np.nonzero(
                        thrusters.records()["commanded_on_s"][200:, thruster])[0]
                    start = 200 if kind == "stuck_on" else 200 + commanded[0]
                    (isolation,) = thrusters.isolations
                    self.assertEqual((isolation["thruster"], isolation["kind"]), (thruster, kind))
                    self.assertLessEqual(isolation["tick"] - start, 2)
                    self.assertFalse(thrusters.healthy[thruster])
                    self.assertFalse(np.any(thrusters.records()["actual_on_s"]
                                            [isolation["tick"]:, thruster]))

    def test_the_suite_flies_every_fault_once_on_its_own_episode(self):
        specs = m14_specs()
        nominal = [spec for spec in specs if spec["category"] == "nominal"]
        faults = [spec for spec in specs if spec["category"] == "thruster-fault"]
        self.assertEqual((len(nominal), len(faults)), (28, 12))
        self.assertEqual(len({spec["id"] for spec in specs}), 40)
        self.assertEqual([spec["index"] for spec in faults], [spec["index"]
                                                              for spec in nominal[:12]])
        self.assertEqual({(spec["thruster_fault"]["kind"], spec["thruster_fault"]["group"],
                           spec["thruster_fault"]["trigger_gap_m"]) for spec in faults},
                         {(kind, group, gap) for kind in ("stuck_on", "stuck_off")
                          for group in FAULT_GROUPS for gap in TRIGGER_GAPS_M})
        directions = nominal_matrix()[:3]
        self.assertTrue(np.all(directions[1, FAULT_GROUPS["closing"]] == 1))
        self.assertTrue(np.all(directions[1, FAULT_GROUPS["lateral"]] == 0))
        for spec in specs:
            self.assertEqual(spec["streams"]["thruster_errors"][2], 6)
            self.assertEqual(spec["streams"]["imu"][2], 7)

    def test_the_fault_hits_the_candidate_that_fired_longest_just_before(self):
        thrusters = Thrusters([5, 0, 6], [5, 0, 7], 100.0, [20, 25, 30], DT_S,
                              {"kind": "stuck_off", "candidates": [2, 6, 10, 14]})
        for tick in range(150):
            thrusters.fire([0.0, 1.0, 0.0] if tick < 40 else [0.3, 0.0, 0.0], np.zeros(3))
            thrusters.observe(np.zeros(3), tick + 1, False)
        used = thrusters.records()["commanded_on_s"][-100:]
        thrusters.fail(150)
        self.assertEqual(thrusters.failed_thruster,
                         [2, 6, 10, 14][int(np.argmax(used[:, [2, 6, 10, 14]].sum(axis=0)))])

    def test_scoring_times_a_stuck_off_fault_from_its_first_command(self):
        spec = next(spec for spec in m14_specs() if spec["thruster_fault"]
                    and spec["thruster_fault"]["kind"] == "stuck_off")
        thruster, steps = spec["thruster_fault"]["candidates"][0], 100
        hardware = hardware_matrix(spec["streams"]["thruster_errors"])
        commanded = np.zeros((steps, COUNT))
        commanded[30:40, thruster] = 0.01          # the busiest candidate before the fault
        commanded[60, thruster] = 0.01
        actual = commanded.copy()
        actual[60:, thruster] = 0.0
        produced = actual @ hardware.T / DT_S
        trajectory = {"applied_forces_body_n": np.stack([np.zeros((steps, 3)),
                                                         produced[:, :3]], axis=1),
                      "applied_torques_body_nm": np.stack([np.zeros((steps, 3)),
                                                           produced[:, 3:]], axis=1)}
        event = {"kind": "safety_detection", "tick": 62,
                 "reason": f"thruster_{thruster:02d}_stuck_off_isolated"}
        summary = {"thrusters": {"fault_tick": 40, "failed_thruster": thruster,
                                 "propellant_kg": 0.0, "saturated_ticks": 0}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory / "thrusters.npz", commanded_on_s=commanded, actual_on_s=actual,
                     statistic=np.zeros(steps), monitoring=np.ones(steps, bool),
                     hardware_matrix=hardware)
            check = _thruster_check(directory, {"dt_ns": 100_000_000}, trajectory, [event],
                                    summary, spec)
            self.assertLess(check["applied_error"], 1e-12)
            self.assertAlmostEqual(check["detection_latency_s"], 0.2)
            self.assertTrue(check["correct"])
            late = {**event, "tick": 81}
            self.assertFalse(_thruster_check(directory, {"dt_ns": 100_000_000}, trajectory,
                                             [late], summary, spec)["correct"])
            commanded[60, thruster] = 0.0
            np.savez(directory / "thrusters.npz", commanded_on_s=commanded, actual_on_s=actual,
                     statistic=np.zeros(steps), monitoring=np.ones(steps, bool),
                     hardware_matrix=hardware)
            unused = _thruster_check(directory, {"dt_ns": 100_000_000}, trajectory, [],
                                     summary, spec)
            self.assertFalse(unused["exercised"])
            self.assertTrue(unused["correct"])
            json.dumps(unused)
            # A command after the latch, with monitoring off, does not count as exercise.
            commanded[90, thruster] = 0.01
            monitoring = np.arange(steps) < 85
            np.savez(directory / "thrusters.npz", commanded_on_s=commanded, actual_on_s=actual,
                     statistic=np.zeros(steps), monitoring=monitoring, hardware_matrix=hardware)
            latched = _thruster_check(directory, {"dt_ns": 100_000_000}, trajectory, [],
                                      summary, spec)
            self.assertFalse(latched["exercised"])
            self.assertTrue(latched["correct"])
            # The recorded failed thruster must be the busiest candidate before the fault.
            other = next(index for index in spec["thruster_fault"]["candidates"]
                         if index != thruster)
            with self.assertRaisesRegex(ValueError, "fired longest"):
                _thruster_check(directory, {"dt_ns": 100_000_000}, trajectory, [],
                                {"thrusters": {**summary["thrusters"], "failed_thruster": other}},
                                spec)


if __name__ == "__main__":
    unittest.main()
