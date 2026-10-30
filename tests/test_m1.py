"""Focused acceptance checks for bindings, relative frames, and saved experiments."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from rpod import OrbitEngine
from rpod.frames import relative_state
from rpod.recording import collect_trajectory, load_run, replay_run, run_scenario, summarize
from rpod.scenarios import make_scenario


def make_engine(scenario="circular"):
    config = make_scenario(scenario)
    return OrbitEngine(
        config["mu_m3_s2"],
        config["dt_ns"],
        config["initial_positions_m"],
        config["initial_velocities_mps"],
    )


class BindingContracts(unittest.TestCase):
    def test_snapshot_reset_and_chunked_stepping(self):
        engine = make_engine()
        initial = engine.snapshot()
        self.assertIsInstance(initial.positions_m, tuple)
        self.assertIsInstance(initial.positions_m[0], tuple)
        with self.assertRaises(AttributeError):
            initial.tick = 3
        with self.assertRaises(TypeError):
            initial.positions_m[0][0] = 0
        after = engine.step(20)
        self.assertEqual((after.tick, after.time_ns), (20, 20_000_000_000))
        self.assertEqual(initial.tick, 0)
        self.assertEqual(engine.step(0).positions_m, after.positions_m)
        reset = engine.reset()
        self.assertEqual(
            (reset.tick, reset.time_ns, reset.positions_m, reset.velocities_mps),
            (0, 0, initial.positions_m, initial.velocities_mps),
        )
        for _ in range(20):
            individual = engine.step()
        self.assertEqual(individual.positions_m, after.positions_m)
        self.assertEqual(individual.velocities_mps, after.velocities_mps)

    def test_invalid_python_inputs_and_overflow_preserve_state(self):
        config = make_scenario("circular")
        positions, velocities = config["initial_positions_m"], config["initial_velocities_mps"]
        for mu, dt, r, v in (
            (0.0, 1, positions, velocities),
            (config["mu_m3_s2"], 0, positions, velocities),
            (config["mu_m3_s2"], 1, [[0.0]*3]*2, velocities),
            (config["mu_m3_s2"], 1, [[float("nan")]*3]*2, velocities),
        ):
            with self.subTest(mu=mu, dt=dt), self.assertRaises(ValueError):
                OrbitEngine(mu, dt, r, v)
        with self.assertRaises((ValueError, TypeError)):
            OrbitEngine(config["mu_m3_s2"], 1, [[1.0, 2.0]], velocities)
        engine = make_engine()
        with self.assertRaises(OverflowError):
            engine.step((2**64 - 1) // config["dt_ns"] + 1)
        self.assertEqual(engine.snapshot().tick, 0)


class PhysicalRecords(unittest.TestCase):
    def test_stationary_rotating_frame_and_long_orbit_accuracy(self):
        manifest = make_scenario("circular")
        trajectory = collect_trajectory(make_engine(), 6000)
        rho, rho_dot, basis = relative_state(
            trajectory["positions_m"], trajectory["velocities_mps"]
        )
        self.assertLess(np.linalg.norm(rho - rho[0], axis=-1).max(), 1e-5)
        self.assertLess(np.linalg.norm(rho_dot, axis=-1).max(), 1e-8)
        np.testing.assert_allclose(basis[0], np.eye(3), atol=1e-15)
        summary = summarize(manifest, trajectory)
        self.assertLess(max(summary["analytical_max_position_error_m"]), 1e-3)
        self.assertLess(max(summary["analytical_max_velocity_error_mps"]), 1e-6)
        self.assertLess(max(summary["max_fractional_specific_energy_change"]), 1e-10)
        self.assertLess(max(summary["max_fractional_angular_momentum_vector_change"]), 1e-10)

    def test_drift_initialization_and_degenerate_frame(self):
        circular, drift = make_scenario("circular"), make_scenario("drift")
        np.testing.assert_array_equal(circular["initial_positions_m"], drift["initial_positions_m"])
        np.testing.assert_allclose(
            np.asarray(drift["initial_velocities_mps"])[1]
            - np.asarray(circular["initial_velocities_mps"])[1],
            [0, 0.05, 0.02],
            atol=1e-11,
        )
        with self.assertRaisesRegex(ValueError, "undefined"):
            relative_state([[1, 0, 0], [2, 0, 0]], [[1, 0, 0], [1, 0, 0]])


class SavedPipeline(unittest.TestCase):
    def test_both_scenarios_round_trip_and_replay_without_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            for scenario in ("circular", "drift"):
                with self.subTest(scenario=scenario):
                    output = Path(temporary) / scenario
                    summary = run_scenario(scenario, output)
                    manifest, trajectory = load_run(output)
                    self.assertEqual(summary["samples"], 601)
                    np.testing.assert_array_equal(
                        trajectory["tick"], np.arange(601, dtype=np.uint64)
                    )
                    expected = collect_trajectory(make_engine(scenario), 600)
                    for key in trajectory:
                        np.testing.assert_array_equal(trajectory[key], expected[key])
                    if scenario == "drift":
                        self.assertNotIn("analytical_max_position_error_m", summary)
                        self.assertGreater(summary["max_relative_position_change_m"], 10.0)
                    with (
                        patch(
                            "rpod.recording.OrbitEngine",
                            side_effect=AssertionError("replay ran physics"),
                        ),
                        patch("rpod.visualization.open_replay") as viewer,
                    ):
                        path = replay_run(output, save_only=True)
                        viewer.assert_not_called()
                    self.assertGreater(path.stat().st_size, 0)
                    with self.assertRaisesRegex(ValueError, "empty directory"):
                        run_scenario(scenario, output)
                    self.assertEqual(manifest["body_order"], ["target", "chaser"])

    def test_loader_rejects_corrupted_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_scenario("circular", output)
            manifest, original = load_run(output)
            for field in ("tick", "positions_m", "velocities_mps"):
                broken = {key: array.copy() for key, array in original.items()}
                if field == "tick":
                    broken[field][5] += 1
                else:
                    broken[field][5, 0, 0] = np.nan
                np.savez(output / "trajectory.npz", **broken)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_run(output)
            np.savez(output / "trajectory.npz", **original)
            manifest["schema_version"] = 999
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_run(output)
            manifest["schema_version"] = 1
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (output / "trajectory.npz").write_bytes(b"PK\x03\x04bad")
            with self.assertRaisesRegex(ValueError, "invalid numerical record"):
                load_run(output)
            with ZipFile(output / "trajectory.npz", "w") as archive:
                for key in original:
                    archive.writestr(f"{key}.npy", b"invalid NumPy payload")
            with self.assertRaisesRegex(ValueError, "NumPy arrays"):
                load_run(output)

    def test_module_cli_reports_errors_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "existing.txt").write_text("preserve", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "rpod", "run", "--output", temporary],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("empty directory", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
