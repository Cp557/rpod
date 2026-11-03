"""Surrogate error algebra, navigation timing, intervals and campaign reproducibility."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod import fleet, monte_carlo
from rpod.monte_carlo import clopper_pearson, run_campaign
from rpod.surrogate import TIME_EDGES_S, SurrogateNavigation, apply_errors, estimate_errors

from test_fleet import random_states
from test_visual_env import near_handoff


def flat_model(std=0.0, rejection=0.0):
    """Zero-mean errors with one deviation everywhere and one rejection probability."""
    bins = len(TIME_EDGES_S) + 1
    group = {"centers": [0.0, 1e3], "mean": [[0.0] * 6] * 2, "std": [[std] * 6] * 2}
    return {"position_velocity": group, "attitude_rate": group, "lag1_correlation": [0.5] * 12,
            "rejection": {"probability": [rejection] * bins}}


class Surrogate(unittest.TestCase):
    def test_injected_errors_are_recovered(self):
        truth = random_states(32, 5)[:, 0]
        errors = np.random.default_rng(6).normal(0, [0.05] * 3 + [0.003] * 3 + [0.002] * 3
                                                 + [1e-4] * 3, (32, 12))
        recovered = estimate_errors(truth, apply_errors(truth, errors))
        np.testing.assert_allclose(recovered[:, :6], errors[:, :6], atol=1e-9)
        # Small-angle injection and measurement agree to third order in the angle.
        np.testing.assert_allclose(recovered[:, 6:9], errors[:, 6:9], atol=1e-7)
        np.testing.assert_allclose(recovered[:, 9:], errors[:, 9:], atol=1e-15)

    def test_navigation_follows_camera_timing_and_rejections(self):
        states = random_states(3, 7)
        active = np.ones(3, bool)
        navigation = SurrogateNavigation(flat_model(), 3, 100_000_000, seed=1)
        for tick in range(16):
            estimates = navigation.update(tick, states, active)
            delivered = tick % 10 == 2
            self.assertTrue(np.all(estimates.visual_update == (
                fleet.ACCEPTED_UPDATE if delivered else fleet.NO_UPDATE)))
            self.assertTrue(np.all(estimates.available == (tick >= 2)))
            if tick >= 2:
                expected = (tick - (tick - 2) % 10 - 2) * 100_000_000
                self.assertTrue(np.all(estimates.capture_time_ns == expected))
                np.testing.assert_allclose(estimates.target, states[:, 0], atol=1e-12)
        rejected = SurrogateNavigation(flat_model(rejection=1.0), 3, 100_000_000, seed=1)
        estimates = [rejected.update(tick, states, active) for tick in range(3)][-1]
        self.assertTrue(np.all(estimates.visual_update == fleet.FAILED_UPDATE))
        self.assertFalse(estimates.available.any())

    def test_clopper_pearson_matches_known_values(self):
        np.testing.assert_allclose(clopper_pearson(0, 10), (0.0, 1 - 0.025 ** 0.1), atol=1e-9)
        np.testing.assert_allclose(clopper_pearson(5, 10), (0.187086, 0.812914), atol=1e-6)
        np.testing.assert_allclose(clopper_pearson(10, 10), (0.025 ** 0.1, 1.0), atol=1e-9)

    def test_campaign_is_reproducible(self):
        original = monte_carlo.episode_initial_state

        def short(spec):  # 30 s episodes keep the test fast
            return {**original(spec), "max_steps": 300}

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch("rpod.visual_run._handoff", return_value=near_handoff()), \
                patch("rpod.monte_carlo.episode_initial_state", side_effect=short):
            model = Path(directory) / "model.json"
            model.write_text(json.dumps(flat_model(std=0.002, rejection=0.02)), encoding="utf-8")
            reports = [run_campaign(Path(directory) / name, 6, 3, model, batch_size=4)
                       for name in ("first", "second")]
            arrays = []
            for name in ("first", "second"):
                with np.load(Path(directory) / name / "episodes.npz") as data:
                    arrays.append(dict(data))
            for key in arrays[0]:
                np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
            for report in reports:
                report.pop("performance")
                report["campaign"].pop("model")
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(reports[0]["episodes"], 6)


if __name__ == "__main__":
    unittest.main()
