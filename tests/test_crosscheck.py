"""The Basilisk comparison path, exercised against rpod's own trajectory."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rpod.crosscheck import cases, compare, mrp_quaternions, rpod_trajectory


class CrossCheck(unittest.TestCase):
    def test_mrp_quaternions_match_axis_angle(self):
        axis, angle = np.array([0.6, -0.8, 0.0]), 2.5
        quaternion = mrp_quaternions([np.tan(angle / 4) * axis])[0]
        np.testing.assert_allclose(quaternion, [*np.sin(angle / 2) * axis, np.cos(angle / 2)],
                                   atol=1e-15)

    def test_identical_trajectories_pass_and_a_shifted_one_fails(self):
        case = {**next(case for case in cases() if case["name"] == "torque_free_tumble"),
                "duration_s": 100.0}
        ours = rpod_trajectory(case)
        quaternions = ours[:, 6:10] * np.sign(ours[:, 9:10])
        external = {"positions_m": ours[:, :3].tolist(), "velocities_mps": ours[:, 3:6].tolist(),
                    "mrp": (quaternions[:, :3] / (1 + quaternions[:, 3:])).tolist(),
                    "body_rates": ours[:, 10:].tolist()}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "cases.json").write_text(json.dumps({"cases": [case]}), encoding="utf-8")
            for name, shift in (("same", 0.0), ("shifted", 0.01)):
                shifted = {**external, "positions_m": (ours[:, :3] + shift).tolist()}
                (directory / f"{name}.json").write_text(json.dumps(
                    {"software": {}, "cases": {case["name"]: shifted}}), encoding="utf-8")
                report = compare(directory / "cases.json", directory / f"{name}.json",
                                 directory / f"{name}-report.json")
                self.assertEqual(report["passed"], shift == 0.0)


if __name__ == "__main__":
    unittest.main()
