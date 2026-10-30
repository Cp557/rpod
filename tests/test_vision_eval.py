"""Frozen M3 fixture integrity and inexpensive source-validation regressions."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod.camera import CameraModel
from rpod.cli import main
from rpod.scenarios import make_scenario
from rpod.vision_eval import _qualify, evaluate_saved_dataset


def complete_records():
    rows = []
    capture_id = 0
    for gap in (4.5, 5.0, 10.0, 20.0, 30.0, 32.0):
        for variant in range(6):
            for light in range(3):
                record = {
                    "capture_id": capture_id, "gap_m": gap, "variant": variant,
                    "light": light, "partition": "calibration" if variant in (0, 5) else "test",
                    "positive": True, "degraded": False, "pose": {"valid": True},
                    "translation_error_m": 0.0, "rotation_error_deg": 0.0,
                }
                rows.extend((record, {**record, "capture_id": capture_id + 10_000,
                                      "degraded": True}))
                capture_id += 1
    for variant in ("empty", "occluded", "out_of_view", "unknown_ids"):
        for gap in (4.5, 5.0, 10.0, 20.0, 30.0, 32.0):
            rows.append({
                "capture_id": capture_id, "gap_m": gap, "variant": variant,
                "partition": "test", "positive": False, "degraded": False,
                "target_present": variant != "empty", "pose": {"valid": False},
            })
            capture_id += 1
    return rows


class VisionEvaluationContracts(unittest.TestCase):
    def test_frozen_suite_rejects_missing_duplicate_and_relabelled_cases(self):
        records = complete_records()
        self.assertEqual(len(records), 240)
        report = _qualify(records)
        self.assertTrue(report["qualification_passed"])
        self.assertEqual(report["negative_count"], 24)
        missing_negatives = [row for row in records if row["positive"]]
        duplicate = deepcopy(records)
        duplicate[1] = deepcopy(duplicate[0])
        wrong_partition = deepcopy(records)
        wrong_partition[0]["partition"] = "test"
        wrong_light = deepcopy(records)
        wrong_light[0]["light"] = 1
        for broken in (missing_negatives, duplicate, wrong_partition, wrong_light):
            with self.subTest(rows=len(broken)), self.assertRaisesRegex(ValueError, "M3"):
                _qualify(broken)

        # Incomplete observation streams must fail before reading or estimating any image.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "truth").mkdir()
            (directory / "dataset.json").write_text(
                json.dumps({"camera": CameraModel().to_dict()}), encoding="utf-8"
            )
            for path, rows in (
                (directory / "truth" / "poses.jsonl", records),
                (directory / "observations.jsonl", duplicate),
            ):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with (
                patch("rpod.vision_eval.read_rgb", side_effect=AssertionError("image read")),
                patch("rpod.vision_eval.estimate_pose", side_effect=AssertionError("estimation")),
                self.assertRaisesRegex(ValueError, "M3"),
            ):
                evaluate_saved_dataset(directory)

        rejected = next(row for row in records if row["capture_id"] == 3)
        rejected["pose"] = {"valid": False, "reason": "insufficient_markers"}
        worst = next(row for row in records if row["capture_id"] == 60)
        worst.update(translation_error_m=0.1, rotation_error_deg=1.0)
        report = _qualify(records)
        self.assertEqual(report["rejected_positive_cases"][0]["capture_id"], 3)
        self.assertEqual(report["rejected_positive_cases"][0]["reason"], "insufficient_markers")
        for metric in ("translation", "rotation"):
            self.assertEqual(report["worst_accepted_cases"][metric][0]["capture_id"], 60)

    def test_schema_one_source_is_rejected_before_output_or_renderer_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "m1", Path(temporary) / "evaluation"
            source.mkdir()
            manifest = make_scenario("circular")
            manifest["num_steps"] = 0
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(
                source / "trajectory.npz", tick=np.array([0], dtype=np.uint64),
                positions_m=np.array([manifest["initial_positions_m"]], dtype=np.float64),
                velocities_mps=np.array([manifest["initial_velocities_mps"]], dtype=np.float64),
            )
            errors = io.StringIO()
            with (
                patch("rpod.vision_eval.BlenderRenderer", side_effect=AssertionError("renderer")),
                redirect_stderr(errors),
            ):
                status = main([
                    "perception-eval", "--source", str(source), "--output", str(output),
                ])
            self.assertEqual(status, 1)
            self.assertIn("M2 schema-2", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
