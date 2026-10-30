"""Fast onboard-boundary and observation-timing checks, without Blender."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod.observations import ChaserInputs, NavigationEstimate, OwnNavigation
from rpod.approach import make_approach_scenario
from rpod.visual_control import MissionSpec, VisualController
from rpod.visual_run import correct_visual_manifest, run_visual


def onboard_inputs(time_ns, capture_ns=0, update=None, gap=30.0, target_offset=0.0):
    radius, mu = 6_778_137.0, 3.986004418e14
    rate = np.sqrt(mu / radius**3)
    target_r = np.array([radius + target_offset, 0.0, 0.0])
    own_r = np.array([radius, -gap - 1.5, 0.0])
    target_v = np.array([0.0, rate * radius, 0.0])
    own_v = target_v + np.cross([0, 0, rate], own_r - [radius, 0, 0])
    own = OwnNavigation(time_ns, own_r, own_v, np.array([0., 0., 0., 1.]),
                        np.array([0., 0., rate]))
    target = NavigationEstimate(
        time_ns, target_r, target_v, np.array([0., 0., 0., 1.]),
        np.array([0., 0., rate]), np.eye(12), capture_ns, "fresh",
    )
    return ChaserInputs(time_ns, own, target, update)


class VisualControlContracts(unittest.TestCase):
    def test_autonomy_boundary_uses_estimate_and_rejects_future_inputs(self):
        mission = MissionSpec()
        with self.assertRaises(FrozenInstanceError):
            mission.final_gap_m = 1
        with self.assertRaises(TypeError):
            MissionSpec(initial_target_position_m=[1, 2, 3])
        controllers = [VisualController(mission), VisualController(mission)]
        with patch("rpod.visual_run._handoff", side_effect=AssertionError("truth access")):
            for second in range(3):
                now, capture = (second * 10 + 2) * 100_000_000, second * 1_000_000_000
                decisions = [controller.update(onboard_inputs(
                    now, capture, True, target_offset=offset
                )) for controller, offset in zip(controllers, [0.0, 0.5])]
        self.assertEqual([decision.phase for decision in decisions], [1, 1])
        self.assertGreater(
            np.linalg.norm(decisions[0].force_body_n - decisions[1].force_body_n), 0.01
        )
        with self.assertRaisesRegex(ValueError, "future"):
            controllers[0].update(onboard_inputs(3_000_000_000, 4_000_000_000, True))
        aborting = VisualController(MissionSpec(abort_gap_m=20.0))
        for second in range(3):
            aborting.update(onboard_inputs(
                (second * 10 + 2) * 100_000_000, second * 1_000_000_000, True
            ))
        retreat = aborting.update(onboard_inputs(2_300_000_000, 2_000_000_000, gap=19.9))
        self.assertEqual(retreat.phase, 2)
        self.assertTrue(retreat.control_update)
        self.assertEqual(retreat.abort_reason, "estimated_gap_abort_request")

    def test_acquisition_braking_reacquisition_and_age_deadlines(self):
        controller = VisualController(MissionSpec())
        first = controller.update(onboard_inputs(200_000_000, 0, True))
        self.assertEqual(first.phase, 4)
        duplicate = controller.update(onboard_inputs(300_000_000, 0, True))
        self.assertEqual(duplicate.phase, 4)
        controller.update(onboard_inputs(1_200_000_000, 1_000_000_000, True))
        acquired = controller.update(onboard_inputs(2_200_000_000, 2_000_000_000, True))
        self.assertTrue(acquired.reacquired)
        self.assertEqual(acquired.phase, 1)
        braking = controller.update(onboard_inputs(3_200_000_000, 2_000_000_000, False))
        self.assertEqual(braking.phase, 5)
        self.assertTrue(braking.control_update)
        self.assertEqual(braking.achieved_dwell_ns, 0)
        one = controller.update(onboard_inputs(4_200_000_000, 4_000_000_000, True))
        self.assertEqual(one.phase, 5)
        resumed = controller.update(onboard_inputs(5_200_000_000, 5_000_000_000, True))
        self.assertEqual(resumed.phase, 1)
        self.assertTrue(resumed.reacquired)
        boundary = controller.update(onboard_inputs(8_000_000_000, 5_000_000_000))
        self.assertIsNone(boundary.outcome)
        lost = controller.update(onboard_inputs(8_100_000_000, 5_000_000_000))
        self.assertEqual((lost.phase, lost.outcome), (6, "navigation_lost"))
        acquiring = VisualController(MissionSpec())
        own = onboard_inputs(10_000_000_000).own
        expired = acquiring.update(ChaserInputs(own.time_ns, own, None, False))
        self.assertEqual(expired.outcome, "acquisition_failed")

    def test_invalid_frame_resets_onboard_dwell_before_completion(self):
        controller = VisualController(MissionSpec())
        last_capture = 0
        completion = None
        saw_braking = False
        for tick in range(443):
            now = tick * 100_000_000
            delivered = tick >= 2 and tick % 10 == 2
            valid = delivered and tick != 122
            if valid:
                last_capture = (tick - 2) * 100_000_000
            inputs = onboard_inputs(
                now, last_capture, valid if delivered else None, gap=5.0
            )
            if tick == 241:
                inputs = replace(inputs, target=replace(inputs.target, health="predicted"))
            elif tick == 341:
                inputs = replace(inputs, target=replace(
                    inputs.target, capture_time_ns=now - 1_300_000_000
                ))
            decision = controller.update(inputs)
            saw_braking |= decision.phase == 5
            if decision.outcome:
                completion = (tick, decision.outcome)
                break
        self.assertTrue(saw_braking)
        self.assertEqual(completion, (442, "success"))

    def test_unqualified_calibration_cannot_start_physics_or_renderer(self):
        with tempfile.TemporaryDirectory() as temporary:
            calibration = Path(temporary) / "calibration.json"
            calibration.write_text(json.dumps({"qualification_passed": False}))
            output = Path(temporary) / "run"
            with (
                patch("rpod.visual_run.visual_initial_state",
                      side_effect=AssertionError("physics started")),
                patch("rpod.rendering.BlenderRenderer",
                      side_effect=AssertionError("renderer started")),
            ):
                with self.assertRaisesRegex(ValueError, "M3 qualification"):
                    run_visual("nominal", 0, output, calibration)
            self.assertFalse(output.exists())
            calibration.write_text(json.dumps({
                "qualification_passed": True, "range_bins": [100.0],
                "translation_variance": [[0.01, 0.01, 0.01]],
                "rotation_variance": [[0.001, 0.001, 0.001]],
            }))
            initial = make_approach_scenario("approach")
            initial.update({"schema_version": 3, "scenario": "visual-abort",
                            "visual_case": "abort", "max_steps": 1, "sun_angle_deg": 0.0})

            class FailedWorker:
                provenance = {}
                performance = {}

                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    raise RuntimeError("mock worker startup failed")

                def __exit__(self, *args):
                    pass

            with (
                patch("rpod.visual_run.visual_initial_state", return_value=initial),
                patch("rpod.rendering.BlenderRenderer", FailedWorker),
            ):
                report = run_visual("abort", 0, output, calibration)
            self.assertEqual(report["outcome"], "render_or_pipeline_failure")
            self.assertIn("mock worker startup failed", report["failure_reason"])
            self.assertFalse(report["qualification_passed"])
            self.assertEqual(report["physical_achieved_dwell_ns"], 0)
            self.assertGreater(report["minimum_com_range_m"], 3)
            self.assertGreater(report["minimum_port_gap_m"], 2)
            self.assertEqual(report["max_applied_forces_n"], [0, 0])
            self.assertEqual(report["max_applied_torques_nm"], [0, 0])
            frozen = json.loads((output / "manifest.json").read_text())
            np.testing.assert_array_equal(frozen["sunlight_direction_inertial"], [0, 1, 0])
            self.assertEqual(set(frozen["autonomy_implementation_sha256"]),
                             {"visual_control.py", "navigation.py", "visual_agent.py",
                              "visual_env.py", "visual_run.py", "lidar.py"})
            self.assertEqual(frozen["goals"]["retreat_gap_m"], 30)
            self.assertEqual(frozen["scheduled_abort_gap_m"], 20)
            self.assertIn("recorded handoff", frozen["frames"]["inertial"])
            execution_hashes = frozen["autonomy_implementation_sha256"].copy()
            frozen["goals"]["retreat_gap_m"] = 60
            frozen["scheduled_abort_gap_m"] = None
            frozen["frames"]["inertial"] = "original M2 description"
            frozen.pop("record_metadata_correction")
            frozen.pop("record_writer_implementation_sha256")
            original_manifest = json.dumps(frozen).encode()
            (output / "manifest.json").write_bytes(original_manifest)
            correction = correct_visual_manifest(output)
            self.assertTrue(correction["changed"])
            corrected = json.loads((output / "manifest.json").read_text())
            self.assertEqual(corrected["autonomy_implementation_sha256"], execution_hashes)
            self.assertEqual(
                (output / "manifest.before-metadata-correction.json").read_bytes(),
                original_manifest,
            )
            self.assertEqual(corrected["goals"]["retreat_gap_m"], 30)
            self.assertEqual(corrected["scheduled_abort_gap_m"], 20)
            self.assertFalse(correct_visual_manifest(output)["changed"])
            with np.load(output / "trajectory.npz", allow_pickle=False) as record:
                self.assertEqual(record["tick"].shape, (1,))
                self.assertEqual(record["requested_forces_body_n"].shape, (0, 2, 3))
            estimate_rows = (output / "estimates.jsonl").read_text().splitlines()
            self.assertEqual(len(estimate_rows), 1)
            self.assertIsNone(json.loads(estimate_rows[0])["target"])


if __name__ == "__main__":
    unittest.main()
