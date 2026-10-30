"""Saved visual-data contracts and replay, using tiny synthetic record fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rerun as rr

from rpod.recording import load_events, load_run, replay_run
from rpod.visual_recording import load_visual_run
from rpod.vision_records import sha256_file, write_jsonl, write_rgb
from rpod.vision_replay import export_dataset


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def write_fixture(directory):
    directory = Path(directory)
    (directory / "images").mkdir(parents=True)
    calibration = {"qualification_passed": True}
    write_json(directory / "calibration.json", calibration)
    image = directory / "images" / "000000.png"
    write_rgb(image, np.full((8, 8, 3), 100, dtype=np.uint8))
    positions = np.array([[[6_778_137.0, 0.0, 0.0], [6_778_137.0, -31.5, 0.0]]] * 3)
    velocity = np.array([[[0.0, 7_668.0, 0.0]] * 2] * 3)
    quaternion = np.array([[[0.0, 0.0, 0.0, 1.0]] * 2] * 3)
    rates = np.zeros((3, 2, 3))
    trajectory = {
        "tick": np.arange(3, dtype=np.uint64),
        "positions_m": positions,
        "velocities_mps": velocity,
        "quaternions_xyzw": quaternion,
        "angular_velocities_body_radps": rates,
        "own_position_m": positions[:, 1].copy(),
        "own_velocity_mps": velocity[:, 1].copy(),
        "own_quaternion_xyzw": quaternion[:, 1].copy(),
        "own_angular_velocity_body_radps": rates[:, 1].copy(),
        "phase": np.array([4, 4, 3], dtype=np.uint8),
        "desired_gap_m": np.full(3, 30.0),
        "control_update": np.zeros(2, dtype=np.bool_),
        **{f"{kind}_{quantity}_body_{unit}": np.zeros((2, 2, 3))
           for kind in ("requested", "applied")
           for quantity, unit in (("forces", "n"), ("torques", "nm"))},
    }
    manifest = {
        "schema_version": 3, "scenario": "visual-nominal", "visual_case": "nominal",
        "body_order": ["target", "chaser"], "dt_ns": 100_000_000,
        "num_steps": 2, "max_steps": 6000, "camera_period_ticks": 10,
        "delivery_delay_ticks": 2, "mu_m3_s2": 3.986004418e14,
        "earth_radius_m": 6_378_137.0, "masses_kg": [200.0, 100.0],
        "inertias_kg_m2": [[100.0, 120.0, 140.0], [20.0, 25.0, 30.0]],
        "force_limits_n": [2.0, 2.0], "torque_limits_nm": [0.5, 0.5],
        "docking_ports_body_m": [[0.0, -1.0, 0.0], [0.0, 0.5, 0.0]],
        "camera": {
            "width": 8, "height": 8,
            "K": [[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]],
            "mount_position_body_m": [0.0, 0.0, 0.0], "rotation_bc": np.eye(3).tolist(),
        },
        "calibration_file": "calibration.json",
        "calibration_sha256": sha256_file(directory / "calibration.json"),
        **{f"initial_{field}": trajectory[field][0].tolist() for field in (
            "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps"
        )},
    }
    pose = {
        "capture_time_ns": 0, "delivery_time_ns": 200_000_000, "valid": True,
        "rotation_ct": np.eye(3).tolist(), "translation_ct": [0.0, 0.0, 30.0],
        "image_corners": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "reason": "", "marker_ids": [0, 1], "inlier_count": 8,
    }
    observations = [{
        "capture_id": 0, "capture_time_ns": 0,
        "scheduled_delivery_time_ns": 200_000_000, "actual_delivery_time_ns": 200_000_000,
        "image_path": "images/000000.png", "image_sha256": sha256_file(image),
        "fault": None, "pose_estimate": pose, "navigation_accepted": True,
    }]
    estimates = [{
        "time_ns": tick * 100_000_000, "target": None,
        "visual_update": None, "used_for_control": False, "onboard_metrics": None,
    } for tick in range(3)]
    estimates[-1].update({
        "visual_update": True,
        "target": {
            "time_ns": 200_000_000, "capture_time_ns": 0, "health": "fresh",
            "position_m": positions[-1, 0].tolist(),
            "velocity_mps": velocity[-1, 0].tolist(),
            "quaternion_xyzw": quaternion[-1, 0].tolist(),
            "angular_velocity_inertial_radps": [0.0, 0.0, 0.0],
            "covariance": (np.eye(12) * 1e-4).tolist(),
        },
    })
    events = [
        {"tick": 0, "time_ns": 0, "phase": 4,
         "kind": "supervisor_transition", "reason": "acquisition_started"},
        {"tick": 2, "time_ns": 200_000_000, "phase": 3,
         "kind": "outcome", "reason": "time_limit"},
    ]
    write_json(directory / "manifest.json", manifest)
    np.savez(directory / "trajectory.npz", **trajectory)
    write_json(directory / "events.json", events)
    write_jsonl(directory / "observations.jsonl", observations)
    write_jsonl(directory / "estimates.jsonl", estimates)
    return manifest, trajectory, observations, estimates


class VisualRecords(unittest.TestCase):
    def test_saved_replay_does_not_execute_runtime_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            expected, trajectory, _, _ = write_fixture(directory)
            manifest, loaded = load_run(directory)
            self.assertEqual(manifest, expected)
            np.testing.assert_array_equal(loaded["positions_m"], trajectory["positions_m"])
            self.assertEqual(load_events(directory, manifest)[0]["phase"], 4)
            with (
                patch("rpod.RigidBodyEngine", side_effect=AssertionError("physics ran")),
                patch("rpod.navigation.VisualNavigation", side_effect=AssertionError("filter ran")),
                patch("rpod.perception.estimate_pose", side_effect=AssertionError("detector ran")),
                patch("rpod.rendering.BlenderRenderer", side_effect=AssertionError("renderer ran")),
                patch("rpod.visualization.open_replay") as viewer,
            ):
                output = replay_run(directory, save_only=True)
            viewer.assert_not_called()
            self.assertGreater(output.stat().st_size, 0)

    def test_corrupt_states_intervals_images_paths_and_future_estimates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for corruption in ("quaternion", "interval", "image", "path", "future", "delivery"):
                with self.subTest(corruption=corruption):
                    directory = Path(temporary) / corruption
                    _, trajectory, observations, estimates = write_fixture(directory)
                    if corruption == "quaternion":
                        trajectory["quaternions_xyzw"][1, 0] *= 2
                        np.savez(directory / "trajectory.npz", **trajectory)
                    elif corruption == "interval":
                        trajectory["applied_forces_body_n"] = np.zeros((1, 2, 3))
                        np.savez(directory / "trajectory.npz", **trajectory)
                    elif corruption == "image":
                        observations[0]["image_sha256"] = "0" * 64
                        write_jsonl(directory / "observations.jsonl", observations)
                    elif corruption == "path":
                        escape = Path(temporary) / "escape.png"
                        write_rgb(escape, np.zeros((8, 8, 3), dtype=np.uint8))
                        observations[0]["image_path"] = "../escape.png"
                        observations[0]["image_sha256"] = sha256_file(escape)
                        write_jsonl(directory / "observations.jsonl", observations)
                    elif corruption == "future":
                        estimates[1]["target"] = copy.deepcopy(estimates[2]["target"])
                        estimates[1]["target"]["time_ns"] = 100_000_000
                        write_jsonl(directory / "estimates.jsonl", estimates)
                    else:
                        observations[0]["actual_delivery_time_ns"] -= 1
                        write_jsonl(directory / "observations.jsonl", observations)
                    with self.assertRaises(ValueError):
                        load_visual_run(directory)

    def test_dataset_estimate_geometry_is_restored_after_invalid_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "truth").mkdir()
            image = directory / "image.png"
            write_rgb(image, np.zeros((8, 8, 3), dtype=np.uint8))
            write_json(directory / "dataset.json", {
                "artifact_type": "perception_dataset", "schema_version": 1, "frame_count": 3,
            })
            rows, labels = [], []
            for index, valid in enumerate((True, False, True)):
                rows.append({
                    "capture_id": index, "image": "image.png", "image_sha256": sha256_file(image),
                    "pose": {"valid": valid, "reason": "", "marker_ids": [0, 1],
                             "rotation_ct": np.eye(3).tolist(), "translation_ct": [0, 0, 10],
                             "image_corners": None},
                })
                labels.append({
                    "capture_id": index, "rotation_ct": np.eye(3).tolist(),
                    "translation_ct": [0, 0, 10], "translation_error_m": 0.0,
                    "rotation_error_deg": 0.0,
                })
            write_jsonl(directory / "observations.jsonl", rows)
            write_jsonl(directory / "truth" / "poses.jsonl", labels)
            with patch("rpod.vision_replay.rr.RecordingStream") as stream:
                export_dataset(directory)
            calls = [call for call in stream.return_value.log.call_args_list
                     if call.args[0] == "/pose/estimate"]
            self.assertEqual(len(calls), 3)
            self.assertIsInstance(calls[0].args[1], rr.Boxes3D)
            self.assertIsInstance(calls[1].args[1], rr.Clear)
            self.assertIsInstance(calls[2].args[1], rr.Boxes3D)


if __name__ == "__main__":
    unittest.main()
