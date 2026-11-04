"""Fast environment-interface and M5 suite checks with a stub renderer and actual saved pixels."""

from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict, fields
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod import m5_suite
from rpod.approach import STATE_FIELDS, make_approach_scenario
from rpod.camera import CameraModel
from rpod.m5_suite import run_suite, suite_report, suite_specs
from rpod.observations import ChaserCommand, Delivery, EnvObservation, LidarDelivery
from rpod.policy import initial_parameters, save_policy
from rpod.vision_records import json_value, read_jsonl
from rpod.visual_agent import VisualAgent
from rpod.visual_control import MissionSpec
from rpod.visual_env import VisualEnv
from rpod.visual_recording import _load_visual, load_visual_run
from rpod.visual_run import (initial_state, run_episode, run_visual, sunlight_direction,
                             visual_initial_state)


FIXTURES = Path(__file__).parent / "fixtures"
CENTER = FIXTURES / "vision" / "center.png"
TRACE = FIXTURES / "m4_trace.json"
VOLATILE_SUMMARY = {"wall_seconds", "renderer_performance", "performance"}
ORIGINAL_EPISODE_INITIAL_STATE = m5_suite.episode_initial_state


class StubRenderer:
    """Writes one committed actual Blender image for every capture."""

    provenance = {}

    def __init__(self, output_dir, blender_path=None, camera=None, scene=None):
        self.performance = {"startup_seconds": 0.0, "render_seconds": [],
                            "request_seconds": [], "peak_rss_bytes": None, "rss_scope": "stub"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def render(self, rotation_ct, translation_ct, output_path, light_direction_c=None, earth=None):
        shutil.copyfile(CENTER, output_path)
        return {}


def write_calibration(directory, scene=None):
    provenance = json.loads((FIXTURES / "vision" / "provenance.json").read_text(encoding="utf-8"))
    path = Path(directory) / "calibration.json"
    path.write_text(json.dumps({**provenance["navigation_calibration"], "qualification_passed": True,
                                **({"scene": scene} if scene else {})}), encoding="utf-8")
    return path


def handoff():
    """A renderer-free stand-in for the recorded M2 handoff."""
    scenario = make_approach_scenario("approach")
    states = {f"initial_{field}": np.asarray(scenario[f"initial_{field}"], dtype=np.float64).tolist()
              for field in STATE_FIELDS}
    return {**states, "handoff_source": "test", "handoff_source_tick": 0,
            "handoff_source_time_ns": 0}


def near_handoff():
    """The test handoff moved in to the 31.5 m M2 terminal COM range, co-moving."""
    states = handoff()
    positions = np.asarray(states["initial_positions_m"])
    velocities = np.asarray(states["initial_velocities_mps"])
    positions[1] = positions[0] + (positions[1] - positions[0]) * 0.315
    velocities[1] = velocities[0]
    return {**states, "initial_positions_m": positions.tolist(),
            "initial_velocities_mps": velocities.tolist()}


def trace_initial(seed=7):
    """A short M4 case: missing capture, black frame, then a two-capture navigation loss."""
    with patch("rpod.visual_run._handoff", return_value=handoff()):
        initial = visual_initial_state("abort", seed)
    initial.update(max_steps=200, loss_schedule={"missing_capture_s": [4, 12, 13],
                                                 "black_capture_s": [8]})
    return initial


def env_initial(seed=7):
    """The trace manifest plus the mission and lighting that run_episode adds."""
    initial = trace_initial(seed)
    initial.update(mission=json_value(asdict(MissionSpec())),
                   sunlight_direction_inertial=[0.0, 1.0, 0.0])
    return initial


def run_trace(output, calibration):
    with (
        patch("rpod.visual_run.visual_initial_state", return_value=trace_initial()),
        patch("rpod.rendering.BlenderRenderer", StubRenderer),
    ):
        return run_visual("abort", 7, output, calibration)


def run_env(output, seed=7, ticks=25):
    """Zero commands for a few ticks; returns the environment and its observations."""
    env = VisualEnv(CameraModel())
    with patch("rpod.rendering.BlenderRenderer", StubRenderer), env:
        stream = [env.reset(env_initial(seed), output)]
        for _ in range(ticks):
            stream.append(env.step(ChaserCommand(np.zeros(3), np.zeros(3))))
    return env, stream


@contextmanager
def no_environment_calls():
    with ExitStack() as stack:
        for name in ("rpod.approach.RigidBodyEngine", "rpod.rendering.BlenderRenderer",
                     "rpod.visual_run._handoff", "rpod.visual_env.make_engine"):
            stack.enter_context(patch(name, side_effect=AssertionError(f"environment access: {name}")))
        yield


def trace_digest(directory):
    """Hash every saved record except wall-clock timing and renderer metadata."""
    directory = Path(directory)

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        trajectory = {key: [str(data[key].dtype), list(data[key].shape),
                            hashlib.sha256(data[key].tobytes()).hexdigest()]
                      for key in sorted(data.files)}
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return {
        "outcome": summary["outcome"],
        "trajectory": trajectory,
        "events": digest(json.loads((directory / "events.json").read_text(encoding="utf-8"))),
        "observations": digest([
            {key: value for key, value in row.items() if key != "render_metadata"}
            for row in read_jsonl(directory / "observations.jsonl")
        ]),
        "estimates": digest(read_jsonl(directory / "estimates.jsonl")),
        "summary": digest({key: value for key, value in summary.items()
                           if key not in VOLATILE_SUMMARY}),
    }


class VisualEnvironment(unittest.TestCase):
    def test_m4_trace_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_trace(output, write_calibration(temporary))
            self.assertEqual(trace_digest(output),
                             json.loads(TRACE.read_text(encoding="utf-8")))

    def test_reset_is_deterministic_for_a_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = [run_env(Path(temporary) / name, seed)
                    for name, seed in (("a", 7), ("b", 7), ("c", 8))]
        (first, first_stream), (repeat, repeat_stream), (other, _) = runs
        for field, values in first.states.items():
            np.testing.assert_array_equal(values, repeat.states[field])
        strip = lambda env: [{key: value for key, value in row.items() if key != "render_metadata"}
                             for row in env.observations]
        self.assertEqual(strip(first), strip(repeat))
        poses = [[delivery.pose.translation_ct for observation in stream
                  for delivery in observation.deliveries] for stream in (first_stream, repeat_stream)]
        self.assertEqual(len(poses[0]), 3)
        np.testing.assert_array_equal(poses[0], poses[1])
        self.assertGreater(np.linalg.norm(np.asarray(first.states["positions_m"][0][0])
                                          - other.states["positions_m"][0][0]), 1e-3)

    def test_commands_are_validated_saturated_and_claims_checked(self):
        env = VisualEnv(CameraModel())
        with (tempfile.TemporaryDirectory() as temporary,
              patch("rpod.rendering.BlenderRenderer", StubRenderer), env):
            env.reset(env_initial(), Path(temporary) / "run")
            for command in (ChaserCommand(np.full(3, np.nan), np.zeros(3)),
                            ChaserCommand(np.zeros(2), np.zeros(3)),
                            ChaserCommand(np.zeros(3), np.zeros(3), "docked")):
                with self.assertRaises(ValueError):
                    env.step(command)
            self.assertEqual(len(env.states["tick"]), 1)
            self.assertIsNotNone(env.step(ChaserCommand([10.0, 0, 0], [0, 0, 3.0])))
            np.testing.assert_array_equal(env.commands["requested_forces_body_n"][0][1], [10, 0, 0])
            self.assertAlmostEqual(np.linalg.norm(env.commands["applied_forces_body_n"][0][1]), 2.0)
            self.assertAlmostEqual(np.linalg.norm(env.commands["applied_torques_body_nm"][0][1]), 0.5)
            self.assertIsNone(env.step(ChaserCommand(np.zeros(3), np.zeros(3), "success")))
            self.assertEqual((env.outcome, env.failure_reason),
                             ("false_completion", "physical_dwell_failed"))
            self.assertEqual(len(env.states["tick"]), 2)
            with self.assertRaises(RuntimeError):
                env.step(ChaserCommand(np.zeros(3), np.zeros(3)))

    def test_client_sees_only_declared_observations(self):
        self.assertEqual([field.name for field in fields(EnvObservation)],
                         ["time_ns", "own", "deliveries", "lidar"])
        self.assertEqual([field.name for field in fields(Delivery)],
                         ["image", "pose", "own_at_capture"])
        self.assertEqual([field.name for field in fields(LidarDelivery)],
                         ["scan", "pose", "own_at_capture"])
        with tempfile.TemporaryDirectory() as temporary:
            _, stream = run_env(Path(temporary) / "run")
            calibration = json.loads(write_calibration(temporary).read_text(encoding="utf-8"))
        agent = VisualAgent(MissionSpec(), CameraModel(), calibration)
        with no_environment_calls():
            commands = [agent.act(observation) for observation in stream]
        self.assertEqual(agent.phase[-1], 1)
        self.assertTrue(all(command.claim is None for command in commands))

    def test_lidar_alone_acquires_and_tracks_through_a_camera_blackout(self):
        with patch("rpod.visual_run._handoff", return_value=near_handoff()):
            initial = initial_state("visual-lidar-check", 7, lidar_noise_seed=[7, 0, 4],
                                    loss_schedule={"missing_capture_s": [],
                                                   "black_capture_s": list(range(41))})
        initial["max_steps"] = 400
        with (tempfile.TemporaryDirectory() as temporary,
              patch("rpod.rendering.BlenderRenderer", StubRenderer)):
            path, output = write_calibration(temporary), Path(temporary) / "run"
            calibration = json.loads(path.read_text(encoding="utf-8"))
            summary = run_episode(initial, MissionSpec(), calibration, path, output)
            _, trajectory, events, _, estimates, lidar = _load_visual(output)
        self.assertEqual(summary["accepted_observations"], 0)
        self.assertEqual(summary["lidar_accepted_observations"], len(lidar))
        reasons = [event["reason"] for event in events]
        self.assertIn("vision_acquired", reasons)
        self.assertNotIn("visual_brake", reasons)
        self.assertEqual(summary["outcome"], "time_limit")
        errors = [np.linalg.norm(np.asarray(row["target"]["position_m"])
                                 - trajectory["positions_m"][tick, 0])
                  for tick, row in enumerate(estimates) if tick >= 50]
        self.assertLess(max(errors), 0.1)


class M5Suite(unittest.TestCase):
    def test_suite_is_frozen_and_operational_cases_stay_in_the_qualified_envelope(self):
        specs = suite_specs()
        self.assertEqual(specs, suite_specs())
        self.assertEqual(json.loads(json.dumps(specs)), specs)
        self.assertNotEqual(specs, suite_specs(seed=6))
        self.assertEqual(Counter(spec["category"] for spec in specs),
                         {"operational": 28, "fault": 6, "stress": 6})
        seeds = [spec["init_seed"] for spec in specs]
        self.assertEqual(len(set(seeds)), 40)
        self.assertGreaterEqual(min(seeds), 1000)
        for spec in specs:
            conditions = spec["conditions"]
            if spec["category"] == "operational":
                self.assertEqual(spec["expected_outcome"], "success")
                self.assertLessEqual(abs(conditions["sun_angle_deg"]), 30)
                self.assertNotIn("loss_schedule", conditions)
                degradation = conditions["image_degradation"]
                self.assertTrue(degradation is None
                                or (degradation["noise_dn"], degradation["blur_px"]) == (2.0, 0.5))
                disturbance = conditions["chaser_disturbance"]
                self.assertTrue(np.all(np.abs(disturbance["offset_lvlh_m"]) <= [0.5, 0.5, 0.25]))
                self.assertTrue(np.all(np.abs(disturbance["delta_v_lvlh_mps"]) <= 0.01))
                self.assertLessEqual(disturbance["rotation_deg"], 3)
            elif spec["category"] == "fault":
                schedule = conditions.get("loss_schedule", {})
                lost = sorted(schedule.get("missing_capture_s", [])
                              + schedule.get("black_capture_s", []))
                # One lost capture ages the estimate to 2.1 s; two consecutive reach 3.1 s > 3 s.
                expected = {"abort": "retreated", "dropped-frame": "success",
                            "extended-loss": "navigation_lost"}[spec["kind"]]
                self.assertEqual(spec["expected_outcome"], expected)
                self.assertEqual(len(lost) >= 2, expected == "navigation_lost")
                self.assertEqual(lost, list(range(lost[0], lost[0] + len(lost))) if lost else [])
                self.assertEqual(spec["abort_gap_m"] is not None, spec["kind"] == "abort")

    def test_lidar_suite_changes_only_noise_streams_and_camera_loss_expectations(self):
        for camera, lidar in zip(suite_specs(), suite_specs(lidar=True)):
            self.assertEqual(lidar["streams"].pop("lidar_noise"), [5, camera["index"], 4])
            if camera["kind"] == "extended-loss":
                self.assertEqual(lidar["expected_outcome"], "success")
                lidar["expected_outcome"] = camera["expected_outcome"]
            self.assertEqual(lidar, camera)

    def test_realistic_suite_changes_only_the_scene_and_keeps_the_sun_up_for_every_episode(self):
        for simple, realistic in zip(suite_specs(), suite_specs(scene="realistic")):
            self.assertEqual(realistic.pop("scene"), "realistic")
            self.assertEqual(realistic, simple)
        with patch("rpod.visual_run._handoff", return_value=handoff()):
            for spec in suite_specs(scene="realistic"):
                initial = m5_suite.episode_initial_state(spec)
                chaser = np.asarray(initial["initial_positions_m"][1])
                radius = np.linalg.norm(chaser)
                # The Sun is opposite the incident ray. Its elevation falls at most at the
                # orbital rate; its whole disc must stay above Earth's limb until the step limit.
                elevation = np.degrees(np.arcsin(-sunlight_direction(initial) @ chaser / radius))
                dip = np.degrees(np.arccos(initial["earth_radius_m"] / radius))
                duration_s = initial["max_steps"] * initial["dt_ns"] / 1e9
                setting = np.degrees(np.sqrt(initial["mu_m3_s2"] / radius**3)) * duration_s
                self.assertGreater(elevation - setting, -dip + 0.265, spec["id"])

    def test_resume_refuses_a_changed_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            calibration, output = write_calibration(temporary), Path(temporary) / "suite"
            with patch("rpod.m5_suite.suite_specs", return_value=[]):
                self.assertFalse(run_suite(output, calibration)["passed"])
            with patch("rpod.m5_suite.run_episode", side_effect=AssertionError("episode ran")):
                with self.assertRaisesRegex(ValueError, "frozen with different"):
                    run_suite(output, calibration)
                with self.assertRaisesRegex(ValueError, "qualified on the simple scene"):
                    run_suite(Path(temporary) / "realistic", calibration, scene="realistic")

    def test_report_is_recomputed_from_saved_records(self):
        def short(spec):
            initial = ORIGINAL_EPISODE_INITIAL_STATE(spec)
            initial["max_steps"] = 40
            return initial

        cases = ((False, "simple", "reference"), (True, "simple", "reference"),
                 (False, "realistic", "reference"), (False, "realistic", "learned"))
        for lidar, scene, controller in cases:
            specs = suite_specs(lidar=lidar, scene=scene)
            chosen = [specs[0], next(spec for spec in specs if spec["kind"] == "extended-loss")]
            with (
                self.subTest(lidar=lidar, scene=scene, controller=controller),
                tempfile.TemporaryDirectory() as temporary,
                redirect_stdout(io.StringIO()),
                patch("rpod.m5_suite.suite_specs", return_value=chosen),
                patch("rpod.m5_suite.episode_initial_state", side_effect=short),
                patch("rpod.visual_run._handoff", return_value=handoff()),
                patch("rpod.rendering.BlenderRenderer", StubRenderer),
            ):
                output = Path(temporary) / "suite"
                calibration = write_calibration(temporary, scene if scene != "simple" else None)
                policy = Path(temporary) / "policy.npy"
                save_policy(policy, initial_parameters(3), {})
                report = run_suite(output, calibration, lidar, scene, controller, policy)
                self.assertEqual(suite_report(output), report)
                self.assertTrue(report["gates"]["records_valid_and_uniform"])
                self.assertFalse(report["passed"])
                self.assertEqual(report.get("scene", "simple"), scene)
                self.assertEqual("safety_filter_interventions" in report, controller == "learned")
                self.assertEqual("lidar_pose_by_range" in report["categories"]["operational"],
                                 lidar)
                self.assertEqual([row["id"] for row in report["failed_episodes"]],
                                 [spec["id"] for spec in chosen])
                for row in report["failed_episodes"]:
                    manifest, _ = load_visual_run(output / row["path"])
                    self.assertEqual(manifest["m5_episode"]["id"], row["id"])


if __name__ == "__main__":
    unittest.main()
