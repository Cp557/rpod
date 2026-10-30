"""Fast actual-image evidence for the image/navigation/onboard-command boundary."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpod.camera import CameraModel
from rpod.navigation import VisualNavigation
from rpod.observations import ChaserInputs, ImageObservation, OwnNavigation
from rpod.perception import estimate_pose
from rpod.scenarios import MU_M3_S2, ORBIT_RADIUS_M
from rpod.vision_records import read_rgb
from rpod.visual_control import MissionSpec, VisualController


FIXTURES = Path(__file__).parent / "fixtures" / "vision"


def circular_own(time_ns):
    """Closed-form own navigation, independent of any simulated target or engine."""
    rate = math.sqrt(MU_M3_S2 / ORBIT_RADIUS_M**3)
    angle = rate * time_ns / 1e9
    cosine, sine = math.cos(angle), math.sin(angle)
    return OwnNavigation(
        time_ns,
        ORBIT_RADIUS_M * np.array([cosine, sine, 0.0]),
        ORBIT_RADIUS_M * rate * np.array([-sine, cosine, 0.0]),
        np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]),
        np.array([0.0, 0.0, rate]),
    )


@contextmanager
def no_environment_calls():
    with ExitStack() as stack:
        for name in (
            "rpod.approach.RigidBodyEngine",
            "rpod.recording.OrbitEngine",
            "rpod.rendering.BlenderRenderer",
            "rpod.visual_run._handoff",
        ):
            stack.enter_context(patch(name, side_effect=AssertionError(f"unexpected environment access: {name}")))
        yield


class VisualImageEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))

    def fixture(self, name):
        record = next(row for row in self.provenance["fixtures"] if row["file"] == name)
        content = (FIXTURES / name).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])
        return read_rgb(FIXTURES / name)

    def allowed_stream(self, pixels):
        camera = CameraModel()
        np.testing.assert_array_equal(camera.K, self.provenance["camera"]["K"])
        navigation = VisualNavigation(camera, self.provenance["navigation_calibration"])
        controller = VisualController(MissionSpec())
        decisions = []
        for capture_id in range(3):
            capture_ns = capture_id * 1_000_000_000
            delivery_ns = capture_ns + 200_000_000
            observation = ImageObservation(capture_id, capture_ns, delivery_ns, pixels)
            pose = estimate_pose(observation, camera)
            self.assertTrue(pose.valid, pose.reason)
            self.assertTrue(navigation.observe(pose, circular_own(capture_ns)))
            target = navigation.predict(delivery_ns)
            self.assertIsNotNone(target)
            decision = controller.update(ChaserInputs(
                delivery_ns, circular_own(delivery_ns), target, True,
            ))
            decisions.append(decision)
        self.assertEqual(decisions[-1].phase, 1)
        return target.position_m.copy(), np.array([
            np.r_[decision.force_body_n, decision.torque_body_nm] for decision in decisions
        ])

    def test_real_pixels_change_estimates_and_commands_without_a_truth_channel(self):
        center, yaw = self.fixture("center.png"), self.fixture("yaw.png")
        with no_environment_calls(), tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "evaluation-truth.json"
            sidecar.write_text(json.dumps({"target_position_m": [1, 2, 3]}), encoding="utf-8")
            center_position, center_commands = self.allowed_stream(center)
            # Evaluation labels stay outside ImageObservation, own navigation, and ChaserInputs.
            sidecar.write_text(json.dumps({"target_position_m": [-9e9, 8e8, -7e7]}), encoding="utf-8")
            repeated_position, repeated_commands = self.allowed_stream(center)
            yaw_position, yaw_commands = self.allowed_stream(yaw)
        np.testing.assert_array_equal(repeated_position, center_position)
        np.testing.assert_array_equal(repeated_commands, center_commands)
        self.assertGreater(np.linalg.norm(yaw_position - center_position), 0.1)
        self.assertGreater(np.linalg.norm(yaw_commands[-1] - center_commands[-1]), 0.01)

    def test_black_frames_never_acquire_before_the_ten_second_deadline(self):
        camera = CameraModel()
        pixels = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        navigation = VisualNavigation(camera, self.provenance["navigation_calibration"])
        controller = VisualController(MissionSpec())
        with no_environment_calls():
            for tick in range(101):
                now = tick * 100_000_000
                delivered = tick >= 2 and tick % 10 == 2
                accepted = None
                if delivered:
                    capture_ns = now - 200_000_000
                    pose = estimate_pose(ImageObservation(tick // 10, capture_ns, now, pixels), camera)
                    self.assertFalse(pose.valid)
                    accepted = navigation.observe(pose, circular_own(capture_ns))
                    self.assertFalse(accepted)
                estimate = navigation.predict(now)
                self.assertIsNone(estimate)
                decision = controller.update(ChaserInputs(now, circular_own(now), estimate, accepted))
                self.assertFalse(controller.acquired)
        self.assertEqual((decision.phase, decision.outcome), (6, "acquisition_failed"))
        np.testing.assert_array_equal(decision.force_body_n, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
