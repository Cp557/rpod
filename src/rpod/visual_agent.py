"""Reference visual-environment client: the frozen M4 filter and supervisor, optionally
fusing lidar poses."""

from .lidar import LidarNavigation
from .navigation import VisualNavigation
from .observations import ChaserCommand, ChaserInputs
from .vision_records import json_value
from .visual_control import VisualController

# A failed delivery brakes unless another sensor's capture this recent was accepted.
_SUPPORT_NS = 1_000_000_000


class VisualAgent:
    """Consumes only EnvObservation and keeps the onboard telemetry schema 3 records.

    With a lidar model, lidar poses update the same filter as camera poses.
    """

    def __init__(self, mission, camera, calibration, lidar=None):
        self.navigator = (VisualNavigation(camera=camera, calibration=calibration)
                          if lidar is None else LidarNavigation(camera, calibration, lidar))
        self.controller = VisualController(mission)
        self.decision = None
        self.phase, self.desired_gap_m, self.control_update = [], [], []
        self.estimates, self.events, self.accepted, self.lidar_accepted = [], [], {}, {}
        self.latest_capture_ns = None

    def _observe(self, delivery):
        accepted = bool(self.navigator.observe(delivery.pose, delivery.own_at_capture))
        if accepted:
            capture = delivery.own_at_capture.time_ns
            self.latest_capture_ns = max(capture, self.latest_capture_ns or 0)
        return accepted

    def act(self, observation) -> ChaserCommand:
        self.phase.append(self.controller.phase)
        self.desired_gap_m.append(self.controller.profile_start)
        results = []
        for delivery in observation.deliveries:
            self.accepted[delivery.image.capture_id] = self._observe(delivery)
            results.append(self.accepted[delivery.image.capture_id])
        for delivery in observation.lidar:
            self.lidar_accepted[delivery.scan.capture_id] = self._observe(delivery)
            results.append(self.lidar_accepted[delivery.scan.capture_id])
        visual_update = None
        if any(results):
            visual_update = True
        elif results and (self.latest_capture_ns is None
                          or observation.time_ns - self.latest_capture_ns > _SUPPORT_NS):
            visual_update = False
        estimate = self.navigator.predict(observation.time_ns)
        decision = self.controller.update(
            ChaserInputs(observation.time_ns, observation.own, estimate, visual_update)
        )
        if decision.reacquired:
            self.navigator.mark_reacquired()
        self.decision = decision
        self.events.extend(decision.events)
        self.phase[-1], self.desired_gap_m[-1] = decision.phase, decision.desired_gap_m
        self.control_update.append(decision.control_update)
        self.estimates.append({"time_ns": observation.time_ns,
                               "target": json_value(estimate),
                               "visual_update": visual_update,
                               "used_for_control": decision.control_update,
                               "onboard_metrics": json_value(decision.metrics)})
        return ChaserCommand(decision.force_body_n, decision.torque_body_nm, decision.outcome)
