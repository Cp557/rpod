"""The learned approach law behind the safety filter, for one episode or a fleet.

Both controllers keep the reference supervisor and attitude law. In approach (phase 1) the
policy sets the translational acceleration, the safety filter can replace it with the
reference law's hold at the current gap, and completion does not wait for the reference gap
profile. Braking, retreat and acquisition use the reference law unchanged.
"""

import numpy as np

from .approach import commanded_wrench
from .fleet import FleetController, relative_force, wrench
from .policy import OBSERVATION_SIZE, act, observation
from .safety_filter import needs_backup
from .visual_control import VisualController

# Files whose behavior the learned law depends on, hashed into suite provenance.
IMPLEMENTATION_FILES = ("policy.py", "safety_filter.py", "learned_control.py", "fleet.py")


def _states(pair):
    return np.concatenate([np.asarray(pair.positions_m), np.asarray(pair.velocities_mps),
                           np.asarray(pair.quaternions_xyzw),
                           np.asarray(pair.angular_velocities_body_radps)], axis=-1)[None]


class LearnedController(VisualController):
    """`VisualController` with the learned approach law; counts safety-filter interventions."""

    def __init__(self, mission, parameters):
        super().__init__(mission)
        self.parameters = parameters
        self.interventions = 0

    def _reference_complete(self, reference):
        return self.phase == 1 or super()._reference_complete(reference)

    def _wrench(self, pair, metrics, reference, inputs):
        force, torque = super()._wrench(pair, metrics, reference, inputs)
        if self.phase != 1:
            return force, torque
        m = {key: np.asarray(value)[None] for key, value in metrics.items()}
        age = np.array([(inputs.time_ns - inputs.target.capture_time_ns) / 1e9])
        acceleration = act(self.parameters, observation(m, self.goal, age))
        if needs_backup(m, acceleration, self.goal)[0]:
            self.interventions += 1
            hold, _ = commanded_wrench(pair, self.mission.controller_config(), metrics,
                                       (metrics["gap"], 0.0, 0.0))
            return hold[1], torque
        return relative_force(_states(pair), self.mission, m, acceleration)[0], torque


class LearnedFleetController(FleetController):
    """`FleetController` with the learned approach law.

    `parameters` is one policy vector, or (G, P) vectors for G equal, consecutive groups of
    episodes, as evolution strategies evaluate candidates.
    """

    def __init__(self, mission, count, parameters, abort_gaps_m=None):
        super().__init__(mission, count, abort_gaps_m)
        self.parameters = np.asarray(parameters)
        self.interventions = np.zeros(count, dtype=np.int64)

    def _act(self, observations, rows):
        if self.parameters.ndim == 1:
            return act(self.parameters, observations)
        full = np.zeros((self.count, OBSERVATION_SIZE))
        full[rows] = observations
        grouped = full.reshape(len(self.parameters), -1, OBSERVATION_SIZE)
        return act(self.parameters, grouped).reshape(self.count, 3)[rows]

    def _reference_complete(self, reference):
        return (self.phase == 1) | super()._reference_complete(reference)

    def _wrench(self, pair, m, reference, rows, time_ns, estimates):
        force, torque = super()._wrench(pair, m, reference, rows, time_ns, estimates)
        approach = self.phase[rows] == 1
        if not approach.any():
            return force, torque
        subset = np.flatnonzero(rows)[approach]
        sub = {key: value[subset] for key, value in m.items()}
        age = (time_ns - estimates.capture_time_ns[subset]) / 1e9
        acceleration = self._act(observation(sub, self.goal[subset], age), subset)
        backup = needs_backup(sub, acceleration, self.goal[subset])
        zeros = np.zeros(len(subset))
        hold = wrench(pair[subset], self.mission, sub, (sub["gap"], zeros, zeros))[0]
        force[approach] = np.where(backup[:, None], hold,
                                   relative_force(pair[subset], self.mission, sub, acceleration))
        self.interventions[subset[backup]] += 1
        return force, torque
