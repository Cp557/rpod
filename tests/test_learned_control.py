"""The learned approach law: policy evaluation, the safety filter, and scalar-fleet agreement."""

import unittest

import numpy as np

from rpod import fleet
from rpod.learned_control import LearnedController, LearnedFleetController
from rpod.observations import ChaserInputs, NavigationEstimate, OwnNavigation
from rpod.policy import (ACCELERATION_LIMIT_MPS2, OBSERVATION_SIZE, PARAMETER_COUNT, act,
                         initial_parameters)
from rpod.safety_filter import closing_limit, needs_backup
from rpod.visual_control import MissionSpec
from rpod.frames import quaternion_matrices

from helpers import random_states


def port_metrics(port_r, port_v):
    return {"port_r": np.asarray(port_r, dtype=float), "port_v": np.asarray(port_v, dtype=float)}


class LearnedControl(unittest.TestCase):
    def test_grouped_policy_matches_one_policy_per_group(self):
        rng = np.random.default_rng(1)
        parameters = rng.normal(0, 0.3, (3, PARAMETER_COUNT))
        observations = rng.normal(0, 1, (3, 4, OBSERVATION_SIZE))
        grouped = act(parameters, observations)
        for group in range(3):
            np.testing.assert_allclose(grouped[group], act(parameters[group], observations[group]),
                                       rtol=1e-12, atol=1e-15)
        self.assertLessEqual(np.abs(grouped).max(), ACCELERATION_LIMIT_MPS2)

    def test_safety_filter_keeps_braking_room_corridor_and_hold_point(self):
        goal = 5.0
        self.assertAlmostEqual(closing_limit(np.array([30.0]), goal)[0], 0.3)
        self.assertAlmostEqual(closing_limit(np.array([4.75]), goal)[0], 0.01)
        zero = np.zeros((1, 3))
        cases = [  # (port_r, port_v, acceleration, backup expected)
            ([0, -20, 0], [0, 0.1, 0], zero, False),   # slow and centred
            ([0, -6, 0], [0, 0.2, 0], zero, True),     # too fast for the room left
            ([3, -10, 0], [0, 0.05, 0], zero, True),   # outside the corridor
            ([0, -4.8, 0], [0, 0.05, 0], zero, True),  # would pass the hold point margin
            ([0, -20, 0], [0, 0.29, 0], [[0, 0.02, 0]], True),  # accelerating past the cap
        ]
        for port_r, port_v, acceleration, expected in cases:
            with self.subTest(port_r=port_r, port_v=port_v):
                backup = needs_backup(port_metrics([port_r], [port_v]),
                                      np.asarray(acceleration, dtype=float), goal)
                self.assertEqual(bool(backup[0]), expected)

    def test_fleet_matches_the_scalar_learned_controller(self):
        mission = MissionSpec()
        count, ticks = 4, 300
        parameters = initial_parameters(3) * 8  # larger outputs, so policy and filter both act
        start = random_states(count, 9, max_angle_deg=2.0)
        start[:2, 1, 3:6] = start[:2, 0, 3:6] + [0, 0.35, 0]  # closing above the filter's cap
        controllers = [LearnedController(mission, parameters) for _ in range(count)]
        vector = LearnedFleetController(mission, count, parameters)
        active, states = np.ones(count, bool), start.copy()
        for tick in range(ticks):
            time_ns = tick * mission.dt_ns
            states[:, 1, 1] = start[:, 1, 1] + 0.012 * tick
            visual = np.full(count, fleet.ACCEPTED_UPDATE if tick % 10 == 2 else fleet.NO_UPDATE)
            capture = np.full(count, max(tick - tick % 10 - 8, 0) * mission.dt_ns)
            target = states[:, 0].copy()
            estimates = fleet.Estimates(target, np.full(count, tick >= 2), capture,
                                        np.ones(count, bool), visual)
            inertial = np.einsum("nij,nj->ni", quaternion_matrices(target[:, 6:10]),
                                 target[:, 10:])
            claims = vector.update(time_ns, states[:, 1], estimates, active)
            for index, controller in enumerate(controllers):
                if not active[index]:
                    continue
                own = OwnNavigation(time_ns, *(states[index, 1, part] for part in (
                    slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13))))
                estimate = (NavigationEstimate(time_ns, target[index, :3], target[index, 3:6],
                                               target[index, 6:10], inertial[index],
                                               np.zeros((12, 12)), int(capture[index]), "fresh")
                            if tick >= 2 else None)
                update = {fleet.NO_UPDATE: None, fleet.ACCEPTED_UPDATE: True}[int(visual[index])]
                decision = controller.update(ChaserInputs(time_ns, own, estimate, update))
                self.assertEqual(vector.phase[index], decision.phase, (tick, index))
                self.assertEqual(fleet.OUTCOMES[claims[index]], decision.outcome or "")
                np.testing.assert_allclose(vector.force[index], decision.force_body_n,
                                           rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(vector.torque[index], decision.torque_body_nm,
                                           rtol=1e-9, atol=1e-12)
                self.assertEqual(vector.interventions[index], controller.interventions)
            active &= claims == 0
        self.assertTrue((vector.interventions > 0).any() and (vector.phase == 1).any())


if __name__ == "__main__":
    unittest.main()
