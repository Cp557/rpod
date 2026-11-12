"""The vectorized fleet must reproduce the scalar reference law, supervisor and oracle."""

import unittest
from unittest.mock import patch

import numpy as np

from rpod import fleet
from rpod.approach import commanded_wrench, gap_profile, state_metrics
from rpod.frames import quaternion_matrices
from rpod.observations import ChaserInputs, NavigationEstimate, OwnNavigation
from rpod.visual_control import MissionSpec, VisualController
from rpod.visual_run import initial_state, paired_truth_baseline

from helpers import near_handoff, random_states, snapshot


class FleetMatchesScalar(unittest.TestCase):
    def test_metrics_profile_and_wrench_match_the_scalar_law(self):
        mission = MissionSpec()
        config = mission.controller_config()
        states = random_states(64, 1)
        rng = np.random.default_rng(2)
        start, elapsed = rng.uniform(5, 30, 64), rng.uniform(0, 250, 64)
        reference = fleet.gap_profile(start, 5.0, elapsed, 0.15, 0.005)
        m = fleet.metrics(states, mission.docking_ports_body_m)
        force, target_torque, chaser_torque = fleet.wrench(states, mission, m, reference)
        for index, state in enumerate(states):
            scalar_reference = gap_profile(start[index], 5.0, elapsed[index], 0.15, 0.005)
            np.testing.assert_allclose([value[index] for value in reference], scalar_reference,
                                       rtol=1e-12, atol=1e-12)
            expected = state_metrics(snapshot(state), config)
            for key in ("rho", "rho_dot", "port_r", "port_v", "gap", "alignment_deg",
                        "relative_rate"):
                np.testing.assert_allclose(m[key][index], expected[key], rtol=1e-12, atol=1e-12)
            forces, torques = commanded_wrench(snapshot(state), config, expected, scalar_reference)
            np.testing.assert_allclose(force[index], forces[1], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(target_torque[index], torques[0], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(chaser_torque[index], torques[1], rtol=1e-11, atol=1e-12)

    def test_controller_matches_visual_controller_on_scripted_inputs(self):
        mission = MissionSpec(abort_gap_m=None)
        count, ticks = 6, 400
        rng = np.random.default_rng(3)
        start = random_states(count, 4, max_angle_deg=2.0)
        start[3, 1, 1] = start[3, 0, 1] - 27.5  # starts inside its 25 m abort gap after 1.5 s
        start[4, 1, 6:10] = [np.sin(np.deg2rad(12.5)), 0, 0, np.cos(np.deg2rad(12.5))]
        controllers = [VisualController(mission) for _ in range(count)]
        vector = fleet.FleetController(mission, count, abort_gaps_m=[np.nan, 20, np.nan, 25,
                                                                     np.nan, np.nan])
        for index, gap in enumerate([None, 20, None, 25, None, None]):
            controllers[index] = VisualController(MissionSpec(abort_gap_m=gap))
        active, visited = np.ones(count, bool), set()
        states = start.copy()
        for tick in range(ticks):
            time_ns = tick * mission.dt_ns
            # Move the chaser in along -Y; script deliveries, rejections and dropouts.
            states[:, 1, 1] = start[:, 1, 1] + 0.12 * tick * 0.1
            visual = np.where(rng.random(count) < 0.12, fleet.FAILED_UPDATE,
                              fleet.ACCEPTED_UPDATE)
            visual[tick % 10 != 2] = fleet.NO_UPDATE
            available = np.full(count, tick >= 2)
            if 150 <= tick < 200:  # episode 1 loses navigation
                available[1], visual[1] = False, fleet.NO_UPDATE
            capture = np.full(count, (tick - tick % 10 - 8) * mission.dt_ns)
            fresh = rng.random(count) > 0.02
            target = states[:, 0].copy()
            estimates = fleet.Estimates(target, available, capture, fresh, visual)
            inertial_rates = np.einsum("nij,nj->ni", quaternion_matrices(target[:, 6:10]),
                                       target[:, 10:])
            claims = vector.update(time_ns, states[:, 1], estimates, active)
            for index, controller in enumerate(controllers):
                if not active[index]:
                    continue
                own = OwnNavigation(time_ns, *(states[index, 1, slice_] for slice_ in (
                    slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13))))
                estimate = (NavigationEstimate(time_ns, target[index, :3], target[index, 3:6],
                                               target[index, 6:10], inertial_rates[index],
                                               np.zeros((12, 12)), int(capture[index]),
                                               "fresh" if fresh[index] else "predicted")
                            if available[index] else None)
                update = {fleet.NO_UPDATE: None, fleet.FAILED_UPDATE: False,
                          fleet.ACCEPTED_UPDATE: True}[int(visual[index])]
                decision = controller.update(ChaserInputs(time_ns, own, estimate, update))
                self.assertEqual(vector.phase[index], decision.phase, (tick, index))
                self.assertEqual(fleet.OUTCOMES[claims[index]], decision.outcome or "")
                self.assertEqual(fleet.ABORT_REASONS[vector.abort_reason[index]],
                                 decision.abort_reason or "")
                np.testing.assert_allclose(vector.force[index], decision.force_body_n,
                                           rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(vector.torque[index], decision.torque_body_nm,
                                           rtol=1e-10, atol=1e-12)
            visited.update(vector.phase[active].tolist())
            active &= claims == 0
        self.assertLessEqual({1, 2, 5, 6}, visited)
        self.assertLessEqual({"estimated_alignment_limit", "estimated_gap_abort_request"},
                             {fleet.ABORT_REASONS[code] for code in vector.abort_reason})

    def test_truth_fleet_matches_the_paired_oracle(self):
        with patch("rpod.visual_run._handoff", return_value=near_handoff()):
            initials = [initial_state("visual-m5-operational", seed) for seed in (1001, 1002)]
        missions = [MissionSpec(), MissionSpec(abort_gap_m=20.0)]
        expected = [paired_truth_baseline(initial, mission)
                    for initial, mission in zip(initials, missions)]
        controller = fleet.FleetController(MissionSpec(), 2, abort_gaps_m=[np.nan, 20.0])
        result = fleet.run_fleet(initials, MissionSpec(),
                                 fleet.TruthNavigation(2, initials[0]["dt_ns"]), controller)
        for index, oracle in enumerate(expected):
            self.assertEqual(fleet.OUTCOMES[result["outcome"][index]], oracle["outcome"])
            self.assertEqual(result["end_tick"][index] * initials[0]["dt_ns"],
                             oracle["elapsed_time_ns"])
        self.assertEqual([fleet.OUTCOMES[code] for code in result["outcome"]],
                         ["success", "retreated"])


if __name__ == "__main__":
    unittest.main()
