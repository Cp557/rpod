"""Dynamics cross-check against Basilisk: shared cases, rpod trajectories and differences.

`basilisk_cases` writes initial states and constant body-frame wrenches. The separate Python
3.13 tool `tools/basilisk/propagate.py` propagates them with Basilisk's RK4 at the same step,
and `compare` reports the largest differences from rpod's `RigidBodyEngine`.
"""

import json
from pathlib import Path

import numpy as np

from . import RigidBodyEngine
from .approach import make_approach_scenario
from .fleet import quaternion_products
from .recording import write_json

# Agreement required between two RK4 integrations of the same equations at the same step.
TOLERANCES = {"position_m": 1e-3, "velocity_mps": 1e-6, "attitude_rad": 1e-6,
              "body_rate_radps": 1e-9}


def cases():
    """One orbit of point-mass gravity, a torque-free tumble, and a constant body wrench."""
    scenario = make_approach_scenario("approach")
    position, velocity = scenario["initial_positions_m"][0], scenario["initial_velocities_mps"][0]
    period = 2 * np.pi * np.sqrt(np.linalg.norm(position) ** 3 / scenario["mu_m3_s2"])
    tilted = [np.sin(0.3) * 0.6, np.sin(0.3) * -0.8, 0.0, np.cos(0.3)]
    common = {"mu_m3_s2": scenario["mu_m3_s2"], "dt_s": scenario["dt_ns"] / 1e9,
              "position_m": list(position), "velocity_mps": list(velocity)}
    return [
        {**common, "name": "point_mass_orbit", "duration_s": round(period / 100) * 100.0,
         "sample_s": 100.0, "mass_kg": 200.0, "inertia_kg_m2": [100.0, 120.0, 140.0],
         "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "body_rate_radps": [0.0, 0.0, 0.0],
         "force_body_n": [0.0, 0.0, 0.0], "torque_body_nm": [0.0, 0.0, 0.0]},
        {**common, "name": "torque_free_tumble", "duration_s": 1000.0, "sample_s": 10.0,
         "mass_kg": 200.0, "inertia_kg_m2": [100.0, 120.0, 140.0], "quaternion_xyzw": tilted,
         "body_rate_radps": [0.02, -0.03, 0.015],
         "force_body_n": [0.0, 0.0, 0.0], "torque_body_nm": [0.0, 0.0, 0.0]},
        {**common, "name": "constant_body_wrench", "duration_s": 300.0, "sample_s": 10.0,
         "mass_kg": 100.0, "inertia_kg_m2": [20.0, 25.0, 30.0], "quaternion_xyzw": tilted,
         "body_rate_radps": [0.001, 0.002, -0.001],
         "force_body_n": [0.3, -0.5, 0.2], "torque_body_nm": [0.01, 0.02, -0.015]},
    ]


def basilisk_cases(output):
    write_json(Path(output), {"artifact_type": "dynamics_crosscheck_cases", "cases": cases()})


def rpod_trajectory(case):
    """Propagate a case with `RigidBodyEngine`; body 1 is an unforced companion 1 km away."""
    dt_ns = round(case["dt_s"] * 1e9)
    companion = np.asarray(case["position_m"]) + [0.0, 0.0, 1000.0]
    engine = RigidBodyEngine(
        case["mu_m3_s2"], dt_ns, [case["position_m"], companion],
        [case["velocity_mps"], case["velocity_mps"]], [case["quaternion_xyzw"], [0, 0, 0, 1]],
        [case["body_rate_radps"], [0, 0, 0]], [case["mass_kg"], 1.0],
        [case["inertia_kg_m2"], [1.0, 1.0, 1.0]], [10.0, 0.0], [1.0, 0.0],
    )
    every = round(case["sample_s"] / case["dt_s"])
    samples, snapshot = [], engine.snapshot()
    for step in range(round(case["duration_s"] / case["dt_s"]) + 1):
        if step % every == 0:
            samples.append([*snapshot.positions_m[0], *snapshot.velocities_mps[0],
                            *snapshot.quaternions_xyzw[0],
                            *snapshot.angular_velocities_body_radps[0]])
        if step < round(case["duration_s"] / case["dt_s"]):
            snapshot = engine.step([case["force_body_n"], [0, 0, 0]],
                                   [case["torque_body_nm"], [0, 0, 0]])
    return np.asarray(samples)


def mrp_quaternions(mrp):
    """XYZW quaternions from (k, 3) modified Rodrigues parameters (either set)."""
    mrp = np.asarray(mrp, dtype=np.float64)
    squared = np.sum(mrp * mrp, axis=-1, keepdims=True)
    return np.concatenate([2 * mrp, 1 - squared], axis=-1) / (1 + squared)


def compare(cases_path, basilisk_path, output):
    """Largest rpod-Basilisk differences per case, against `TOLERANCES`."""
    case_list = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]
    external = json.loads(Path(basilisk_path).read_text(encoding="utf-8"))
    rows = {}
    for case in case_list:
        ours, theirs = rpod_trajectory(case), external["cases"][case["name"]]
        count = min(len(ours), len(theirs["positions_m"]))
        quaternions = mrp_quaternions(theirs["mrp"])[:count]
        relative = quaternion_products(ours[:count, 6:10] * [-1, -1, -1, 1], quaternions)
        differences = {
            "position_m": np.linalg.norm(ours[:count, :3] - theirs["positions_m"][:count], axis=-1),
            "velocity_mps": np.linalg.norm(ours[:count, 3:6] - theirs["velocities_mps"][:count],
                                           axis=-1),
            "attitude_rad": 2 * np.arcsin(np.clip(np.linalg.norm(relative[:, :3], axis=-1), 0, 1)),
            "body_rate_radps": np.linalg.norm(ours[:count, 10:] - theirs["body_rates"][:count],
                                              axis=-1),
        }
        rows[case["name"]] = {
            "samples": count, "duration_s": case["duration_s"],
            **{f"max_{key}": float(value.max()) for key, value in differences.items()},
            "passed": bool(all(value.max() <= TOLERANCES[key]
                               for key, value in differences.items())),
        }
    report = {"artifact_type": "dynamics_crosscheck", "reference": external["software"],
              "integrator": "RK4 in both, same fixed step", "tolerances": TOLERANCES,
              "cases": rows, "passed": all(row["passed"] for row in rows.values())}
    write_json(Path(output), report)
    return report
