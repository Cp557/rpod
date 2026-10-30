"""Resolved, deterministic orbital experiments for M1."""

import math

MU_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_137.0
ORBIT_RADIUS_M = EARTH_RADIUS_M + 400_000.0
APPROACH_SCENARIOS = ("approach", "approach-disturbed", "approach-retreat")
SCENARIOS = ("circular", "drift") + APPROACH_SCENARIOS


def make_scenario(name: str) -> dict:
    """Return fresh SI initial conditions; body order is target, chaser."""
    if name in APPROACH_SCENARIOS:
        from .approach import make_approach_scenario

        return make_approach_scenario(name)
    if name not in ("circular", "drift"):
        raise ValueError(f"unknown scenario: {name!r}")
    radius = ORBIT_RADIUS_M
    speed = math.sqrt(MU_M3_S2 / radius)
    phase = -100.0 / radius
    positions = [
        [radius, 0.0, 0.0],
        [radius * math.cos(phase), radius * math.sin(phase), 0.0],
    ]
    velocities = [
        [0.0, speed, 0.0],
        [-speed * math.sin(phase), speed * math.cos(phase), 0.0],
    ]
    perturbation = [0.0, 0.05, 0.02] if name == "drift" else [0.0, 0.0, 0.0]
    # At t=0 the target's radial/along-track/normal basis equals the inertial basis.
    velocities[1] = [v + dv for v, dv in zip(velocities[1], perturbation)]
    return {
        "schema_version": 1,
        "scenario": name,
        "body_order": ["target", "chaser"],
        "mu_m3_s2": MU_M3_S2,
        "earth_radius_m": EARTH_RADIUS_M,
        "dt_ns": 1_000_000_000,
        "num_steps": 600,
        "initial_positions_m": positions,
        "initial_velocities_mps": velocities,
        "initial_arc_separation_m": 100.0,
        "chaser_delta_v_target_lvlh_mps": perturbation,
        "units": {"position": "m", "velocity": "m/s", "time": "ns"},
        "frames": {
            "inertial": (
                "Earth-centered, fixed axes; initial target on +X, velocity +Y; "
                "no epoch or Earth rotation"
            ),
            "target_lvlh": (
                "+X radial outward, +Y along track, +Z orbit normal; "
                "R_NL columns are local axes in inertial coordinates"
            ),
            "prescribed_orientation": (
                "both spacecraft body axes aligned to target LVLH; "
                "no rotational dynamics"
            ),
        },
        "model": (
            "independent point-mass central gravity; fixed-step RK4; "
            "no contact or Earth collision detection"
        ),
    }
