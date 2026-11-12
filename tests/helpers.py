"""Fixtures shared by several test modules."""

from types import SimpleNamespace

import numpy as np

from rpod import fleet
from rpod._core import TargetMesh
from rpod.approach import STATE_FIELDS, make_approach_scenario


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


def random_states(count, seed, max_angle_deg=20.0):
    """Pairs near the 30 m handoff with random offsets, attitudes and rates."""
    rng = np.random.default_rng(seed)
    scenario = make_approach_scenario("approach")
    base = fleet.manifest_state(scenario)
    states = np.repeat(base[None], count, axis=0)
    states[:, 1, :3] = states[:, 0, :3] + [0, -31.5, 0] + rng.normal(0, [0.5, 3, 0.5], (count, 3))
    states[:, :, 3:6] += rng.normal(0, 0.02, (count, 2, 3))
    axes = rng.normal(size=(count, 2, 3))
    angles = rng.uniform(0, np.deg2rad(max_angle_deg), (count, 2, 1))
    axes *= np.sin(angles / 2) / np.linalg.norm(axes, axis=-1, keepdims=True)
    states[:, :, 6:10] = np.concatenate([axes, np.cos(angles / 2)], axis=-1)
    states[:, :, 10:] += rng.normal(0, 0.003, (count, 2, 3))
    return states


def snapshot(state):
    return SimpleNamespace(positions_m=state[:, :3], velocities_mps=state[:, 3:6],
                           quaternions_xyzw=state[:, 6:10],
                           angular_velocities_body_radps=state[:, 10:])


def box(center, half):
    """Triangles of an axis-aligned box."""
    corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float)
    faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
             [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return np.asarray(center) + corners * half, np.array(faces)


def asymmetric_mesh():
    """A bus with a wing and a boom, so no rotation maps it onto itself."""
    parts = [box((0, 0, 0), (1.0, 1.0, 1.0)), box((3.0, 0.2, 0), (2.0, 0.05, 0.8)),
             box((-0.4, 0.5, -1.8), (0.15, 0.15, 0.8))]
    vertices, triangles, offset = [], [], 0
    for part_vertices, part_triangles in parts:
        vertices.append(part_vertices)
        triangles.append(part_triangles + offset)
        offset += len(part_vertices)
    return TargetMesh(np.vstack(vertices), np.vstack(triangles).astype(np.uint32))
