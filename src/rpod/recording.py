"""Headless experiments and viewer-independent numerical records."""

from importlib.metadata import version
import json
from pathlib import Path
import platform
from zipfile import BadZipFile

import numpy as np

from . import OrbitEngine, _core
from .frames import relative_state
from .scenarios import APPROACH_SCENARIOS, SCENARIOS, make_scenario


def software_info() -> dict:
    return {
        "rpod": version("rpod"),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "rerun_sdk": version("rerun-sdk"),
        "platform": platform.platform(),
        "rustc": _core.__rustc_version__,
        "native_debug_assertions": _core.__debug_assertions__,
    }


def collect_trajectory(engine: OrbitEngine, num_steps: int) -> dict:
    """Collect initial state and each subsequent tick, with no disk or rendering."""
    if not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    ticks = np.empty(num_steps + 1, dtype=np.uint64)
    positions = np.empty((num_steps + 1, 2, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    snapshot = engine.snapshot()
    for index in range(num_steps + 1):
        ticks[index] = snapshot.tick
        positions[index] = snapshot.positions_m
        velocities[index] = snapshot.velocities_mps
        if index < num_steps:
            snapshot = engine.step()
    return {"tick": ticks, "positions_m": positions, "velocities_mps": velocities}


def summarize(manifest: dict, trajectory: dict) -> dict:
    positions = trajectory["positions_m"]
    velocities = trajectory["velocities_mps"]
    rho, rho_dot, _ = relative_state(positions, velocities)
    ranges = np.linalg.norm(rho, axis=-1)
    energy = (
        0.5 * np.sum(velocities**2, axis=-1)
        - manifest["mu_m3_s2"] / np.linalg.norm(positions, axis=-1)
    )
    momentum = np.cross(positions, velocities)
    result = {
        "samples": len(positions),
        "elapsed_time_ns": int(trajectory["tick"][-1]) * manifest["dt_ns"],
        "range_m": {
            "initial": float(ranges[0]),
            "final": float(ranges[-1]),
            "min": float(ranges.min()),
            "max": float(ranges.max()),
        },
        "final_relative_position_m": rho[-1].tolist(),
        "final_relative_velocity_mps": rho_dot[-1].tolist(),
        "max_relative_position_change_m": float(np.linalg.norm(rho - rho[0], axis=-1).max()),
        "max_relative_speed_mps": float(np.linalg.norm(rho_dot, axis=-1).max()),
        "max_fractional_specific_energy_change": np.max(
            np.abs(energy - energy[0]) / np.abs(energy[0]), axis=0
        ).tolist(),
        "max_fractional_angular_momentum_vector_change": np.max(
            np.linalg.norm(momentum - momentum[0], axis=-1)
            / np.linalg.norm(momentum[0], axis=-1),
            axis=0,
        ).tolist(),
    }
    if manifest["scenario"] == "circular":
        # Closed-form circular motion, independent of the numerical integrator.
        r0 = np.asarray(manifest["initial_positions_m"])
        v0 = np.asarray(manifest["initial_velocities_mps"])
        n = np.sqrt(manifest["mu_m3_s2"] / np.linalg.norm(r0, axis=-1)**3)
        time_s = trajectory["tick"].astype(np.float64) * (manifest["dt_ns"] / 1e9)
        angle = time_s[:, None, None] * n[None, :, None]
        expected_r = r0 * np.cos(angle) + v0 / n[:, None] * np.sin(angle)
        expected_v = -r0 * n[:, None] * np.sin(angle) + v0 * np.cos(angle)
        result["analytical_max_position_error_m"] = np.max(
            np.linalg.norm(positions - expected_r, axis=-1), axis=0
        ).tolist()
        result["analytical_max_velocity_error_mps"] = np.max(
            np.linalg.norm(velocities - expected_v, axis=-1), axis=0
        ).tolist()
    return result


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_scenario(scenario: str, output: Path) -> dict:
    """Run a built-in experiment and save numerical and derived visual artifacts."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    manifest = make_scenario(scenario)
    events, outcome = None, None
    if scenario in APPROACH_SCENARIOS:
        from .approach import simulate_approach

        trajectory, events, outcome = simulate_approach(manifest)
        manifest["max_steps"] = manifest["num_steps"]
        manifest["num_steps"] = int(trajectory["tick"][-1])
    else:
        engine = OrbitEngine(
            manifest["mu_m3_s2"],
            manifest["dt_ns"],
            manifest["initial_positions_m"],
            manifest["initial_velocities_mps"],
        )
        trajectory = collect_trajectory(engine, manifest["num_steps"])
    manifest["software"] = software_info()
    summary = summarize(manifest, trajectory)
    if outcome is not None:
        summary.update(outcome)
        summary["controller_updates"] = int(trajectory["control_update"].sum())
        for quantity, units in (("forces", "n"), ("torques", "nm")):
            norms = np.linalg.norm(trajectory[f"applied_{quantity}_body_{units}"], axis=-1)
            summary[f"max_applied_{quantity}_{units}"] = norms.max(axis=0, initial=0).tolist()
            summary[f"integrated_{quantity}_norm_{units}_s"] = (
                norms.sum(axis=0) * manifest["dt_ns"] / 1e9
            ).tolist()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    np.savez(output / "trajectory.npz", **trajectory)
    write_json(output / "summary.json", summary)
    if events is not None:
        write_json(output / "events.json", events)
    from .visualization import export_replay

    if events is None:
        export_replay(manifest, trajectory, output / "replay.rrd")
    else:
        export_replay(manifest, trajectory, output / "replay.rrd", events=events)
    return summary


def load_run(directory: Path) -> tuple[dict, dict]:
    """Load schema 1/2 records without constructing a simulator or unpickling."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and manifest.get("schema_version") == 3:
        from .visual_recording import load_visual_run

        return load_visual_run(directory)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in (1, 2)
    ):
        raise ValueError("unsupported numerical record schema; expected version 1 or 2")
    try:
        steps, dt_ns = manifest["num_steps"], manifest["dt_ns"]
        if (
            type(steps) is not int
            or steps < 0
            or type(dt_ns) is not int
            or not 0 < dt_ns <= 2**64 - 1
            or steps * dt_ns > 2**64 - 1
        ):
            raise ValueError("invalid recorded step count or timestep")
        if manifest["body_order"] != ["target", "chaser"] or manifest["scenario"] not in SCENARIOS:
            raise ValueError("invalid body order or scenario")
        rigid = manifest["schema_version"] == 2
        if rigid != (manifest["scenario"] in APPROACH_SCENARIOS):
            raise ValueError("scenario and record schema disagree")
        for field in ("mu_m3_s2", "earth_radius_m"):
            if not np.isfinite(manifest[field]) or manifest[field] <= 0:
                raise ValueError(f"invalid {field}")
        initial_r = np.asarray(manifest["initial_positions_m"], dtype=np.float64)
        initial_v = np.asarray(manifest["initial_velocities_mps"], dtype=np.float64)
        with (
            (directory / "trajectory.npz").open("rb") as source,
            np.load(source, allow_pickle=False) as data,
        ):
            keys = ["tick", "positions_m", "velocities_mps"]
            if rigid:
                from .approach import COMMAND_FIELDS

                keys += [
                    "quaternions_xyzw",
                    "angular_velocities_body_radps",
                    "phase",
                    "desired_gap_m",
                    "control_update",
                    *COMMAND_FIELDS,
                ]
            trajectory = {key: data[key] for key in keys}
        if any(not isinstance(array, np.ndarray) for array in trajectory.values()):
            raise ValueError("recorded fields must be NumPy arrays")
        tick = trajectory["tick"]
        if (
            tick.dtype != np.dtype("uint64")
            or tick.shape != (steps + 1,)
            or not np.array_equal(tick, np.arange(steps + 1, dtype=np.uint64))
        ):
            raise ValueError("recorded ticks must be contiguous from zero")
        for key, initial in (("positions_m", initial_r), ("velocities_mps", initial_v)):
            array = trajectory[key]
            if (
                array.dtype != np.dtype("float64")
                or array.shape != (steps + 1, 2, 3)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid recorded {key}")
            if initial.shape != (2, 3) or not np.array_equal(array[0], initial):
                raise ValueError(f"recorded initial {key} disagrees with manifest")
        if np.any(np.linalg.norm(trajectory["positions_m"], axis=-1) == 0):
            raise ValueError("recorded positions contain a singular radius")
        if rigid:
            _validate_rigid_record(manifest, trajectory)
    except (KeyError, TypeError, OverflowError, BadZipFile, EOFError) as error:
        raise ValueError(f"invalid numerical record: {error}") from error
    return manifest, trajectory


def _validate_rigid_record(manifest, trajectory):
    """Check the additional state/interval contracts of controlled records."""
    from .approach import COMMAND_FIELDS

    steps = manifest["num_steps"]
    shapes = {
        "quaternions_xyzw": (steps + 1, 2, 4),
        "angular_velocities_body_radps": (steps + 1, 2, 3),
        "desired_gap_m": (steps + 1,),
    }
    shapes.update({key: (steps, 2, 3) for key in COMMAND_FIELDS})
    for key, shape in shapes.items():
        array = trajectory[key]
        if (
            array.dtype != np.dtype("float64")
            or array.shape != shape
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid recorded {key}")
    for key, dtype, shape in (
        ("phase", np.uint8, (steps + 1,)),
        ("control_update", np.bool_, (steps,)),
    ):
        array = trajectory[key]
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"invalid recorded {key}")
    if np.any(trajectory["phase"] > 3) or trajectory["phase"][-1] != 3:
        raise ValueError("invalid recorded supervisor phases")
    q = trajectory["quaternions_xyzw"]
    if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError("recorded quaternions must be unit length")
    for field, shape, positive in (
        ("masses_kg", (2,), True),
        ("inertias_kg_m2", (2, 3), True),
        ("force_limits_n", (2,), False),
        ("torque_limits_nm", (2,), False),
        ("docking_ports_body_m", (2, 3), None),
    ):
        values = np.asarray(manifest[field], dtype=np.float64)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid {field}")
        if positive is not None and np.any(values <= 0 if positive else values < 0):
            raise ValueError(f"invalid {field}")
    for quantity, units, limit in (
        ("forces", "n", "force_limits_n"),
        ("torques", "nm", "torque_limits_nm"),
    ):
        norms = np.linalg.norm(trajectory[f"applied_{quantity}_body_{units}"], axis=-1)
        if np.any(norms > np.asarray(manifest[limit]) + 1e-10):
            raise ValueError(f"recorded applied {quantity} exceed limits")
    for key in ("quaternions_xyzw", "angular_velocities_body_radps"):
        initial = np.asarray(manifest[f"initial_{key}"], dtype=np.float64)
        if initial.shape != trajectory[key][0].shape or not np.allclose(
            initial, trajectory[key][0], atol=1e-12, rtol=0
        ):
            raise ValueError(f"recorded initial {key} disagrees with manifest")


def load_events(directory, manifest):
    """Read tick-ordered events; their timestamps share the physical clock."""
    if manifest["schema_version"] == 1:
        return []
    if manifest["schema_version"] == 3:
        from .visual_recording import load_visual_events

        return load_visual_events(directory, manifest)
    events = json.loads((Path(directory) / "events.json").read_text(encoding="utf-8"))
    if not isinstance(events, list) or not events:
        raise ValueError("controlled records require an event list")
    previous_tick = -1
    kinds = {
        "supervisor_transition",
        "safety_detection",
        "physical_boundary",
        "outcome",
        "abort_request",
    }
    for event in events:
        if (
            not isinstance(event, dict)
            or type(event.get("tick")) is not int
            or not previous_tick <= event["tick"] <= manifest["num_steps"]
            or event["tick"] < 0
            or type(event.get("time_ns")) is not int
            or event["time_ns"] != event["tick"] * manifest["dt_ns"]
            or type(event.get("phase")) is not int
            or not 0 <= event["phase"] <= 3
            or not isinstance(event.get("kind"), str)
            or event["kind"] not in kinds
            or not isinstance(event.get("reason"), str)
        ):
            raise ValueError("invalid tick-ordered event record")
        previous_tick = event["tick"]
    if events[-1]["kind"] != "outcome" or events[-1]["tick"] != manifest["num_steps"]:
        raise ValueError("controlled records must end with a timestamped outcome")
    return events


def replay_run(directory: Path, save_only: bool = False) -> Path:
    if (Path(directory) / "dataset.json").is_file():
        from .vision_replay import replay_dataset

        return replay_dataset(directory, save_only)
    header = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if isinstance(header, dict) and header.get("schema_version") == 3:
        from .visual_recording import replay_visual_run

        return replay_visual_run(directory, save_only)
    manifest, trajectory = load_run(directory)
    from .visualization import export_replay, open_replay

    if manifest["schema_version"] == 2:
        path = export_replay(
            manifest,
            trajectory,
            Path(directory) / "replay.rrd",
            events=load_events(directory, manifest),
        )
    else:
        path = export_replay(manifest, trajectory, Path(directory) / "replay.rrd")
    if not save_only:
        open_replay(path)
    return path
