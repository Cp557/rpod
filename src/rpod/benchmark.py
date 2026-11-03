"""Fixed-workload benchmarks for the release-mode orbital extension."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

import numpy as np
import psutil

from . import OrbitEngine, __version__, _core
from ._core import RigidBodyBatch
from .recording import collect_trajectory
from .scenarios import make_scenario


def _timing_summary(durations_ns: list[int], completed_count: int) -> dict:
    median_ns = median(durations_ns)
    return {
        "durations_ns": durations_ns,
        "median_duration_ns": median_ns,
        "completed_counts": [completed_count] * len(durations_ns),
        "median_count_per_second": completed_count * 1_000_000_000 / median_ns,
    }


def _metadata() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "rpod": __version__,
        "numpy": np.__version__,
        "rerun_sdk": version("rerun-sdk"),
        "psutil": psutil.__version__,
        "rustc": _core.__rustc_version__,
        "debug_assertions": _core.__debug_assertions__,
    }


def _require_release(output):
    if _core.__debug_assertions__:
        raise RuntimeError(
            "Benchmarking requires a release-mode Rust extension; rebuild and "
            "install the extension in release mode first."
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    return output


def _write_report(output, report):
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(serialized)
    return report


def run_benchmark(output: Path) -> dict:
    """Measure fixed workloads and write a new JSON report without overwriting."""
    output = _require_release(output)
    scenario = make_scenario("circular")
    engine = OrbitEngine(
        scenario["mu_m3_s2"],
        scenario["dt_ns"],
        scenario["initial_positions_m"],
        scenario["initial_velocities_mps"],
    )
    process = psutil.Process()
    rss_before_bytes = process.memory_info().rss
    started_at = datetime.now(timezone.utc).isoformat()

    native_ticks = 1_000_000
    native_durations = []
    native_warmup_ns = None
    for repetition in range(6):
        engine.reset()
        started_ns = perf_counter_ns()
        engine.step(native_ticks)
        duration_ns = perf_counter_ns() - started_ns
        if engine.snapshot().tick != native_ticks:
            raise RuntimeError("Native benchmark did not complete its tick count")
        if repetition == 0:
            native_warmup_ns = duration_ns
        else:
            native_durations.append(duration_ns)

    recorded_ticks = 10_000
    recorded_durations = []
    recorded_warmup_ns = None
    recorded_bytes = 0
    for repetition in range(6):
        engine.reset()
        started_ns = perf_counter_ns()
        trajectory = collect_trajectory(engine, recorded_ticks)
        duration_ns = perf_counter_ns() - started_ns
        if (
            engine.snapshot().tick != recorded_ticks
            or len(trajectory["tick"]) != recorded_ticks + 1
            or int(trajectory["tick"][-1]) != recorded_ticks
        ):
            raise RuntimeError("Recording benchmark did not record every tick")
        recorded_bytes = sum(
            array.nbytes
            for array in trajectory.values()
            if isinstance(array, np.ndarray)
        )
        del trajectory
        if repetition == 0:
            recorded_warmup_ns = duration_ns
        else:
            recorded_durations.append(duration_ns)

    reset_calls = 10_000
    reset_durations = []
    reset_warmup_ns = None
    for repetition in range(6):
        engine.step(1)
        started_ns = perf_counter_ns()
        for _ in range(reset_calls):
            engine.reset()
        duration_ns = perf_counter_ns() - started_ns
        if engine.snapshot().tick != 0:
            raise RuntimeError("Reset benchmark did not restore tick zero")
        if repetition == 0:
            reset_warmup_ns = duration_ns
        else:
            reset_durations.append(duration_ns)

    rss_after_bytes = process.memory_info().rss
    report = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "metadata": _metadata(),
        "scenario": scenario,
        "timing": {
            "clock": "perf_counter_ns",
            "warmup_repetitions_per_workload": 1,
            "measured_repetitions_per_workload": 5,
            "excluded": [
                "engine construction",
                "resets before stepping workloads",
                "post-run validation",
                "disk I/O",
                "Rerun logging and viewing",
            ],
        },
        "workloads": {
            "native_bulk": {
                "description": (
                    "One Python call to OrbitEngine.step(1000000); native "
                    "two-spacecraft propagation and the returned snapshot"
                ),
                "count_unit": "physics ticks",
                "warmup_duration_ns": native_warmup_ns,
                **_timing_summary(native_durations, native_ticks),
            },
            "python_recording": {
                "description": (
                    "Per-tick Python stepping and copied NumPy recording via "
                    "collect_trajectory for 10000 physics ticks"
                ),
                "count_unit": "physics ticks",
                "samples_per_repetition": recorded_ticks + 1,
                "recorded_array_bytes_per_repetition": recorded_bytes,
                "warmup_duration_ns": recorded_warmup_ns,
                **_timing_summary(recorded_durations, recorded_ticks),
            },
            "python_reset_calls": {
                "description": (
                    "10000 Python calls to OrbitEngine.reset(); the first "
                    "restores an advanced state and later calls restore "
                    "the already-initial state"
                ),
                "count_unit": "reset calls",
                "warmup_duration_ns": reset_warmup_ns,
                **_timing_summary(reset_durations, reset_calls),
            },
        },
        "process_memory": {
            "rss_before_bytes": rss_before_bytes,
            "rss_after_bytes": rss_after_bytes,
            "rss_change_bytes": rss_after_bytes - rss_before_bytes,
            "scope": (
                "Process RSS before and after all workloads, including "
                "warmups; not peak memory or memory per environment"
            ),
        },
    }
    return _write_report(output, report)


def _batch(count, threads):
    from .approach import make_approach_scenario
    from .fleet import manifest_state

    scenario = make_approach_scenario("approach")
    state = manifest_state(scenario)
    return scenario, RigidBodyBatch(
        np.repeat(state[None], count, axis=0), scenario["mu_m3_s2"], scenario["dt_ns"],
        scenario["masses_kg"], scenario["inertias_kg_m2"], scenario["force_limits_n"],
        scenario["torque_limits_nm"], threads=threads,
    )


def run_batch_benchmark(output: Path, counts=(1, 64, 1024, 16384, 131072),
                        threads=(1, 2, 4, 6, 12), fleet_counts=(4096, 16384)) -> dict:
    """Measure batched stepping over environment and thread counts, memory per environment,
    reset latency, and closed-loop fleet throughput; write a new JSON report."""
    from .fleet import TruthNavigation, run_fleet
    from .visual_control import MissionSpec
    from .visual_run import initial_state

    output = _require_release(output)
    started_at = datetime.now(timezone.utc).isoformat()
    native = []
    for count in counts:
        for thread_count in threads:
            _, batch = _batch(count, thread_count)
            steps = max(1, 2_000_000 // count)
            zeros, active = np.zeros((count, 2, 3)), np.ones(count, bool)
            durations = []
            for repetition in range(6):
                started_ns = perf_counter_ns()
                batch.step(zeros, zeros, active, steps)
                if repetition:
                    durations.append(perf_counter_ns() - started_ns)
            native.append({"environments": count, "threads": thread_count,
                           "ticks_per_call": steps,
                           **_timing_summary(durations, count * steps)})
    scenario, batch = _batch(max(counts), 0)
    rows = np.repeat(batch.states()[:1], 1024, axis=0)
    durations = []
    for repetition in range(6):
        started_ns = perf_counter_ns()
        batch.reset(list(range(1024)), rows)
        if repetition:
            durations.append(perf_counter_ns() - started_ns)
    resets = _timing_summary(durations, 1024)
    del batch
    closed_loop = []
    for count in fleet_counts:
        initials = [{**initial_state("visual-m5-operational", 50_000 + index), "max_steps": 600}
                    for index in range(count)]
        durations = []
        for repetition in range(4):
            started_ns = perf_counter_ns()
            result = run_fleet(initials, MissionSpec(), TruthNavigation(count, scenario["dt_ns"]))
            if repetition:
                durations.append(perf_counter_ns() - started_ns)
        closed_loop.append({"environments": count, "simulated_seconds_per_episode": 60.0,
                            **_timing_summary(durations, int(result["end_tick"].sum()))})
    return _write_report(output, {
        "schema_version": 1, "started_at_utc": started_at, "metadata": _metadata(),
        "workloads": {
            "native_batch": {
                "description": ("RigidBodyBatch.step with zero commands, holding each call's "
                                "ticks; about 2 million environment ticks per repetition; "
                                "1 warmup and 5 measured repetitions"),
                "count_unit": "environment ticks (two rigid bodies each)", "grid": native,
            },
            "memory_per_environment_bytes": {
                "value": RigidBodyBatch.bytes_per_environment,
                "scope": ("Rust engine state per environment (size of RigidBodyEngine); "
                          "excludes the Python fleet's per-episode arrays"),
            },
            "reset_latency": {"description": "RigidBodyBatch.reset of 1024 environments",
                              "count_unit": "environment resets", **resets},
            "closed_loop_fleet": {
                "description": ("run_fleet with truth navigation and the reference autonomy "
                                "for 60 simulated seconds per episode; Python supervisor and "
                                "law every tick; 1 warmup and 3 measured runs"),
                "runs": closed_loop,
            },
        },
    })
