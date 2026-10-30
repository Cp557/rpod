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


def run_benchmark(output: Path) -> dict:
    """Measure fixed workloads and write a new JSON report without overwriting."""
    if _core.__debug_assertions__:
        raise RuntimeError(
            "Benchmarking requires a release-mode Rust extension; rebuild and "
            "install the extension in release mode first."
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")

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
        "metadata": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "rpod": __version__,
            "numpy": np.__version__,
            "rerun_sdk": version("rerun-sdk"),
            "psutil": psutil.__version__,
            "rustc": _core.__rustc_version__,
            "debug_assertions": _core.__debug_assertions__,
        },
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
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(serialized)
    return report
