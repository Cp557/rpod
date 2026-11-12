"""Fleet Monte Carlo campaigns of the M5 operational approach with a navigation surrogate.

A campaign samples the M5 operational distribution with a new seed, runs every episode in
the batched engine with the fleet autonomy and a fitted surrogate, and reports outcome
counts with a Clopper-Pearson interval. Any episode can be replayed as a full rendered run.
"""

from collections import namedtuple
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from . import fleet
from .m5_suite import episode_initial_state, operational_conditions
from .recording import write_json
from .surrogate import SurrogateNavigation, episode_samples, fit_surrogate, half_specs
from .vision_records import json_value, new_output, sha256_file
from .visual_control import MissionSpec
from .visual_run import load_qualified_calibration, run_episode

QUANTILES = (0.0, 0.01, 0.5, 0.99, 1.0)


def campaign_spec(seed, index, scene="realistic"):
    """One operational episode of a campaign; seeds stay clear of the suites' 1000+index."""
    return {
        "id": f"{index:07d}", "index": index, "category": "operational", "kind": "approach",
        "init_seed": seed * 10_000_000 + index, "abort_gap_m": None,
        "expected_outcome": "success", "conditions": operational_conditions(seed, index),
        "streams": {"conditions": [seed, index, 1], "image_noise": [seed, index, 3]},
        "scene": scene,
    }


# What a campaign flies: its seeded episode specs, their initial states, the fleet controller
# and the environment scoring.
Scenario = namedtuple("Scenario", "name spec initial_state controller scoring")
OPERATIONAL = Scenario("operational", campaign_spec, episode_initial_state,
                       fleet.FleetController, fleet.Scoring)
SCOPES = {"operational": "M5 operational distribution",
          "tumbling": "M5 operational distribution against the free-tumbling M11 target"}


def _betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b), by continued fraction."""
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x > (a + 1) / (a + b + 2):
        return 1 - _betai(b, a, 1 - x)
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d if abs(d) > 1e-300 else 1e300
    result = d
    for m in range(1, 10_000):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + numerator * d
            d = 1 / d if abs(d) > 1e-300 else 1e300
            c = 1 + numerator / c if abs(c) > 1e-300 else 1e300
            result *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return front * result / a


def _beta_quantile(probability, a, b):
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if _betai(a, b, middle) < probability else (low, middle)
    return (low + high) / 2


def clopper_pearson(successes, count, confidence=0.95):
    """Exact two-sided binomial confidence interval for a success fraction."""
    alpha = 1 - confidence
    lower = 0.0 if successes == 0 else _beta_quantile(alpha / 2, successes, count - successes + 1)
    upper = 1.0 if successes == count else _beta_quantile(1 - alpha / 2, successes + 1,
                                                          count - successes)
    return lower, upper


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return dict(zip(("min", "p01", "median", "p99", "max"),
                    np.quantile(values, QUANTILES).tolist()))


def run_batch(specs, model, seed, controller=None, threads=0, record=False,
              scenario=OPERATIONAL):
    """Run campaign specs in one fleet batch with surrogate navigation."""
    initials = [scenario.initial_state(spec) for spec in specs]
    navigation = SurrogateNavigation(model, len(specs), initials[0]["dt_ns"], seed, record)
    mission = MissionSpec()
    controller = controller or scenario.controller(
        mission, len(specs), abort_gaps_m=[spec["abort_gap_m"] or np.nan for spec in specs])
    result = fleet.run_fleet(initials, mission, navigation, controller, threads,
                             scenario.scoring(mission))
    result["completion_time_s"] = result["end_tick"] * initials[0]["dt_ns"] / 1e9
    return result, navigation


def campaign_report(config, results, wall_seconds):
    outcomes = results["outcome"]
    success = outcomes == fleet.SUCCESS
    count, successes = len(outcomes), int(success.sum())
    return json_value({
        "artifact_type": "monte_carlo_report", "campaign": config, "episodes": count,
        "scope": (f"{SCOPES[config['scenario']]} with a noisy-state navigation surrogate "
                  f"fitted to rendered realistic-scene records; {config['controller']} controller "
                  "in the batched engine; not image-based navigation"),
        "outcomes": {fleet.OUTCOMES[code]: int((outcomes == code).sum())
                     for code in np.unique(outcomes)},
        "success_fraction": successes / count,
        "success_fraction_95_interval": clopper_pearson(successes, count),
        "completion_time_s": _quantiles(results["completion_time_s"][success]),
        "delta_v_mps": _quantiles(results["delta_v_mps"][success]),
        "minimum_port_gap_m": _quantiles(results["minimum_port_gap_m"]),
        "minimum_com_range_m": _quantiles(results["minimum_com_range_m"]),
        "final_port_speed_mps": _quantiles(results["final_port_speed_mps"][success]),
        "episodes_with_brakes": int((results["brakes"] > 0).sum()),
        "safety_filter_interventions": {
            "episodes_with_any": int((results["interventions"] > 0).sum()),
            "per_episode": _quantiles(results["interventions"])},
        "true_state_outside_safe_set": {
            "episodes_with_any": int((results["envelope_violation_ticks"] > 0).sum()),
            "ticks_per_episode": _quantiles(results["envelope_violation_ticks"])},
        "maximum_brakes": int(results["brakes"].max()),
        "abort_reasons": {fleet.ABORT_REASONS[code]: int((results["abort_reason"] == code).sum())
                          for code in np.unique(results["abort_reason"]) if code},
        "performance": {"wall_seconds": wall_seconds,
                        "episodes_per_second": count / wall_seconds,
                        "env_ticks_per_second": float(results["end_tick"].sum()) / wall_seconds},
    })


def run_campaign(output, count, seed, model_path, batch_size=16384, threads=0,
                 controller_factory=None, label="reference", provenance=None,
                 scenario=OPERATIONAL):
    """Run a campaign and write campaign.json, episodes.npz, failures.jsonl and report.json.

    `controller_factory(mission, count)` builds a fleet controller; the reference by default.
    """
    output, model_path = new_output(output), Path(model_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    source = Path(__file__).parent
    config = {
        "seed": seed, "episodes": count, "batch_size": batch_size, "controller": label,
        "scenario": scenario.name,
        **(provenance or {}),
        "model": model_path.as_posix(), "model_sha256": sha256_file(model_path),
        "implementation_sha256": {name: sha256_file(source / name) for name in (
            "fleet.py", "surrogate.py", "monte_carlo.py")},
    }
    write_json(output / "campaign.json", config)
    started, parts = perf_counter(), []
    for first in range(0, count, batch_size):
        specs = [scenario.spec(seed, index)
                 for index in range(first, min(first + batch_size, count))]
        controller = controller_factory(MissionSpec(), len(specs)) if controller_factory else None
        result, _ = run_batch(specs, model, [seed, first], controller, threads,
                              scenario=scenario)
        parts.append(result)
        done = first + len(specs)
        print(f"{done}/{count} episodes, {perf_counter() - started:.0f} s", flush=True)
    keys = ("outcome", "end_tick", "completion_time_s", "minimum_port_gap_m",
            "minimum_com_range_m", "final_port_speed_mps", "final_gap_m", "delta_v_mps",
            "brakes", "abort_reason", "interventions", "envelope_violation_ticks")
    results = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    wall = perf_counter() - started
    np.savez_compressed(output / "episodes.npz", **results)
    with (output / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for index in np.flatnonzero(results["outcome"] != fleet.SUCCESS):
            stream.write(json.dumps({"index": int(index),
                                     "outcome": fleet.OUTCOMES[results["outcome"][index]],
                                     "spec": scenario.spec(seed, int(index))}) + "\n")
    report = campaign_report(config, results, wall)
    write_json(output / "report.json", report)
    return report


def replay_episode(campaign, index, calibration_path, output):
    """Run one campaign episode as a full rendered visual episode. Only campaigns of the
    reference controller in the operational scenario (M9) can be replayed; the others would
    need their own controller and environment."""
    config = json.loads((Path(campaign) / "campaign.json").read_text(encoding="utf-8"))
    # M9 campaigns predate the scenario key; every campaign records its controller.
    kind = (config["controller"], config.get("scenario", OPERATIONAL.name))
    if kind != ("reference", OPERATIONAL.name):
        raise ValueError("only reference-controller operational campaigns can be replayed; "
                         f"this one is {kind[0]} in {kind[1]}")
    calibration = load_qualified_calibration(calibration_path)
    spec = campaign_spec(config["seed"], index)
    return run_episode(episode_initial_state(spec), MissionSpec(), calibration,
                       Path(calibration_path), output, paired_baseline=True)


def validate_surrogate(suite_directory, model, draws=200, seed=0, held_out_half=1,
                       scenario=OPERATIONAL):
    """Compare the surrogate fleet with the rendered suite it was fitted to.

    Each of the 28 operational episodes runs `draws` times; per metric, at least 25 rendered
    values must lie inside the surrogate's central 95% band. Held-out estimate-error medians
    and P95 must lie within 25% of the surrogate's.
    """
    suite_directory = Path(suite_directory)
    suite = json.loads((suite_directory / "suite.json").read_text(encoding="utf-8"))
    specs = [spec for spec in suite["episodes"] if spec["category"] == "operational"]
    result, navigation = run_batch([spec for spec in specs for _ in range(draws)], model, seed,
                                   record=True, scenario=scenario)
    rendered = []
    for spec in specs:
        path = suite_directory / "episodes" / spec["id"]
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        events = json.loads((path / "events.json").read_text(encoding="utf-8"))
        rendered.append({
            "outcome": summary["outcome"],
            "completion_time_s": summary["elapsed_time_ns"] / 1e9,
            "minimum_port_gap_m": summary["minimum_port_gap_m"],
            "final_port_speed_mps": summary["final_port_speed_mps"],
            "brakes": sum(event.get("reason") == "visual_brake" for event in events),
        })
    metrics = {}
    for name in ("completion_time_s", "minimum_port_gap_m", "final_port_speed_mps", "brakes"):
        values = np.asarray(result[name], dtype=np.float64).reshape(len(specs), draws)
        bands = np.quantile(values, [0.025, 0.975], axis=1).T
        inside = [low <= row[name] <= high for row, (low, high) in zip(rendered, bands)]
        metrics[name] = {"inside_band": int(sum(inside)), "episodes": len(specs),
                         "passed": bool(sum(inside) >= 25),
                         "outside": [spec["id"] for spec, ok in zip(specs, inside) if not ok]}
    held_out = [episode_samples(suite_directory / "episodes" / spec["id"])["errors"]
                for spec in half_specs(suite, held_out_half)]
    recorded = np.concatenate(navigation.recorded)
    errors = {}
    for name, rows in (("position_m", slice(0, 3)), ("attitude_rad", slice(6, 9))):
        actual = np.linalg.norm(np.concatenate(held_out)[:, rows], axis=-1)
        modeled = np.linalg.norm(recorded[:, rows], axis=-1)
        pairs = {stat: (float(np.quantile(actual, q)), float(np.quantile(modeled, q)))
                 for stat, q in (("median", 0.5), ("p95", 0.95))}
        errors[name] = {stat: {"held_out": a, "surrogate": m, "relative": abs(m - a) / a}
                        for stat, (a, m) in pairs.items()}
        errors[name]["passed"] = bool(all(value["relative"] <= 0.25 for value in
                                     errors[name].values() if isinstance(value, dict)))
    outcomes = result["outcome"]
    return {
        "draws_per_episode": draws, "seed": seed,
        "surrogate_outcomes": {fleet.OUTCOMES[code]: int((outcomes == code).sum())
                               for code in np.unique(outcomes)},
        "rendered_outcomes": {outcome: sum(row["outcome"] == outcome for row in rendered)
                              for outcome in {row["outcome"] for row in rendered}},
        "outcome_metrics": metrics, "estimate_errors": errors,
        "passed": all(value["passed"] for value in (*metrics.values(), *errors.values())),
    }


def fit_and_validate(suite_directory, output, draws=200, scenario=OPERATIONAL,
                     failures="deliveries"):
    """Validate the fitting procedure on half the suite, then fit the model on all of it."""
    validation = validate_surrogate(
        suite_directory, fit_surrogate(suite_directory, half=0, failures=failures), draws,
        scenario=scenario)
    model = fit_surrogate(suite_directory, failures=failures)
    model["validation"] = {
        "procedure": ("fitted on half 0 of the operational episodes; rendered outcome metrics "
                      "of all 28 compared with surrogate bands; errors compared with half 1"),
        **validation,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A tracked artifact: LF bytes, so its recorded hash matches a fresh checkout.
    output.write_text(json.dumps(json_value(model), indent=2, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    return model
