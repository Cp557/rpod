"""Evolution-strategies training of the learned approach policy in the batched fleet.

OpenAI-ES with antithetic pairs: each generation perturbs the mean policy by ±σε per pair, runs
every candidate on the same freshly drawn episodes with the same surrogate noise, ranks the
candidates by fitness and moves the mean with Adam. Training (seed 10) and validation (seed 11)
episodes are disjoint from the M9 campaign (seed 9) and the suites (1000 + index). The mean
policy with the best validation score is kept.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import fleet
from .learned_control import LearnedFleetController
from .monte_carlo import campaign_spec
from .m5_suite import episode_initial_state
from .policy import PARAMETER_COUNT, initial_parameters, save_policy
from .surrogate import SurrogateNavigation
from .vision_records import sha256_file
from .visual_control import MissionSpec

TRAINING_SEED, VALIDATION_SEED = 10, 11
TRAINING_MAX_STEPS = 3000


def episode_fitness(result, interventions, goal_gap_m=5.0):
    """Success scores 1 plus up to 1 for speed, less Δv. Failures score below 0: lower the
    farther and faster they ended from the hold point, lower still after an abort, and
    higher for time spent inside the onboard completion envelope. Interventions cost a
    little in both."""
    success = result["outcome"] == fleet.SUCCESS
    speed = (TRAINING_MAX_STEPS / 10 - result["completion_time_s"]) / (TRAINING_MAX_STEPS / 10)
    succeeded = 1 + speed - result["delta_v_mps"]
    miss = np.hypot(result["final_gap_m"] - goal_gap_m, result["final_lateral_m"])
    # Logarithmic, so centimetres near the hold point still count.
    failed = (-1 - 0.25 * np.log1p(miss / 0.05)
              - 0.25 * np.log1p(result["final_port_speed_mps"] / 0.005)
              - (result["abort_reason"] > 0) + 0.5 * np.minimum(result["at_goal_ticks"] / 100, 1))
    return np.where(success, succeeded, failed) - 0.002 * interventions


def evaluate(parameters, specs, model, seed, groups=1):
    """Run `specs` once per parameter group, with the same surrogate noise for every group
    (common random numbers), and return per-episode fitness."""
    initials = [{**episode_initial_state(spec), "max_steps": TRAINING_MAX_STEPS}
                for spec in specs] * groups
    mission = MissionSpec()
    controller = LearnedFleetController(mission, len(initials), parameters)
    navigation = SurrogateNavigation(model, len(initials), initials[0]["dt_ns"], seed,
                                     groups=groups)
    result = fleet.run_fleet(initials, mission, navigation, controller)
    result["completion_time_s"] = result["end_tick"] * initials[0]["dt_ns"] / 1e9
    return episode_fitness(result, controller.interventions), result, controller


def _centered_ranks(values):
    ranks = np.empty(len(values))
    ranks[np.argsort(values)] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def train(output, model_path, generations=300, pairs=64, episodes=16, sigma=0.02,
          learning_rate=0.01, weight_decay=0.005, validation_episodes=1024,
          validate_every=10, seed=0):
    """Train, keep the best validated mean policy, and save it with its provenance."""
    model_path = Path(model_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    mean = initial_parameters(seed)
    first_moment, second_moment = np.zeros(PARAMETER_COUNT), np.zeros(PARAMETER_COUNT)
    validation_specs = [campaign_spec(VALIDATION_SEED, index)
                        for index in range(validation_episodes)]
    best, history, started = None, [], perf_counter()
    for generation in range(generations + 1):
        if generation % validate_every == 0 or generation == generations:
            scores, result, controller = evaluate(mean, validation_specs, model,
                                                  [VALIDATION_SEED, generation])
            success = result["outcome"] == fleet.SUCCESS
            row = {"generation": generation, "wall_seconds": perf_counter() - started,
                   "validation_fitness": float(scores.mean()),
                   "success_fraction": float(success.mean()),
                   "completion_time_s": (float(np.median(result["completion_time_s"][success]))
                                         if success.any() else None),
                   "delta_v_mps": (float(np.median(result["delta_v_mps"][success]))
                                   if success.any() else None),
                   "interventions_per_episode": float(controller.interventions.mean()),
                   "aborted_fraction": float((result["abort_reason"] > 0).mean()),
                   "median_final_miss_m": float(np.median(np.hypot(
                       result["final_gap_m"] - 5.0, result["final_lateral_m"]))),
                   "median_final_speed_mps": float(np.median(result["final_port_speed_mps"])),
                   "median_at_goal_ticks": float(np.median(result["at_goal_ticks"]))}
            history.append(row)
            print(json.dumps(row), flush=True)
            if best is None or row["validation_fitness"] > best[0]["validation_fitness"]:
                best = (row, mean.copy())
        if generation == generations:
            break
        noise = rng.standard_normal((pairs, PARAMETER_COUNT))
        candidates = np.concatenate([mean + sigma * noise, mean - sigma * noise])
        specs = [campaign_spec(TRAINING_SEED, generation * episodes + index)
                 for index in range(episodes)]
        scores, _, _ = evaluate(candidates, specs, model, [TRAINING_SEED, generation],
                                groups=len(candidates))
        fitness = _centered_ranks(scores.reshape(len(candidates), episodes).mean(axis=1))
        gradient = (fitness[:pairs] - fitness[pairs:]) @ noise / (2 * pairs * sigma)
        step = -gradient + weight_decay * mean
        first_moment = 0.9 * first_moment + 0.1 * step
        second_moment = 0.999 * second_moment + 0.001 * step**2
        corrected = first_moment / (1 - 0.9 ** (generation + 1))
        scale = np.sqrt(second_moment / (1 - 0.999 ** (generation + 1))) + 1e-8
        mean = mean - learning_rate * corrected / scale
    row, parameters = best
    source = Path(__file__).parent
    save_policy(output, parameters, {
        "artifact_type": "learned_approach_policy",
        "algorithm": "OpenAI-ES, antithetic pairs, centered ranks, Adam",
        "configuration": {"generations": generations, "pairs": pairs,
                          "episodes_per_generation": episodes, "sigma": sigma,
                          "learning_rate": learning_rate, "weight_decay": weight_decay,
                          "validation_episodes": validation_episodes, "seed": seed,
                          "training_seed": TRAINING_SEED, "validation_seed": VALIDATION_SEED,
                          "training_max_steps": TRAINING_MAX_STEPS},
        "surrogate": model_path.as_posix(), "surrogate_sha256": sha256_file(model_path),
        "implementation_sha256": {name: sha256_file(source / name) for name in (
            "policy.py", "safety_filter.py", "learned_control.py", "fleet.py", "surrogate.py",
            "train_policy.py")},
        "selected": row, "history": history,
    })
    return row
