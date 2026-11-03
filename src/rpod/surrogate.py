"""Navigation surrogate fitted to recorded filter errors, for fleet Monte Carlo.

The surrogate replaces rendering, perception and the filter with what they produced in a
recorded suite: estimate errors at each camera delivery, as functions of range (position,
velocity) or of time since the first estimate (attitude, rate, which converge), with lag-1
correlation between deliveries, and delivery rejections. Between deliveries the position error
drifts with the velocity error, as a filter's prediction would. It is identified as a
noisy-state surrogate, not image-based navigation.
"""

import json
from pathlib import Path

import numpy as np

from .fleet import ACCEPTED_UPDATE, FAILED_UPDATE, NO_UPDATE, Estimates, lvlh, norm, \
    quaternion_products, rotation_matrices
from .vision_records import sha256_file
from .visual_recording import _load_visual

# Interior bin edges: center-of-mass range for position and velocity errors, and seconds since
# the first estimate for attitude and rate errors.
RANGE_EDGES_M = (7.0, 8.45, 11.0, 13.5, 15.95, 19.0, 22.5, 25.95, 28.5)
TIME_EDGES_S = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 160.0)
FRESH_AGE_NS = 1_200_000_000


def estimate_errors(truth, estimate):
    """(k, 12) errors of (k, 13) target estimates against truth, laid out as `fleet.split`:
    LVLH position and velocity, body small-angle attitude, and body rate."""
    basis, _ = lvlh(truth[:, :3], truth[:, 3:6])
    difference = np.einsum("kji,kjl->kil", rotation_matrices(truth[:, 6:10]),
                           rotation_matrices(estimate[:, 6:10]))
    attitude = 0.5 * np.stack([difference[:, 2, 1] - difference[:, 1, 2],
                               difference[:, 0, 2] - difference[:, 2, 0],
                               difference[:, 1, 0] - difference[:, 0, 1]], axis=-1)
    return np.concatenate([
        np.einsum("kji,kj->ki", basis, estimate[:, :3] - truth[:, :3]),
        np.einsum("kji,kj->ki", basis, estimate[:, 3:6] - truth[:, 3:6]),
        attitude, estimate[:, 10:] - truth[:, 10:],
    ], axis=-1)


def apply_errors(truth, errors):
    """Target estimates from (n, 13) truth and (n, 12) errors; inverts `estimate_errors`."""
    basis, _ = lvlh(truth[:, :3], truth[:, 3:6])
    half = np.concatenate([errors[:, 6:9] / 2, np.ones((len(errors), 1))], axis=-1)
    quaternion = quaternion_products(truth[:, 6:10], half / norm(half)[:, None])
    return np.concatenate([
        truth[:, :3] + np.einsum("nij,nj->ni", basis, errors[:, :3]),
        truth[:, 3:6] + np.einsum("nij,nj->ni", basis, errors[:, 3:6]),
        quaternion, truth[:, 10:] + errors[:, 9:],
    ], axis=-1)


def episode_samples(directory):
    """Estimate errors at control deliveries while tracking, with their range and time keys,
    and the episode's delivered and rejected camera captures."""
    manifest, trajectory, _, observations, estimates, _ = _load_visual(directory)
    states = np.concatenate([trajectory[key] for key in (
        "positions_m", "velocities_mps", "quaternions_xyzw", "angular_velocities_body_radps",
    )], axis=-1)
    first = next(row["target"]["capture_time_ns"] for row in estimates if row["target"])
    ticks = [tick for tick, row in enumerate(estimates)
             if row["target"] and tick % 10 == 2 and trajectory["phase"][tick] in (1, 2, 5)]
    rows = [estimates[tick]["target"] for tick in ticks]
    estimate = np.array([[*row["position_m"], *row["velocity_mps"], *row["quaternion_xyzw"],
                          *(rotation_matrices(np.asarray(row["quaternion_xyzw"])).T
                            @ row["angular_velocity_inertial_radps"])] for row in rows])
    truth = states[ticks]
    delivered = [row for row in observations if row.get("pose_estimate") is not None
                 and row["pose_estimate"].get("reason") != "missing_capture"]
    return {
        "ticks": np.asarray(ticks),
        "errors": estimate_errors(truth[:, 0], estimate),
        "range_m": norm(truth[:, 1, :3] - truth[:, 0, :3]),
        "seconds": (np.asarray(ticks) * manifest["dt_ns"] - first) / 1e9,
        "delivery_seconds": np.array([(row["capture_time_ns"] - first) / 1e9
                                      for row in delivered]),
        "rejected": np.array([not row.get("navigation_accepted") for row in delivered]),
    }


def _binned(keys, values, edges):
    """Per-bin key centers, means and standard deviations of (k, 6) values."""
    bins = np.digitize(keys, edges)
    members = [bins == index for index in range(len(edges) + 1)]
    return {"centers": [float(keys[rows].mean()) for rows in members],
            "mean": [values[rows].mean(axis=0).tolist() for rows in members],
            "std": [values[rows].std(axis=0).tolist() for rows in members]}


def _interpolate(group, keys):
    """Per-row (n, 6) means and standard deviations at `keys`, linear between bin centers."""
    centers = np.asarray(group["centers"])
    return [np.stack([np.interp(keys, centers, column) for column in np.asarray(field).T], -1)
            for field in (group["mean"], group["std"])]


def _standardized(model, samples):
    """Errors as standardized residuals under `model`'s means and deviations."""
    means = np.concatenate([_interpolate(model["position_velocity"], samples["range_m"])[0],
                            _interpolate(model["attitude_rate"], samples["seconds"])[0]], -1)
    stds = np.concatenate([_interpolate(model["position_velocity"], samples["range_m"])[1],
                           _interpolate(model["attitude_rate"], samples["seconds"])[1]], -1)
    return (samples["errors"] - means) / stds


def half_specs(suite, half=None):
    """Operational episodes, or one half of them: pairs of indices alternate, so each half
    holds both the degraded (even) and clean (odd) image conditions."""
    return [spec for spec in suite["episodes"] if spec["category"] == "operational"
            and (half is None or (spec["index"] // 2) % 2 == half)]


def fit_surrogate(suite_directory, half=None):
    """Fit on a recorded suite's operational episodes, or on one half (`half_specs`)."""
    suite_directory = Path(suite_directory)
    suite = json.loads((suite_directory / "suite.json").read_text(encoding="utf-8"))
    specs = half_specs(suite, half)
    samples = [episode_samples(suite_directory / "episodes" / spec["id"]) for spec in specs]
    pooled = {key: np.concatenate([sample[key] for sample in samples])
              for key in ("errors", "range_m", "seconds", "delivery_seconds", "rejected")}
    model = {
        "artifact_type": "navigation_surrogate",
        "model": ("noisy-state surrogate: recorded filter errors at camera deliveries, AR(1) "
                  "between deliveries, position drift by the velocity error between them, "
                  "Bernoulli delivery rejection"),
        "source_suite": suite_directory.as_posix(),
        "source_suite_sha256": sha256_file(suite_directory / "suite.json"),
        "source_episodes": [spec["id"] for spec in specs],
        "implementation_sha256": sha256_file(Path(__file__)),
        "samples": len(pooled["errors"]),
        "position_velocity": {"key": "com_range_m", "edges": RANGE_EDGES_M,
                              **_binned(pooled["range_m"], pooled["errors"][:, :6],
                                        RANGE_EDGES_M)},
        "attitude_rate": {"key": "seconds_since_first_estimate", "edges": TIME_EDGES_S,
                          **_binned(pooled["seconds"], pooled["errors"][:, 6:], TIME_EDGES_S)},
    }
    bins = np.digitize(pooled["delivery_seconds"], TIME_EDGES_S)
    model["rejection"] = {
        "key": "seconds_since_first_estimate", "edges": TIME_EDGES_S,
        "deliveries": [int((bins == index).sum()) for index in range(len(TIME_EDGES_S) + 1)],
        "rejections": [int(pooled["rejected"][bins == index].sum())
                       for index in range(len(TIME_EDGES_S) + 1)],
    }
    model["rejection"]["probability"] = [
        rejected / delivered if delivered else 0.0
        for rejected, delivered in zip(model["rejection"]["rejections"],
                                       model["rejection"]["deliveries"])]
    pairs = []
    for sample in samples:
        residuals = _standardized(model, sample)
        consecutive = np.diff(sample["ticks"]) == 10
        pairs.append(np.stack([residuals[:-1][consecutive], residuals[1:][consecutive]], 1))
    pairs = np.concatenate(pairs)
    model["lag1_correlation"] = [float(np.corrcoef(pairs[:, 0, k], pairs[:, 1, k])[0, 1])
                                 for k in range(12)]
    return model


class SurrogateNavigation:
    """Fleet navigation from a fitted surrogate model, for camera-only reference timing:
    captures at whole seconds, delivered 0.2 s later, rejected with the fitted probability
    for the time since the first estimate."""

    def __init__(self, model, count, dt_ns, seed, record=False):
        self.model, self.dt_ns = model, dt_ns
        self.rng = np.random.default_rng(seed)
        self.phi = np.asarray(model["lag1_correlation"])
        self.started = np.zeros(count, bool)
        self.first_ns = np.zeros(count, dtype=np.int64)
        self.capture_time_ns = np.full(count, -1, dtype=np.int64)
        self.updated_ns = np.zeros(count, dtype=np.int64)
        self.residual = np.zeros((count, 12))
        self.errors = np.zeros((count, 12))
        self.recorded = [] if record else None

    def update(self, tick, states, active):
        time_ns = tick * self.dt_ns
        visual = np.full(len(states), NO_UPDATE)
        if tick % 10 == 2:
            capture_ns = time_ns - 2 * self.dt_ns
            since_first = np.where(self.started, (capture_ns - self.first_ns) / 1e9, 0.0)
            rejection = np.asarray(self.model["rejection"]["probability"])[
                np.digitize(since_first, TIME_EDGES_S)]
            accepted = active & (self.rng.random(len(states)) >= rejection)
            noise = self.rng.standard_normal(self.residual.shape)
            first = accepted & ~self.started
            self.started |= accepted
            self.first_ns[first] = capture_ns
            self.residual = np.where(first[:, None], noise, np.where(
                accepted[:, None], self.phi * self.residual + np.sqrt(1 - self.phi**2) * noise,
                self.residual))
            ranges = norm(states[:, 1, :3] - states[:, 0, :3])
            seconds = (time_ns - self.first_ns) / 1e9
            mean_pv, std_pv = _interpolate(self.model["position_velocity"], ranges)
            mean_ar, std_ar = _interpolate(self.model["attitude_rate"], seconds)
            errors = (np.concatenate([mean_pv, mean_ar], -1)
                      + np.concatenate([std_pv, std_ar], -1) * self.residual)
            self.errors[accepted] = errors[accepted]
            self.updated_ns[accepted] = time_ns
            self.capture_time_ns[accepted] = capture_ns
            visual = np.where(accepted, ACCEPTED_UPDATE, np.where(active, FAILED_UPDATE,
                                                                  NO_UPDATE))
            if self.recorded is not None:
                self.recorded.append(self.errors[accepted].copy())
        errors = self.errors.copy()
        errors[:, :3] += errors[:, 3:6] * ((time_ns - self.updated_ns) / 1e9)[:, None]
        target = apply_errors(states[:, 0], errors)
        return Estimates(target, self.started.copy(), self.capture_time_ns.copy(),
                         time_ns - self.capture_time_ns <= FRESH_AGE_NS, visual)
