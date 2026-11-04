"""Learned approach policy: a small NumPy MLP from estimated port state to LVLH acceleration.

The weights are one flat vector, so evolution strategies can perturb them. `act` evaluates one
vector for every row, or one vector per group of rows during training.
"""

import json
from pathlib import Path

import numpy as np

HIDDEN = 64
OBSERVATION_SIZE = 13
LAYERS = ((OBSERVATION_SIZE, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, 3))
PARAMETER_COUNT = sum(inputs * outputs + outputs for inputs, outputs in LAYERS)
# Per LVLH axis; 2 N on the 100 kg chaser allows 0.02 m/s^2 in norm.
ACCELERATION_LIMIT_MPS2 = 0.01


def observation(m, goal_gap_m, age_s):
    """(n, 13) features: port offset from the hold point and port velocity, each at an
    approach scale and a clipped hold-point scale, and the estimate age."""
    offset = m["port_r"].copy()
    offset[:, 1] += goal_gap_m
    velocity = m["port_v"]
    return np.concatenate([offset / 10.0, np.clip(offset, -1.0, 1.0), velocity / 0.1,
                           np.clip(velocity / 0.02, -1.0, 1.0), (age_s / 3.0)[:, None]],
                          axis=-1)


def initial_parameters(seed):
    """Scaled normal weights, zero biases and a small output layer, so actions start near 0."""
    rng = np.random.default_rng(seed)
    parts = []
    for index, (inputs, outputs) in enumerate(LAYERS):
        scale = (0.1 if index == len(LAYERS) - 1 else 1.0) / np.sqrt(inputs)
        parts += [rng.normal(0, scale, inputs * outputs), np.zeros(outputs)]
    return np.concatenate(parts)


def act(parameters, observations):
    """LVLH accelerations.

    `parameters` is one (P,) vector for (n, 7) observations, or (G, P) vectors for (G, k, 7)
    observations, one vector per group of k rows.
    """
    x, offset = observations, 0
    lead = parameters.shape[:-1]
    for index, (inputs, outputs) in enumerate(LAYERS):
        weight = parameters[..., offset:offset + inputs * outputs].reshape(*lead, inputs, outputs)
        offset += inputs * outputs
        bias = parameters[..., offset:offset + outputs].reshape(*lead, 1, outputs) if lead \
            else parameters[offset:offset + outputs]
        offset += outputs
        x = np.matmul(x, weight) + bias
        if index < len(LAYERS) - 1:
            x = np.tanh(x)
    return ACCELERATION_LIMIT_MPS2 * np.tanh(x)


def save_policy(path, parameters, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, parameters)
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")


def load_policy(path):
    parameters = np.load(path)
    if parameters.shape != (PARAMETER_COUNT,) or not np.isfinite(parameters).all():
        raise ValueError(f"a policy needs {PARAMETER_COUNT} finite parameters")
    return parameters
