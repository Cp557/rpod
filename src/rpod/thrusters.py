"""M14: the chaser's thrusters, their faults, and onboard fault detection.

Sixteen 1 N thrusters in four corner pods (the showcase layout) replace the ideal body wrench.
Each tick the onboard thruster driver, below the controller's wrench interface:
- bounds the requested wrench to M13's 2 N and 0.5 N m, then allocates it with a linear
  program over the healthy thrusters: the largest scale s <= 1 of the request, then the least
  total on-time (propellant);
- turns duty cycles into on-times with a per-thruster accumulator that fires once the 10 ms
  minimum on-time has built up;
- compares the force and torque an IMU measures with what its commands should have produced,
  and tests that residual against every single-thruster fault, stuck on or stuck off, over a
  1 s window (a bank of generalized likelihood ratios). A detected fault is isolated: that
  thruster's valve closes and allocation drops it.

The hardware differs from the driver's nominal model: each thruster has a seeded thrust error
and misalignment, and one may fail. A tick's force and torque are the impulse-equivalent
averages of the actual on-times. Mass stays constant; a docking burns about 0.1 kg.
Monitoring stops at the latch, where contact and the stack's inertia break the free-flyer
model the residual assumes.
"""

from collections import deque
from functools import partial
import math

import numpy as np
from scipy.optimize import linprog

from .docking import DockingEnv, _bounded
from .natural_lidar import exponential

THRUST_N, SPECIFIC_IMPULSE_S, STANDARD_GRAVITY_MPS2 = 1.0, 220.0, 9.80665
MINIMUM_ON_TIME_S = 0.010
# Pods at (+-0.36, 0.1, +-0.36) m; each fires outward along x and z, and both ways along y.
POD_XZ_M, POD_Y_M = 0.36, 0.1
COUNT = 16
# Per-thruster hardware errors: thrust scale uniform within +-3%, and a tilt of up to 0.5
# degrees about a random axis across the nominal direction.
THRUST_ERROR, MISALIGNMENT_DEG = 0.03, 0.5
# The request is bounded as the ideal engine bounded it in M13.
REQUEST_FORCE_N, REQUEST_TORQUE_NM = 2.0, 0.5
# One unit of wrench scale outweighs any total on-time (at most 16 thruster-ticks).
SCALE_WEIGHT = 1000.0
# A navigation-grade IMU at the chaser's center of mass: accelerometer white noise 10 ug/rtHz
# and a 10 ug residual bias; gyro angle random walk 0.01 deg/rth and a 0.01 deg/h bias.
ACCEL_NOISE_MPS2_RTHZ = 10e-6 * STANDARD_GRAVITY_MPS2
ACCEL_BIAS_MPS2 = 10e-6 * STANDARD_GRAVITY_MPS2
GYRO_NOISE_RAD_RTS = math.radians(0.01) / 60
GYRO_BIAS_RADPS = math.radians(0.01) / 3600
# Residual allowance per commanded thruster for its unknown thrust error and misalignment.
MODEL_ERROR = 0.04
WINDOW_TICKS = 10
# A fault hits the candidate thruster that fired longest over this many ticks before it.
FAULT_LOOKBACK_TICKS = 100
# Log-likelihood-ratio threshold (chi-square units): far above what noise and model error
# reach, below a single 10 ms pulse of a stuck-off thruster.
DETECTION_THRESHOLD = 50.0


def layout():
    """Positions (m) and unit force directions of the 16 thrusters in chaser body axes, in the
    showcase worker's nozzle order: pods x, z in (-1, 1), x outer; in each pod, exhaust along
    (x, 0, 0), (0, 0, z), (0, -1, 0), (0, 1, 0). Exhaust along d pushes the chaser along -d."""
    positions, directions = [], []
    for x in (-1, 1):
        for z in (-1, 1):
            for exhaust in ((x, 0, 0), (0, 0, z), (0, -1, 0), (0, 1, 0)):
                positions.append((POD_XZ_M * x, POD_Y_M, POD_XZ_M * z))
                directions.append(-np.asarray(exhaust, dtype=np.float64))
    return np.array(positions), np.array(directions)


def wrench_matrix(positions, forces):
    """6 x n: each thruster's force and its torque about the center of mass."""
    return np.vstack([forces.T, np.cross(positions, forces).T])


def nominal_matrix():
    positions, directions = layout()
    return wrench_matrix(positions, THRUST_N * directions)


def hardware_matrix(stream):
    """The true wrench matrix: each thruster's seeded thrust error and misalignment."""
    rng = np.random.default_rng(stream)
    positions, directions = layout()
    scale = 1 + rng.uniform(-THRUST_ERROR, THRUST_ERROR, COUNT)
    tilted = []
    for direction in directions:
        axis = np.cross(direction, rng.normal(size=3))
        angle = math.radians(rng.uniform(0, MISALIGNMENT_DEG))
        tilted.append(exponential(axis / np.linalg.norm(axis) * angle) @ direction)
    return wrench_matrix(positions, THRUST_N * scale[:, None] * np.array(tilted))


def wrench_bounds():
    """Force (N) and torque (N m) norms no thruster combination can exceed, errors included."""
    positions, _ = layout()
    peak = THRUST_N * (1 + THRUST_ERROR)
    return COUNT * peak, peak * float(np.linalg.norm(positions, axis=1).sum())


class AllocationError(Exception):
    """The allocation program failed; it is always feasible (u = 0, s = 0), so this is a bug.
    Not a ValueError, which the environment records as a numerical-failure outcome."""


def allocate(matrix, wrench):
    """Duty cycles in [0, 1] for the columns of `matrix` and the scale s <= 1 of `wrench` they
    produce: the largest s first, then the least total duty."""
    count = matrix.shape[1]
    if not np.any(wrench):
        return np.zeros(count), 1.0
    result = linprog(np.r_[np.ones(count), -SCALE_WEIGHT],
                     A_eq=np.hstack([matrix, -wrench[:, None]]), b_eq=np.zeros(6),
                     bounds=[(0.0, 1.0)] * (count + 1), method="highs")
    if result.status != 0:
        raise AllocationError(result.message)
    duty = np.clip(result.x[:count], 0.0, 1.0)
    duty[duty < 1e-9] = 0.0
    return duty, float(result.x[count])


def axis_authority(matrix):
    """The largest force (N) or torque (N m) along each body axis, by sign, with the other five
    wrench components zero: (6, 2) for +, -."""
    count, authority = matrix.shape[1], np.zeros((6, 2))
    for axis in range(6):
        for column, sign in enumerate((1.0, -1.0)):
            unit = np.zeros(6)
            unit[axis] = sign
            result = linprog(np.r_[np.zeros(count), -1.0],
                             A_eq=np.hstack([matrix, -unit[:, None]]), b_eq=np.zeros(6),
                             bounds=[(0.0, 1.0)] * count + [(0.0, None)], method="highs")
            authority[axis, column] = result.x[count] if result.status == 0 else 0.0
    return authority


def modulate(duty, accumulated, dt_s):
    """On-times (s) for one tick, and the carried accumulation. Each thruster accumulates its
    commanded on-time and fires once that reaches the minimum on-time, for at most the tick;
    one commanded off drops what it had accumulated."""
    accumulated = np.where(duty > 0, accumulated + duty * dt_s, 0.0)
    on = np.where(accumulated >= MINIMUM_ON_TIME_S, np.minimum(accumulated, dt_s), 0.0)
    return on, accumulated - on


class Imu:
    """Per tick, the velocity change from thrust (no gravity: the accelerometer is in free
    fall) and the body rate, each with white noise and a fixed seeded bias, in body axes."""

    def __init__(self, stream, dt_s):
        self.rng, self.dt_s = np.random.default_rng(stream), dt_s

        def bias(magnitude):
            direction = self.rng.normal(size=3)
            return magnitude * direction / np.linalg.norm(direction)

        self.accel_bias, self.gyro_bias = bias(ACCEL_BIAS_MPS2), bias(GYRO_BIAS_RADPS)

    def measure(self, force_body_n, mass_kg, rate_body_radps):
        delta_v = ((np.asarray(force_body_n) / mass_kg + self.accel_bias) * self.dt_s
                   + ACCEL_NOISE_MPS2_RTHZ * math.sqrt(self.dt_s) * self.rng.normal(size=3))
        rate = (np.asarray(rate_body_radps) + self.gyro_bias
                + GYRO_NOISE_RAD_RTS / math.sqrt(self.dt_s) * self.rng.normal(size=3))
        return delta_v, rate


class FaultMonitor:
    """Tests the IMU residual against every single-thruster fault.

    Each tick's residual is the measured force and torque (from the velocity change, and from
    the rate change through Euler's equation) less the nominal wrench of the commanded
    on-times. A thruster stuck on adds its full wrench beyond its command; one stuck off
    removes its commanded wrench. Each hypothesis' log-likelihood ratio against no fault is
    sum((2 r p - p^2) / sigma^2) from the fault's onset to now, maximized over onsets within
    the window, with sigma the sensor noise plus `MODEL_ERROR` of each commanded thruster's
    wrench. Because the hypotheses follow the command history and the onset, a thruster stuck
    off is told from its co-located opposite nozzle stuck on (their wrenches are exact
    negatives) unless the first was commanded fully on throughout.
    """

    def __init__(self, matrix, mass_kg, inertia_kg_m2, dt_s):
        self.matrix, self.mass, self.dt_s = matrix, mass_kg, dt_s
        self.inertia = np.asarray(inertia_kg_m2, dtype=np.float64)
        self.sensor_sigma = np.r_[
            [mass_kg * ACCEL_NOISE_MPS2_RTHZ / math.sqrt(dt_s)] * 3,
            math.sqrt(2) * self.inertia * GYRO_NOISE_RAD_RTS / math.sqrt(dt_s) / dt_s]
        self.history = deque(maxlen=WINDOW_TICKS)
        self.previous_rate = None

    def clear(self):
        self.history.clear()

    def update(self, on_s, delta_v, rate, healthy):
        """Add one tick; return (statistic, thruster, kind) for the likeliest fault."""
        previous, self.previous_rate = self.previous_rate, rate
        if previous is None:
            return 0.0, None, None
        mean = (rate + previous) / 2
        measured = np.r_[self.mass * delta_v / self.dt_s,
                         self.inertia * (rate - previous) / self.dt_s
                         + np.cross(mean, self.inertia * mean)]
        duty = on_s / self.dt_s
        sigma = np.hypot(self.sensor_sigma, MODEL_ERROR * np.abs(self.matrix) @ duty)
        self.history.append((measured - self.matrix @ duty, duty, sigma))
        residuals, duties, sigmas = (np.array(column) for column in zip(*self.history))
        weight = 1 / sigmas ** 2
        terms = np.array([
            np.einsum("kc,kcn->kn", weight, 2 * residuals[:, :, None] * predicted
                      - predicted ** 2)
            for predicted in (self.matrix[None] * (1 - duties)[:, None, :],   # stuck on
                              -self.matrix[None] * duties[:, None, :])])      # stuck off
        # Sums from each candidate onset to the latest tick; the likeliest onset per hypothesis.
        statistics = np.cumsum(terms[:, ::-1], axis=1).max(axis=1)
        statistics = np.where(healthy, statistics, -np.inf)
        kind, thruster = np.unravel_index(np.argmax(statistics), statistics.shape)
        return float(statistics[kind, thruster]), int(thruster), ("stuck_on", "stuck_off")[kind]


class Thrusters:
    """One episode's thruster set: the onboard driver (allocation, pulse modulation, fault
    detection and isolation on the nominal model) and the hardware it drives.

    `fault` is None or {"kind": "stuck_on" | "stuck_off", "candidates": [index, ...],
    "trigger_gap_m"}; the environment calls `fail` when the true port gap reaches the trigger,
    and the candidate that fired longest over the preceding `FAULT_LOOKBACK_TICKS` fails (the
    lowest index on a tie), so the fault hits a thruster in use. Per-tick rows accumulate in
    `log` for `thrusters.npz`.
    """

    def __init__(self, error_stream, imu_stream, mass_kg, inertia_kg_m2, dt_s, fault=None):
        self.nominal, self.hardware = nominal_matrix(), hardware_matrix(error_stream)
        self.mass, self.dt_s, self.fault = mass_kg, dt_s, fault
        self.imu, self.monitor = Imu(imu_stream, dt_s), FaultMonitor(self.nominal, mass_kg,
                                                                     inertia_kg_m2, dt_s)
        self.healthy = np.ones(COUNT, bool)       # onboard: not isolated
        self.accumulated = np.zeros(COUNT)
        self.failed_tick = self.failed_thruster = None
        self.isolations, self._cache = [], {}
        self.log = {key: [] for key in ("commanded_on_s", "actual_on_s", "scale", "statistic",
                                        "monitoring")}

    def fail(self, tick):
        candidates = self.fault["candidates"]
        recent = self.records()["commanded_on_s"][-FAULT_LOOKBACK_TICKS:, candidates].sum(axis=0)
        self.failed_tick, self.failed_thruster = tick, int(candidates[int(np.argmax(recent))])

    def fire(self, force_body_n, torque_body_nm):
        """The actual chaser force and torque for one tick of the requested wrench."""
        request = np.r_[_bounded(np.asarray(force_body_n, dtype=np.float64), REQUEST_FORCE_N),
                        _bounded(np.asarray(torque_body_nm, dtype=np.float64),
                                 REQUEST_TORQUE_NM)]
        key = request.tobytes()
        if key not in self._cache:
            duty, scale = allocate(self.nominal[:, self.healthy], request)
            full = np.zeros(COUNT)
            full[self.healthy] = duty
            self._cache[key] = full, scale
        duty, scale = self._cache[key]
        commanded, self.accumulated = modulate(duty, self.accumulated, self.dt_s)
        actual = commanded.copy()
        if self.failed_tick is not None:
            actual[self.failed_thruster] = (self.dt_s if self.fault["kind"] == "stuck_on"
                                            else 0.0)
        actual[~self.healthy] = 0.0                # closed valves
        self.commanded, self.actual = commanded, actual
        self.log["commanded_on_s"].append(commanded)
        self.log["actual_on_s"].append(actual)
        self.log["scale"].append(scale)
        wrench = self.hardware @ actual / self.dt_s
        self.output = wrench
        return wrench[:3], wrench[3:]

    def observe(self, rate_body_radps, tick, monitoring):
        """After the engine step: measure with the IMU and test for a fault."""
        statistic = 0.0
        if monitoring:
            delta_v, rate = self.imu.measure(self.output[:3], self.mass, rate_body_radps)
            statistic, thruster, kind = self.monitor.update(self.commanded, delta_v, rate,
                                                            self.healthy)
            if statistic > DETECTION_THRESHOLD:
                self.healthy[thruster] = False
                self.accumulated[thruster] = 0.0
                self._cache.clear()
                self.monitor.clear()
                self.isolations.append({"tick": tick, "thruster": thruster, "kind": kind,
                                        "statistic": statistic})
        self.log["statistic"].append(statistic)
        self.log["monitoring"].append(monitoring)

    def propellant_kg(self):
        thrust = np.linalg.norm(self.hardware[:3], axis=0)
        return float(np.sum(self.records()["actual_on_s"] @ thrust)
                     / (SPECIFIC_IMPULSE_S * STANDARD_GRAVITY_MPS2))

    def records(self):
        """Arrays for `thrusters.npz`, one row per physics step."""
        return {"commanded_on_s": np.asarray(self.log["commanded_on_s"]).reshape(-1, COUNT),
                "actual_on_s": np.asarray(self.log["actual_on_s"]).reshape(-1, COUNT),
                "scale": np.asarray(self.log["scale"], dtype=np.float64),
                "statistic": np.asarray(self.log["statistic"], dtype=np.float64),
                "monitoring": np.asarray(self.log["monitoring"], dtype=np.bool_),
                "nominal_matrix": self.nominal, "hardware_matrix": self.hardware}


class ThrusterEngine:
    """The chaser's engine (`RigidBodyEngine`, or `StackEngine` once latched) driven through
    the thrusters: the requested chaser wrench fires them and the engine applies their actual
    output. The manifest's chaser limits are the thruster set's bounds, so it never clips."""

    def __init__(self, engine, thrusters, monitoring):
        self.engine, self.thrusters, self.monitoring = engine, thrusters, monitoring

    def snapshot(self):
        return self.engine.snapshot()

    def step(self, forces_body_n, torques_body_nm):
        forces = np.array(forces_body_n, dtype=np.float64)
        torques = np.array(torques_body_nm, dtype=np.float64)
        forces[1], torques[1] = self.thrusters.fire(forces[1], torques[1])
        after = self.engine.step(forces, torques)
        self.thrusters.observe(np.asarray(after.angular_velocities_body_radps[1]),
                               int(after.tick), self.monitoring)
        return after


class ThrusterDockingEnv(DockingEnv):
    """`DockingEnv` with the chaser flown on `thrusters`: its engine is wrapped at reset and
    again at the latch, its fault starts at the first tick whose true port gap is at most the
    trigger, and each isolation is recorded as a `safety_detection` event."""

    def __init__(self, camera, mesh, lidar_calibration, mechanism, thrusters, keypoints=None):
        super().__init__(camera, mesh, lidar_calibration, keypoints, mechanism)
        self.thrusters = thrusters

    def reset(self, initial, output):
        self.reported = 0
        return super().reset(initial, output)

    def _boundary(self, snapshot, metrics):
        reason = super()._boundary(snapshot, metrics)
        thrusters, fault = self.thrusters, self.thrusters.fault
        if (fault is not None and thrusters.failed_tick is None and not self.mechanism.latched
                and metrics["gap"] <= fault["trigger_gap_m"]):
            thrusters.fail(int(snapshot.tick))
        if not isinstance(self.engine, ThrusterEngine):
            self.engine = ThrusterEngine(self.engine, thrusters,
                                         monitoring=not self.mechanism.latched)
        for isolation in thrusters.isolations[self.reported:]:
            self.events.append({"tick": isolation["tick"],
                                "time_ns": isolation["tick"] * self.initial["dt_ns"],
                                "kind": "safety_detection",
                                "reason": f"thruster_{isolation['thruster']:02d}_"
                                          f"{isolation['kind']}_isolated"})
        self.reported = len(thrusters.isolations)
        return reason


def thruster_docking_env(mesh, lidar_calibration, mechanism, thrusters, keypoints=None):
    """An `env_class` for `visual_run.run_episode`."""
    return partial(ThrusterDockingEnv, mesh=mesh, lidar_calibration=lidar_calibration,
                   mechanism=mechanism, thrusters=thrusters, keypoints=keypoints)
