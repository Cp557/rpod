"""Delayed visual pose updates for an inertial target-state error-state EKF."""

from dataclasses import dataclass
import math

import numpy as np

from .frames import matrix_quaternion, quaternion_product
from .observations import NavigationEstimate
from .scenarios import MU_M3_S2

_STEP_NS = 100_000_000
_LAG_NS = 2_000_000_000
_HIGH_NOISE_NS = 10_000_000_000
_POSITION = slice(0, 3)
_VELOCITY = slice(3, 6)
_ATTITUDE = slice(6, 9)
_RATE = slice(9, 12)
_OBSERVED = np.r_[0:3, 6:9]


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_quaternion(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        quaternion = np.r_[0.5 * rotation_vector, 1.0]
    else:
        quaternion = np.r_[rotation_vector * (math.sin(angle / 2) / angle), math.cos(angle / 2)]
    return quaternion / np.linalg.norm(quaternion)


def _rotation_residual(measured, predicted):
    error = quaternion_product(measured, predicted * [-1, -1, -1, 1])
    if error[3] < 0:
        error = -error
    length = np.linalg.norm(error[:3])
    if length < 1e-12:
        return 2 * error[:3]
    return 2 * math.atan2(length, error[3]) * error[:3] / length


def _valid_time(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and 0 <= value <= 2**64 - 1
    )


def _gravity(position):
    radius = np.linalg.norm(position)
    if not np.isfinite(radius) or radius == 0:
        raise ValueError("navigation position must have a finite, nonzero radius")
    return -MU_M3_S2 * position / radius**3


@dataclass
class _FilterState:
    time_ns: int
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    covariance: np.ndarray


@dataclass
class _Measurement:
    time_ns: int
    position: np.ndarray
    quaternion: np.ndarray
    covariance: np.ndarray


def _propagate(state, time_ns, angular_noise):
    """RK4 central gravity and constant spatial angular rate; covariance in N."""
    duration = (time_ns - state.time_ns) / 1e9
    position, velocity = state.position, state.velocity
    k1_r, k1_v = velocity, _gravity(position)
    k2_r = velocity + k1_v * duration / 2
    k2_v = _gravity(position + k1_r * duration / 2)
    k3_r = velocity + k2_v * duration / 2
    k3_v = _gravity(position + k2_r * duration / 2)
    k4_r = velocity + k3_v * duration
    k4_v = _gravity(position + k3_r * duration)
    next_position = position + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * duration / 6
    next_velocity = velocity + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * duration / 6
    increment = _rotation_quaternion(state.angular_velocity * duration)
    next_quaternion = quaternion_product(increment, state.quaternion)
    next_quaternion /= np.linalg.norm(next_quaternion)

    midpoint = (position + next_position) / 2
    radius = np.linalg.norm(midpoint)
    gradient = MU_M3_S2 * (
        3 * np.outer(midpoint, midpoint) / radius**5 - np.eye(3) / radius**3
    )
    dynamics = np.zeros((12, 12))
    dynamics[_POSITION, _VELOCITY] = np.eye(3)
    dynamics[_VELOCITY, _POSITION] = gradient
    dynamics[_ATTITUDE, _ATTITUDE] = _skew(state.angular_velocity)
    dynamics[_ATTITUDE, _RATE] = np.eye(3)
    scaled = dynamics * duration
    squared = scaled @ scaled
    transition = np.eye(12) + scaled + squared / 2 + squared @ scaled / 6 + squared @ squared / 24
    process = np.zeros((12, 12))
    for first, second, spectral_density in (
        (_POSITION, _VELOCITY, 2.5e-5),
        (_ATTITUDE, _RATE, angular_noise),
    ):
        process[first, first] = np.eye(3) * spectral_density * duration**3 / 3
        process[first, second] = np.eye(3) * spectral_density * duration**2 / 2
        process[second, first] = process[first, second]
        process[second, second] = np.eye(3) * spectral_density * duration
    covariance = transition @ state.covariance @ transition.T + process
    covariance = (covariance + covariance.T) / 2
    if not all(np.isfinite(array).all() for array in (
        next_position, next_velocity, next_quaternion, covariance
    )):
        raise ValueError("navigation propagation produced a nonfinite state")
    return _FilterState(
        time_ns, next_position, next_velocity, next_quaternion,
        state.angular_velocity.copy(), covariance,
    )


def _correct(state, measurement, gate=True):
    residual = np.r_[
        measurement.position - state.position,
        _rotation_residual(measurement.quaternion, state.quaternion),
    ]
    innovation = state.covariance[np.ix_(_OBSERVED, _OBSERVED)] + measurement.covariance
    weighted = np.linalg.solve(innovation, residual)
    if gate and residual @ weighted > 30.0:
        return None
    gain = np.linalg.solve(innovation, state.covariance[:, _OBSERVED].T).T
    correction = gain @ residual
    quaternion = quaternion_product(
        _rotation_quaternion(correction[_ATTITUDE]), state.quaternion
    )
    quaternion /= np.linalg.norm(quaternion)
    remainder = np.eye(12)
    remainder[:, _OBSERVED] -= gain
    covariance = (
        remainder @ state.covariance @ remainder.T
        + gain @ measurement.covariance @ gain.T
    )
    reset = np.eye(12)
    reset[_ATTITUDE, _ATTITUDE] += 0.5 * _skew(correction[_ATTITUDE])
    covariance = reset @ covariance @ reset.T
    return _FilterState(
        state.time_ns,
        state.position + correction[_POSITION],
        state.velocity + correction[_VELOCITY],
        quaternion,
        state.angular_velocity + correction[_RATE],
        (covariance + covariance.T) / 2,
    )


class VisualNavigation:
    """Target r/v, left attitude error, and spatial angular rate from delivered poses.

    The two-second history supports delayed and out-of-order pose updates. Accepted
    later measurements are replayed after an earlier update. Public estimates never
    precede the latest observation's availability, even though correction occurs at
    capture time. Process-noise constants are continuous acceleration spectral
    densities; their integrated r/v and attitude/rate blocks are discretized per step.
    """

    # Target velocity and angular-rate priors at initialization (velocity starts at own
    # velocity, rate at zero), and the rotation process noise (continuous angular-acceleration
    # spectral densities) after and during acquisition. Variants for tumbling targets override
    # these.
    initial_velocity_sigma_mps = 0.1
    initial_rate_sigma_radps = 0.005
    rotation_spectral_density = 4e-8
    acquisition_rotation_spectral_density = 2.5e-5

    def __init__(self, camera=None, calibration=None):
        self.camera = camera
        self._range_bins = np.array([1e9])
        self._translation_variance = np.zeros((1, 3))
        self._rotation_variance = np.zeros((1, 3))
        if calibration is not None:
            bins = np.asarray(calibration["range_bins"], dtype=np.float64)
            translation = np.asarray(calibration["translation_variance"], dtype=np.float64)
            rotation = np.asarray(calibration["rotation_variance"], dtype=np.float64)
            if (
                bins.ndim != 1 or len(bins) == 0
                or translation.shape != (len(bins), 3)
                or rotation.shape != (len(bins), 3)
                or not np.isfinite(bins).all() or np.any(bins <= 0)
                or np.any(np.diff(bins) <= 0)
                or not np.isfinite(translation).all() or np.any(translation < 0)
                or not np.isfinite(rotation).all() or np.any(rotation < 0)
            ):
                raise ValueError("invalid visual-navigation pose calibration")
            self._range_bins = bins.copy()
            self._translation_variance = translation.copy()
            self._rotation_variance = rotation.copy()
        self.reset()

    def reset(self):
        self._current = None
        self._history = []
        self._measurements = {}
        self._last_capture_ns = None
        self._available_ns = None
        self._high_noise_intervals = []

    def mark_reacquired(self):
        if self._current is not None:
            start = self._current.time_ns
            self._high_noise_intervals.append((start, start + _HIGH_NOISE_NS))

    def _path(self, state, time_ns, high_noise_intervals=None):
        intervals = (
            self._high_noise_intervals
            if high_noise_intervals is None
            else high_noise_intervals
        )
        path = []
        while state.time_ns < time_ns:
            stop = min(time_ns, state.time_ns + _STEP_NS)
            for start, end in intervals:
                if state.time_ns < start < stop:
                    stop = start
                if state.time_ns < end < stop:
                    stop = end
            boosted = any(start <= state.time_ns < end for start, end in intervals)
            state = _propagate(state, stop, self.acquisition_rotation_spectral_density
                               if boosted else self.rotation_spectral_density)
            path.append(state)
        return path

    def _state_at(self, time_ns):
        if not self._history or time_ns < self._history[0].time_ns:
            return None
        base = next(state for state in reversed(self._history) if state.time_ns <= time_ns)
        path = self._path(base, time_ns)
        return path[-1] if path else base

    def _prune(self):
        cutoff = self._current.time_ns - _LAG_NS
        anchor = 0
        for index, state in enumerate(self._history):
            if state.time_ns <= cutoff:
                anchor = index
        self._history = self._history[anchor:]
        oldest = self._history[0].time_ns
        self._measurements = {
            time: measurement for time, measurement in self._measurements.items() if time >= oldest
        }
        self._high_noise_intervals = [
            interval for interval in self._high_noise_intervals if interval[1] > oldest
        ]

    def _measurement(self, pose, own):
        from .camera import camera_pose

        translation = np.asarray(pose.translation_ct, dtype=np.float64)
        rotation = np.asarray(pose.rotation_ct, dtype=np.float64)
        own_position = np.asarray(own.position_m, dtype=np.float64)
        own_velocity = np.asarray(own.velocity_mps, dtype=np.float64)
        own_quaternion = np.asarray(own.quaternion_xyzw, dtype=np.float64)
        if (
            translation.shape != (3,) or rotation.shape != (3, 3)
            or own_position.shape != (3,) or own_velocity.shape != (3,)
            or own_quaternion.shape != (4,)
            or not all(np.isfinite(array).all() for array in (
                translation, rotation, own_position, own_velocity, own_quaternion
            ))
            or translation[2] <= 0
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError("invalid visual pose or capture-time own navigation")
        camera_position, camera_rotation = camera_pose(own, self.camera)
        camera_rotation = np.asarray(camera_rotation, dtype=np.float64)
        position = camera_position + camera_rotation @ translation
        quaternion = matrix_quaternion(camera_rotation @ rotation)
        index = min(
            int(np.searchsorted(self._range_bins, translation[2])), len(self._range_bins) - 1
        )
        translation_variance = np.maximum(
            self._translation_variance[index], np.array([0.002, 0.002, 0.01])**2
        )
        rotation_variance = np.maximum(
            self._rotation_variance[index], math.radians(0.25)**2
        )
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = camera_rotation @ np.diag(translation_variance) @ camera_rotation.T
        covariance[3:, 3:] = camera_rotation @ np.diag(rotation_variance) @ camera_rotation.T
        return _Measurement(int(pose.capture_time_ns), position, quaternion, covariance)

    def observe(self, pose, own_at_capture):
        if (
            not pose.valid
            or not _valid_time(pose.capture_time_ns)
            or not _valid_time(pose.delivery_time_ns)
            or pose.delivery_time_ns < pose.capture_time_ns
            or own_at_capture.time_ns != pose.capture_time_ns
        ):
            return False
        capture = int(pose.capture_time_ns)
        now = max(int(pose.delivery_time_ns), self._current.time_ns if self._current else 0)
        if capture < now - _LAG_NS or capture in self._measurements:
            return False
        try:
            measurement = self._measurement(pose, own_at_capture)
            intervals = self._high_noise_intervals
            if self._current is None:
                covariance = np.zeros((12, 12))
                covariance[np.ix_(_OBSERVED, _OBSERVED)] = measurement.covariance
                covariance[_VELOCITY, _VELOCITY] = np.eye(3) * self.initial_velocity_sigma_mps**2
                covariance[_RATE, _RATE] = np.eye(3) * self.initial_rate_sigma_radps**2
                state = _FilterState(
                    capture, measurement.position,
                    np.asarray(own_at_capture.velocity_mps, dtype=np.float64).copy(),
                    measurement.quaternion, np.zeros(3), covariance,
                )
                intervals = [(capture, capture + _HIGH_NOISE_NS)]
                history = [state]
            else:
                state = self._state_at(capture)
                if state is None:
                    return False
                state = _correct(state, measurement)
                if state is None:
                    return False
                history = [entry for entry in self._history if entry.time_ns < capture] + [state]
                for time in sorted(self._measurements):
                    if time <= capture:
                        continue
                    path = self._path(state, time)
                    history.extend(path)
                    state = _correct(
                        path[-1] if path else state, self._measurements[time], gate=False
                    )
                    history[-1] = state
            path = self._path(state, now, intervals)
            history.extend(path)
            state = path[-1] if path else state
            if not all(np.isfinite(array).all() for array in (
                state.position, state.velocity, state.quaternion,
                state.angular_velocity, state.covariance,
            )):
                return False
            _gravity(state.position)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False
        self._current = state
        self._history = history
        self._measurements[capture] = measurement
        self._high_noise_intervals = intervals
        self._last_capture_ns = max(capture, self._last_capture_ns or 0)
        self._available_ns = now
        self._prune()
        return True

    def predict(self, time_ns):
        if not _valid_time(time_ns):
            raise ValueError("navigation time must be a nonnegative u64 nanosecond timestamp")
        time_ns = int(time_ns)
        if self._current is None or time_ns < self._available_ns:
            return None
        if time_ns > self._current.time_ns:
            path = self._path(self._current, time_ns)
            self._history.extend(path)
            self._current = path[-1]
            self._prune()
            state = self._current
        else:
            state = self._state_at(time_ns)
            if state is None:
                return None
        age_ns = time_ns - self._last_capture_ns
        if age_ns <= 1_200_000_000:
            health = "fresh"
        elif age_ns <= 3_000_000_000:
            health = "predicted"
        else:
            health = "lost"
        return NavigationEstimate(
            time_ns, state.position.copy(), state.velocity.copy(), state.quaternion.copy(),
            state.angular_velocity.copy(), state.covariance.copy(),
            self._last_capture_ns, health,
        )
