"""Derived target-local kinematics; physical state stays inertial."""

import numpy as np


def relative_state(positions_m, velocities_mps):
    """Return local state/basis for a central-gravity target, preserving sample axes."""
    positions = np.asarray(positions_m, dtype=np.float64)
    velocities = np.asarray(velocities_mps, dtype=np.float64)
    if positions.shape != velocities.shape or positions.shape[-2:] != (2, 3):
        raise ValueError("positions and velocities must have matching (..., 2, 3) shapes")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("relative state requires finite positions and velocities")
    target_r = positions[..., 0, :]
    target_v = velocities[..., 0, :]
    radius = np.linalg.norm(target_r, axis=-1)
    momentum = np.cross(target_r, target_v)
    momentum_norm = np.linalg.norm(momentum, axis=-1)
    if np.any(radius == 0) or np.any(momentum_norm == 0):
        raise ValueError("target LVLH is undefined for zero radius or angular momentum")
    radial = target_r / radius[..., None]
    normal = momentum / momentum_norm[..., None]
    along_track = np.cross(normal, radial)
    basis_nl = np.stack((radial, along_track, normal), axis=-1)
    rho = np.einsum("...ji,...j->...i", basis_nl, positions[..., 1, :] - target_r)
    inertial_dv_local = np.einsum(
        "...ji,...j->...i", basis_nl, velocities[..., 1, :] - target_v
    )
    omega_local = np.zeros_like(rho)
    omega_local[..., 2] = momentum_norm / radius**2
    rho_dot = inertial_dv_local - np.cross(omega_local, rho)
    return rho, rho_dot, basis_nl


def quaternion_matrices(quaternions_xyzw):
    """Body-to-inertial rotation matrices from finite, nonzero XYZW quaternions."""
    q = np.asarray(quaternions_xyzw, dtype=np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternions must have finite (..., 4) XYZW components")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(norm).all():
        raise ValueError("quaternions must have finite nonzero norms")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def matrix_quaternion(matrix):
    """Convert one proper rotation matrix to a unit XYZW quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    trace = np.trace(rotation)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                scale / 4,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        second, third = (index + 1) % 3, (index + 2) % 3
        scale = 2 * np.sqrt(
            1 + rotation[index, index] - rotation[second, second] - rotation[third, third]
        )
        quaternion = np.empty(4)
        quaternion[index] = scale / 4
        quaternion[second] = (rotation[index, second] + rotation[second, index]) / scale
        quaternion[third] = (rotation[index, third] + rotation[third, index]) / scale
        quaternion[3] = (rotation[third, second] - rotation[second, third]) / scale
    return quaternion / np.linalg.norm(quaternion)


def quaternion_product(first, second):
    """Hamilton product of two XYZW quaternions."""
    first, second = np.asarray(first), np.asarray(second)
    vector = first[3] * second[:3] + second[3] * first[:3]
    vector += np.cross(first[:3], second[:3])
    return np.r_[vector, first[3] * second[3] - first[:3] @ second[:3]]


def port_relative_state(
    positions_m,
    velocities_mps,
    quaternions_xyzw,
    angular_velocities_body_radps,
    docking_ports_body_m,
):
    """Port separation/velocity in COM-defined target LVLH, including lever arms."""
    positions = np.asarray(positions_m, dtype=np.float64)
    velocities = np.asarray(velocities_mps, dtype=np.float64)
    rotations = quaternion_matrices(quaternions_xyzw)
    omega = np.asarray(angular_velocities_body_radps, dtype=np.float64)
    ports = np.asarray(docking_ports_body_m, dtype=np.float64)
    if rotations.shape != positions.shape[:-1] + (3, 3):
        raise ValueError("port quaternion shapes must match spacecraft states")
    if omega.shape != positions.shape or ports.shape != (2, 3):
        raise ValueError("port offsets must be (2, 3) and rates match spacecraft states")
    if not np.isfinite(omega).all() or not np.isfinite(ports).all():
        raise ValueError("port offsets and body rates must be finite")
    rho, rho_dot, basis = relative_state(positions, velocities)
    lever = np.einsum("...bij,bj->...bi", rotations, ports)
    lever_velocity = np.einsum("...bij,...bj->...bi", rotations, np.cross(omega, ports))
    local_lever = np.einsum("...ji,...j->...i", basis, lever[..., 1, :] - lever[..., 0, :])
    local_velocity = np.einsum(
        "...ji,...j->...i", basis, lever_velocity[..., 1, :] - lever_velocity[..., 0, :]
    )
    target_r, target_v = positions[..., 0, :], velocities[..., 0, :]
    frame_rate = np.zeros_like(rho)
    frame_rate[..., 2] = (
        np.linalg.norm(np.cross(target_r, target_v), axis=-1)
        / np.sum(target_r**2, axis=-1)
    )
    return (
        rho + local_lever,
        rho_dot + local_velocity - np.cross(frame_rate, local_lever),
        basis,
    )
