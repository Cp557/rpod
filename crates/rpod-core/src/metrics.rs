use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::{OrbitError, STATE_WIDTH};

/// Relative geometry of one target-chaser pair, as Python's `approach.state_metrics`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PairMetrics {
    /// Chaser center of mass relative to the target, in target LVLH.
    pub rho_m: [f64; 3],
    pub rho_dot_mps: [f64; 3],
    /// LVLH basis in inertial axes, row-major, with radial, along-track and normal columns.
    pub basis: [f64; 9],
    /// Body-to-inertial rotations of the target then the chaser, row-major.
    pub rotations: [[f64; 9]; 2],
    /// Chaser port relative to the target port, in target LVLH, with lever-arm velocities.
    pub port_r_m: [f64; 3],
    pub port_v_mps: [f64; 3],
    pub alignment_deg: f64,
    pub relative_rate_radps: f64,
}

fn vector(state: &[f64; STATE_WIDTH], offset: usize) -> Vector3<f64> {
    Vector3::new(state[offset], state[offset + 1], state[offset + 2])
}

/// The same element formulas as `frames.quaternion_matrices`.
fn rotation(state: &[f64; STATE_WIDTH], offset: usize) -> Result<Matrix3<f64>, OrbitError> {
    let q = &state[offset..offset + 4];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(OrbitError::InvalidState(
            "quaternions must have finite nonzero norms",
        ));
    }
    let [x, y, z, w] = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    Ok(Matrix3::new(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ))
}

fn row_major(matrix: &Matrix3<f64>) -> [f64; 9] {
    std::array::from_fn(|index| matrix[(index / 3, index % 3)])
}

/// Geometry of one (target, chaser) state row with the given body-frame docking ports.
pub fn pair_metrics(
    state: &[f64; STATE_WIDTH],
    ports_body_m: &[[f64; 3]; 2],
) -> Result<PairMetrics, OrbitError> {
    if !state.iter().all(|value| value.is_finite()) {
        return Err(OrbitError::InvalidState("state components must be finite"));
    }
    let (target_r, target_v) = (vector(state, 0), vector(state, 3));
    let radius = target_r.norm();
    let momentum = target_r.cross(&target_v);
    let momentum_norm = momentum.norm();
    if radius == 0.0 || momentum_norm == 0.0 {
        return Err(OrbitError::InvalidState(
            "target LVLH is undefined for zero radius or angular momentum",
        ));
    }
    let radial = target_r / radius;
    let normal = momentum / momentum_norm;
    let basis = Matrix3::from_columns(&[radial, normal.cross(&radial), normal]);
    let frame_rate = Vector3::new(0.0, 0.0, momentum_norm / (radius * radius));
    let rho = basis.tr_mul(&(vector(state, 13) - target_r));
    let rho_dot = basis.tr_mul(&(vector(state, 16) - target_v)) - frame_rate.cross(&rho);
    let rotations = [rotation(state, 6)?, rotation(state, 19)?];
    let rates = [vector(state, 10), vector(state, 23)];
    let ports = ports_body_m.map(Vector3::from);
    let lever = [rotations[0] * ports[0], rotations[1] * ports[1]];
    let lever_velocity = [
        rotations[0] * rates[0].cross(&ports[0]),
        rotations[1] * rates[1].cross(&ports[1]),
    ];
    let local_lever = basis.tr_mul(&(lever[1] - lever[0]));
    let local_velocity = basis.tr_mul(&(lever_velocity[1] - lever_velocity[0]));
    let port_r = rho + local_lever;
    let port_v = rho_dot + local_velocity - frame_rate.cross(&local_lever);
    let cosine = ((rotations[0].component_mul(&rotations[1]).sum() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let relative_rate = (rotations[1] * rates[1] - rotations[0] * rates[0]).norm();
    Ok(PairMetrics {
        rho_m: rho.into(),
        rho_dot_mps: rho_dot.into(),
        basis: row_major(&basis),
        rotations: rotations.map(|matrix| row_major(&matrix)),
        port_r_m: port_r.into(),
        port_v_mps: port_v.into(),
        alignment_deg: cosine.acos().to_degrees(),
        relative_rate_radps: relative_rate,
    })
}

/// Values per row written by `write_metrics`: rho, rho_dot, basis (9), target and chaser
/// rotations (18), port_r, port_v, alignment in degrees and relative rate.
pub const METRIC_WIDTH: usize = 41;

/// `pair_metrics` for many rows on rayon's global pool, written into `out` (`METRIC_WIDTH`
/// values per row). Fails on the first invalid row.
pub fn write_metrics(
    states: &[[f64; STATE_WIDTH]],
    ports_body_m: &[[f64; 3]; 2],
    out: &mut [f64],
) -> Result<(), OrbitError> {
    if out.len() != states.len() * METRIC_WIDTH {
        return Err(OrbitError::InvalidInput(
            "the metric buffer needs METRIC_WIDTH values per row",
        ));
    }
    out.par_chunks_exact_mut(METRIC_WIDTH)
        .zip(states)
        .try_for_each(|(row, state)| {
            let metrics = pair_metrics(state, ports_body_m)?;
            row[..3].copy_from_slice(&metrics.rho_m);
            row[3..6].copy_from_slice(&metrics.rho_dot_mps);
            row[6..15].copy_from_slice(&metrics.basis);
            row[15..33].copy_from_slice(metrics.rotations.as_flattened());
            row[33..36].copy_from_slice(&metrics.port_r_m);
            row[36..39].copy_from_slice(&metrics.port_v_mps);
            row[39] = metrics.alignment_deg;
            row[40] = metrics.relative_rate_radps;
            Ok(())
        })
}
