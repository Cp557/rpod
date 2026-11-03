use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{OrbitError, RigidBodyEngine, RigidBodySnapshot};

/// Values per environment: for the target then the chaser, position, velocity, XYZW
/// quaternion and body angular velocity.
pub const STATE_WIDTH: usize = 26;

/// Vehicle and integration constants shared by every environment in a batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchParameters {
    pub mu_m3_s2: f64,
    pub dt_ns: u64,
    pub masses_kg: [f64; 2],
    pub inertias_kg_m2: [[f64; 3]; 2],
    pub force_limits_n: [f64; 2],
    pub torque_limits_nm: [f64; 2],
}

/// Independent target-chaser pairs, each a `RigidBodyEngine`, stepped on a private thread
/// pool. Environments share nothing, so results do not depend on the thread count.
pub struct RigidBodyBatch {
    parameters: BatchParameters,
    engines: Vec<RigidBodyEngine>,
    pool: ThreadPool,
}

fn engine(
    parameters: &BatchParameters,
    state: &[f64; STATE_WIDTH],
) -> Result<RigidBodyEngine, OrbitError> {
    let part = |offset: usize| -> [[f64; 3]; 2] {
        std::array::from_fn(|body| std::array::from_fn(|axis| state[13 * body + offset + axis]))
    };
    let quaternions =
        std::array::from_fn(|body| std::array::from_fn(|axis| state[13 * body + 6 + axis]));
    RigidBodyEngine::new(
        parameters.mu_m3_s2,
        parameters.dt_ns,
        part(0),
        part(3),
        quaternions,
        part(10),
        parameters.masses_kg,
        parameters.inertias_kg_m2,
        parameters.force_limits_n,
        parameters.torque_limits_nm,
    )
}

fn write_row(snapshot: &RigidBodySnapshot, row: &mut [f64]) {
    for (body, values) in row.as_chunks_mut::<13>().0.iter_mut().enumerate() {
        values[..3].copy_from_slice(&snapshot.positions_m[body]);
        values[3..6].copy_from_slice(&snapshot.velocities_mps[body]);
        values[6..10].copy_from_slice(&snapshot.quaternions_xyzw[body]);
        values[10..].copy_from_slice(&snapshot.angular_velocities_body_radps[body]);
    }
}

impl RigidBodyBatch {
    /// Bytes of engine state per environment.
    pub const BYTES_PER_ENVIRONMENT: usize = std::mem::size_of::<RigidBodyEngine>();

    /// One engine per state row; `threads` 0 lets rayon choose.
    pub fn new(
        parameters: BatchParameters,
        states: &[[f64; STATE_WIDTH]],
        threads: usize,
    ) -> Result<Self, OrbitError> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| OrbitError::InvalidInput("the batch thread pool could not start"))?;
        let engines = states
            .iter()
            .map(|state| engine(&parameters, state))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            parameters,
            engines,
            pool,
        })
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// Restart one environment from a new state at tick zero.
    pub fn reset(&mut self, index: usize, state: &[f64; STATE_WIDTH]) -> Result<(), OrbitError> {
        let slot = self
            .engines
            .get_mut(index)
            .ok_or(OrbitError::InvalidInput("batch index is out of range"))?;
        *slot = engine(&self.parameters, state)?;
        Ok(())
    }

    /// Hold each active environment's commands for `steps` ticks; inactive ones keep their
    /// state. The result holds each environment's error, if any. A failed environment keeps
    /// the ticks it completed, as `RigidBodyEngine::step` does, and the others still advance.
    pub fn step(
        &mut self,
        forces_body_n: &[[[f64; 3]; 2]],
        torques_body_nm: &[[[f64; 3]; 2]],
        active: &[bool],
        steps: u64,
    ) -> Result<Vec<Option<OrbitError>>, OrbitError> {
        let count = self.engines.len();
        if forces_body_n.len() != count || torques_body_nm.len() != count || active.len() != count {
            return Err(OrbitError::InvalidInput(
                "batch commands need one row per environment",
            ));
        }
        let engines = &mut self.engines;
        Ok(self.pool.install(|| {
            engines
                .par_iter_mut()
                .zip(forces_body_n)
                .zip(torques_body_nm)
                .zip(active)
                .map(|(((engine, force), torque), active)| match active {
                    true => engine.step(*force, *torque, steps).err(),
                    false => None,
                })
                .collect()
        }))
    }

    /// Write every environment's state row into `out`, which holds `len() * STATE_WIDTH`
    /// values.
    pub fn write_states(&self, out: &mut [f64]) -> Result<(), OrbitError> {
        if out.len() != self.engines.len() * STATE_WIDTH {
            return Err(OrbitError::InvalidInput(
                "the state buffer needs STATE_WIDTH values per environment",
            ));
        }
        let engines = &self.engines;
        self.pool.install(|| {
            out.par_chunks_exact_mut(STATE_WIDTH)
                .zip(engines)
                .for_each(|(row, engine)| write_row(&engine.snapshot(), row))
        });
        Ok(())
    }
}
