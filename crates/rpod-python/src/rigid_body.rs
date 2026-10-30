use pyo3::prelude::*;

use super::{StateTuple, python_error, vector_tuples};

type QuaternionTuple = (f64, f64, f64, f64);
type QuaternionPair = (QuaternionTuple, QuaternionTuple);

#[pyclass(frozen, module = "rpod._core")]
pub(super) struct RigidBodySnapshot {
    inner: rpod_core::RigidBodySnapshot,
}

impl From<rpod_core::RigidBodySnapshot> for RigidBodySnapshot {
    fn from(inner: rpod_core::RigidBodySnapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl RigidBodySnapshot {
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick
    }

    #[getter]
    fn time_ns(&self) -> u64 {
        self.inner.time_ns
    }

    #[getter]
    fn positions_m(&self) -> StateTuple {
        vector_tuples(self.inner.positions_m)
    }

    #[getter]
    fn velocities_mps(&self) -> StateTuple {
        vector_tuples(self.inner.velocities_mps)
    }

    #[getter]
    fn quaternions_xyzw(&self) -> QuaternionPair {
        let [first, second] = self.inner.quaternions_xyzw;
        (
            (first[0], first[1], first[2], first[3]),
            (second[0], second[1], second[2], second[3]),
        )
    }

    #[getter]
    fn angular_velocities_body_radps(&self) -> StateTuple {
        vector_tuples(self.inner.angular_velocities_body_radps)
    }

    #[getter]
    fn applied_forces_body_n(&self) -> StateTuple {
        vector_tuples(self.inner.applied_forces_body_n)
    }

    #[getter]
    fn applied_torques_body_nm(&self) -> StateTuple {
        vector_tuples(self.inner.applied_torques_body_nm)
    }
}

#[pyclass(module = "rpod._core")]
pub(super) struct RigidBodyEngine {
    inner: rpod_core::RigidBodyEngine,
}

#[pymethods]
impl RigidBodyEngine {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mu_m3_s2: f64,
        dt_ns: u64,
        positions_m: [[f64; 3]; 2],
        velocities_mps: [[f64; 3]; 2],
        quaternions_xyzw: [[f64; 4]; 2],
        angular_velocities_body_radps: [[f64; 3]; 2],
        masses_kg: [f64; 2],
        inertias_kg_m2: [[f64; 3]; 2],
        force_limits_n: [f64; 2],
        torque_limits_nm: [f64; 2],
    ) -> PyResult<Self> {
        Ok(Self {
            inner: rpod_core::RigidBodyEngine::new(
                mu_m3_s2,
                dt_ns,
                positions_m,
                velocities_mps,
                quaternions_xyzw,
                angular_velocities_body_radps,
                masses_kg,
                inertias_kg_m2,
                force_limits_n,
                torque_limits_nm,
            )
            .map_err(python_error)?,
        })
    }

    fn snapshot(&self) -> RigidBodySnapshot {
        self.inner.snapshot().into()
    }

    #[pyo3(signature = (forces_body_n, torques_body_nm, steps=1))]
    fn step(
        &mut self,
        py: Python<'_>,
        forces_body_n: [[f64; 3]; 2],
        torques_body_nm: [[f64; 3]; 2],
        steps: u64,
    ) -> PyResult<RigidBodySnapshot> {
        py.detach(|| self.inner.step(forces_body_n, torques_body_nm, steps))
            .map(RigidBodySnapshot::from)
            .map_err(python_error)
    }

    fn reset(&mut self) -> RigidBodySnapshot {
        self.inner.reset().into()
    }
}
