use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rpod_core::{BatchParameters, METRIC_WIDTH, OrbitError, STATE_WIDTH};

use super::python_error;

fn state_rows(states: &PyReadonlyArray3<'_, f64>) -> PyResult<Vec<[f64; STATE_WIDTH]>> {
    let view = states.as_array();
    if view.shape()[1..] != [2, 13] {
        return Err(PyValueError::new_err("states must have shape (n, 2, 13)"));
    }
    if let Ok(values) = states.as_slice() {
        return Ok(values.as_chunks::<STATE_WIDTH>().0.to_vec());
    }
    Ok(view
        .outer_iter()
        .map(|row| {
            let mut values = [0.0; STATE_WIDTH];
            values
                .iter_mut()
                .zip(row.iter())
                .for_each(|(slot, value)| *slot = *value);
            values
        })
        .collect())
}

fn command_rows(
    commands: &PyReadonlyArray3<'_, f64>,
    count: usize,
) -> PyResult<Vec<[[f64; 3]; 2]>> {
    let view = commands.as_array();
    if view.shape() != [count, 2, 3] {
        return Err(PyValueError::new_err("commands must have shape (n, 2, 3)"));
    }
    if let Ok(values) = commands.as_slice() {
        return Ok(values
            .as_chunks::<6>()
            .0
            .iter()
            .map(|row| [[row[0], row[1], row[2]], [row[3], row[4], row[5]]])
            .collect());
    }
    Ok(view
        .outer_iter()
        .map(|row| std::array::from_fn(|body| std::array::from_fn(|axis| row[[body, axis]])))
        .collect())
}

/// Status codes returned by `step`: 0 ok, 1 invalid command, 2 invalid state, 3 clock overflow.
fn status(error: Option<OrbitError>) -> u8 {
    match error {
        None => 0,
        Some(OrbitError::InvalidInput(_)) => 1,
        Some(OrbitError::InvalidState(_)) => 2,
        Some(OrbitError::ClockOverflow) => 3,
    }
}

#[pyclass(module = "rpod._core")]
pub(super) struct RigidBodyBatch {
    inner: rpod_core::RigidBodyBatch,
}

#[pymethods]
impl RigidBodyBatch {
    #[new]
    #[pyo3(signature = (states, mu_m3_s2, dt_ns, masses_kg, inertias_kg_m2, force_limits_n,
                        torque_limits_nm, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        states: PyReadonlyArray3<'_, f64>,
        mu_m3_s2: f64,
        dt_ns: u64,
        masses_kg: [f64; 2],
        inertias_kg_m2: [[f64; 3]; 2],
        force_limits_n: [f64; 2],
        torque_limits_nm: [f64; 2],
        threads: usize,
    ) -> PyResult<Self> {
        let rows = state_rows(&states)?;
        let parameters = BatchParameters {
            mu_m3_s2,
            dt_ns,
            masses_kg,
            inertias_kg_m2,
            force_limits_n,
            torque_limits_nm,
        };
        py.detach(|| rpod_core::RigidBodyBatch::new(parameters, &rows, threads))
            .map(|inner| Self { inner })
            .map_err(python_error)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Bytes of Rust engine state per environment.
    #[classattr]
    fn bytes_per_environment() -> usize {
        rpod_core::RigidBodyBatch::BYTES_PER_ENVIRONMENT
    }

    /// Restart the listed environments from new (k, 2, 13) states at tick zero.
    fn reset(&mut self, indices: Vec<usize>, states: PyReadonlyArray3<'_, f64>) -> PyResult<()> {
        let rows = state_rows(&states)?;
        if rows.len() != indices.len() {
            return Err(PyValueError::new_err("reset needs one state per index"));
        }
        for (index, row) in indices.into_iter().zip(&rows) {
            self.inner.reset(index, row).map_err(python_error)?;
        }
        Ok(())
    }

    /// Hold (n, 2, 3) body-frame commands for `steps` ticks in every active environment and
    /// return one status code per environment.
    #[pyo3(signature = (forces_body_n, torques_body_nm, active, steps=1))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        forces_body_n: PyReadonlyArray3<'py, f64>,
        torques_body_nm: PyReadonlyArray3<'py, f64>,
        active: PyReadonlyArray1<'py, bool>,
        steps: u64,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let count = self.inner.len();
        let forces = command_rows(&forces_body_n, count)?;
        let torques = command_rows(&torques_body_nm, count)?;
        let active = active.as_array().to_vec();
        let errors = py
            .detach(|| self.inner.step(&forces, &torques, &active, steps))
            .map_err(python_error)?;
        Ok(PyArray1::from_vec(
            py,
            errors.into_iter().map(status).collect(),
        ))
    }

    /// Current (n, 2, 13) states: per body position, velocity, XYZW quaternion, body rate.
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let mut values = vec![0.0; self.inner.len() * STATE_WIDTH];
        py.detach(|| self.inner.write_states(&mut values))
            .map_err(python_error)?;
        PyArray1::from_vec(py, values).reshape([self.inner.len(), 2, 13])
    }
}

/// Pair geometry of (n, 2, 13) states, as `approach.state_metrics`, in an
/// (n, METRIC_WIDTH) array laid out as `rpod_core::write_metrics` documents.
#[pyfunction]
pub(super) fn pair_metrics<'py>(
    py: Python<'py>,
    states: PyReadonlyArray3<'py, f64>,
    docking_ports_body_m: [[f64; 3]; 2],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let owned;
    let rows: &[[f64; STATE_WIDTH]] = match states.as_slice() {
        Ok(values) if states.as_array().shape()[1..] == [2, 13] => values.as_chunks().0,
        _ => {
            owned = state_rows(&states)?;
            &owned
        }
    };
    let mut values = vec![0.0; rows.len() * METRIC_WIDTH];
    py.detach(|| rpod_core::write_metrics(rows, &docking_ports_body_m, &mut values))
        .map_err(python_error)?;
    PyArray1::from_vec(py, values).reshape([rows.len(), METRIC_WIDTH])
}
