use pyo3::prelude::*;
use pyo3::{exceptions::PyOverflowError, exceptions::PyValueError};

mod batch;
mod rigid_body;

type VectorTuple = (f64, f64, f64);
type StateTuple = (VectorTuple, VectorTuple);

fn vector_tuples(values: [[f64; 3]; 2]) -> StateTuple {
    let [first, second] = values;
    (
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
    )
}

fn python_error(error: rpod_core::OrbitError) -> PyErr {
    match error {
        rpod_core::OrbitError::ClockOverflow => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pyclass(frozen, module = "rpod._core")]
struct Snapshot {
    inner: rpod_core::Snapshot,
}

impl From<rpod_core::Snapshot> for Snapshot {
    fn from(inner: rpod_core::Snapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Snapshot {
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
}

#[pyclass(module = "rpod._core")]
struct OrbitEngine {
    inner: rpod_core::OrbitEngine,
}

#[pymethods]
impl OrbitEngine {
    #[new]
    fn new(
        mu_m3_s2: f64,
        dt_ns: u64,
        positions_m: [[f64; 3]; 2],
        velocities_mps: [[f64; 3]; 2],
    ) -> PyResult<Self> {
        Ok(Self {
            inner: rpod_core::OrbitEngine::new(mu_m3_s2, dt_ns, positions_m, velocities_mps)
                .map_err(python_error)?,
        })
    }

    fn snapshot(&self) -> Snapshot {
        self.inner.snapshot().into()
    }

    #[pyo3(signature = (steps=1))]
    fn step(&mut self, py: Python<'_>, steps: u64) -> PyResult<Snapshot> {
        py.detach(|| self.inner.step(steps))
            .map(Snapshot::from)
            .map_err(python_error)
    }

    fn reset(&mut self) -> Snapshot {
        self.inner.reset().into()
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", rpod_core::VERSION)?;
    module.add("__debug_assertions__", cfg!(debug_assertions))?;
    module.add("__rustc_version__", env!("RPOD_RUSTC_VERSION"))?;
    module.add_class::<OrbitEngine>()?;
    module.add_class::<Snapshot>()?;
    module.add_class::<rigid_body::RigidBodyEngine>()?;
    module.add_class::<rigid_body::RigidBodySnapshot>()?;
    module.add_class::<batch::RigidBodyBatch>()?;
    module.add_function(wrap_pyfunction!(batch::pair_metrics, module)?)?;
    Ok(())
}
