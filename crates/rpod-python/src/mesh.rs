use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::python_error;

fn rows<T: Element + Copy>(array: &PyReadonlyArray2<'_, T>) -> PyResult<Vec<[T; 3]>> {
    let view = array.as_array();
    if view.shape()[1] != 3 {
        return Err(PyValueError::new_err("arrays must have shape (n, 3)"));
    }
    Ok(view
        .outer_iter()
        .map(|row| std::array::from_fn(|axis| row[axis]))
        .collect())
}

type Closest<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// A target's rendered geometry in body axes for ray casts and closest points.
#[pyclass(frozen, module = "rpod._core")]
pub(super) struct TargetMesh {
    inner: rpod_core::TargetMesh,
}

#[pymethods]
impl TargetMesh {
    #[new]
    fn new(
        py: Python<'_>,
        vertices_body_m: PyReadonlyArray2<'_, f64>,
        triangles: PyReadonlyArray2<'_, u32>,
    ) -> PyResult<Self> {
        let (vertices, triangles) = (rows(&vertices_body_m)?, rows(&triangles)?);
        py.detach(|| rpod_core::TargetMesh::new(&vertices, &triangles))
            .map(|inner| Self { inner })
            .map_err(python_error)
    }

    #[getter]
    fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    /// Distance along each (n, 3) unit ray to the first surface, NaN without a hit.
    fn cast_rays<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray2<'py, f64>,
        directions: PyReadonlyArray2<'py, f64>,
        max_range: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (origins, directions) = (rows(&origins)?, rows(&directions)?);
        if origins.len() != directions.len() {
            return Err(PyValueError::new_err(
                "one direction per origin is required",
            ));
        }
        let ranges = py.detach(|| self.inner.cast_rays(&origins, &directions, max_range));
        Ok(PyArray1::from_vec(py, ranges))
    }

    /// Nearest surface points, their unit normals and distances for (n, 3) query points.
    fn closest_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Closest<'py>> {
        let points = rows(&points)?;
        let closest = py.detach(|| self.inner.closest_points(&points));
        let count = closest.len();
        let flat = |field: fn(&rpod_core::ClosestPoint) -> [f64; 3]| -> Vec<f64> {
            closest.iter().flat_map(field).collect()
        };
        Ok((
            PyArray1::from_vec(py, flat(|row| row.point)).reshape([count, 3])?,
            PyArray1::from_vec(py, flat(|row| row.normal)).reshape([count, 3])?,
            PyArray1::from_vec(py, closest.iter().map(|row| row.distance).collect()),
        ))
    }
}
