use parry3d_f64::math::Vector;
use parry3d_f64::query::{PointQuery, Ray, RayCast};
use parry3d_f64::shape::{FeatureId, TriMesh};
use rayon::prelude::*;

use crate::OrbitError;

/// A target's rendered geometry in body axes, behind a bounding-volume hierarchy, for lidar
/// ray casts, closest-point (ICP) correspondences and clearance checks.
pub struct TargetMesh {
    mesh: TriMesh,
}

/// Where a query point is nearest the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPoint {
    pub point: [f64; 3],
    /// Unit normal of the nearest triangle (either side), or the direction from the surface to the query
    /// point when the nearest feature is an edge or vertex.
    pub normal: [f64; 3],
    pub distance: f64,
}

impl TargetMesh {
    pub fn new(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Result<Self, OrbitError> {
        if !vertices.iter().flatten().all(|value| value.is_finite())
            || triangles
                .iter()
                .flatten()
                .any(|&index| index as usize >= vertices.len())
        {
            return Err(OrbitError::InvalidInput(
                "mesh vertices must be finite and triangles must index them",
            ));
        }
        let points = vertices
            .iter()
            .map(|&vertex| Vector::from(vertex))
            .collect();
        TriMesh::new(points, triangles.to_vec())
            .map(|mesh| Self { mesh })
            .map_err(|_| OrbitError::InvalidInput("the target mesh has no triangles"))
    }

    pub fn triangle_count(&self) -> usize {
        self.mesh.num_triangles()
    }

    /// Distance along each ray (unit directions) to the first surface, or NaN when there is
    /// none within `max_range`.
    pub fn cast_rays(
        &self,
        origins: &[[f64; 3]],
        directions: &[[f64; 3]],
        max_range: f64,
    ) -> Vec<f64> {
        origins
            .par_iter()
            .zip(directions)
            .map(|(&origin, &direction)| {
                let ray = Ray::new(Vector::from(origin), Vector::from(direction));
                self.mesh
                    .cast_local_ray(&ray, max_range, false)
                    .unwrap_or(f64::NAN)
            })
            .collect()
    }

    /// The nearest surface point to each query point.
    pub fn closest_points(&self, points: &[[f64; 3]]) -> Vec<ClosestPoint> {
        points
            .par_iter()
            .map(|&query| {
                let query = Vector::from(query);
                let (projection, feature) = self.mesh.project_local_point_and_get_feature(query);
                let offset = query - projection.point;
                let distance = offset.length();
                // The feature is the triangle's own (front or back face); the projection
                // names the triangle.
                let face = match feature {
                    FeatureId::Face(_) => self.mesh.triangle(projection.subshape).normal(),
                    _ => None,
                };
                let normal = face
                    .or_else(|| (distance > 0.0).then(|| offset / distance))
                    .unwrap_or(Vector::ZERO);
                ClosestPoint {
                    point: projection.point.into(),
                    normal: normal.into(),
                    distance,
                }
            })
            .collect()
    }
}
