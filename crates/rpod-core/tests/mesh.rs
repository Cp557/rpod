use rpod_core::TargetMesh;

/// A 1 m cube centred on the origin, as 12 outward-wound triangles.
fn cube() -> TargetMesh {
    let vertices: Vec<[f64; 3]> = (0..8)
        .map(|index| {
            [
                if index & 1 == 0 { -0.5 } else { 0.5 },
                if index & 2 == 0 { -0.5 } else { 0.5 },
                if index & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let triangles = [
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ];
    TargetMesh::new(&vertices, &triangles).unwrap()
}

#[test]
fn rays_hit_the_first_surface_or_report_nan() {
    let mesh = cube();
    let ranges = mesh.cast_rays(
        &[[0.0, 0.0, -5.0], [0.2, -0.1, 3.0], [3.0, 3.0, 3.0]],
        &[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        100.0,
    );
    assert!((ranges[0] - 4.5).abs() < 1e-12);
    assert!((ranges[1] - 2.5).abs() < 1e-12);
    assert!(ranges[2].is_nan());
    assert!(mesh.cast_rays(&[[0.0, 0.0, -5.0]], &[[0.0, 0.0, 1.0]], 4.0)[0].is_nan());
}

#[test]
fn closest_points_give_face_normals_and_distances() {
    let mesh = cube();
    let [face, corner] = mesh.closest_points(&[[0.1, -0.2, 2.0], [2.0, 2.0, 2.0]])[..] else {
        panic!("two results expected")
    };
    let expected = [0.1, -0.2, 0.5];
    assert!(
        face.point
            .iter()
            .zip(expected)
            .all(|(value, want)| (value - want).abs() < 1e-12)
    );
    assert!((face.normal[2].abs() - 1.0).abs() < 1e-12 && (face.distance - 1.5).abs() < 1e-12);
    assert!(corner.point.iter().all(|value| (value - 0.5).abs() < 1e-12));
    assert!((corner.distance - 1.5 * 3f64.sqrt()).abs() < 1e-12);
}

#[test]
fn invalid_meshes_are_rejected() {
    assert!(TargetMesh::new(&[[0.0; 3]; 3], &[[0, 1, 3]]).is_err());
    assert!(TargetMesh::new(&[[f64::NAN, 0.0, 0.0]; 3], &[[0, 1, 2]]).is_err());
    assert!(TargetMesh::new(&[[0.0; 3]; 3], &[]).is_err());
}
