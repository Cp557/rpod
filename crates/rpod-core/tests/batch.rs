use rpod_core::{BatchParameters, OrbitError, RigidBodyBatch, RigidBodyEngine, STATE_WIDTH};

const MU: f64 = 3.986004418e14;
const RADIUS: f64 = 6_778_137.0;
const PARAMETERS: BatchParameters = BatchParameters {
    mu_m3_s2: MU,
    dt_ns: 100_000_000,
    masses_kg: [200.0, 100.0],
    inertias_kg_m2: [[100.0, 120.0, 140.0], [20.0, 25.0, 30.0]],
    force_limits_n: [2.0, 2.0],
    torque_limits_nm: [0.5, 0.5],
};

/// A distinct state per environment: the chaser trails along track by 30 m plus `index`.
fn state(index: usize) -> [f64; STATE_WIDTH] {
    let speed = (MU / RADIUS).sqrt();
    let offset = index as f64;
    let mut row = [0.0; STATE_WIDTH];
    row[..3].copy_from_slice(&[RADIUS, 0.0, 0.0]);
    row[3..6].copy_from_slice(&[0.0, speed, 0.0]);
    row[6..10].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
    row[10..13].copy_from_slice(&[0.001 * offset, -0.002, 0.0011]);
    row[13..16].copy_from_slice(&[RADIUS, -30.0 - offset, 0.5]);
    row[16..19].copy_from_slice(&[0.01, speed, -0.002 * offset]);
    let half = 0.05 * offset;
    row[19..23].copy_from_slice(&[half.sin(), 0.0, 0.0, half.cos()]);
    row[23..].copy_from_slice(&[0.0, 0.003, -0.001 * offset]);
    row
}

/// Commands that exceed the limits for some environments, so bounding is exercised.
fn commands(index: usize) -> ([[f64; 3]; 2], [[f64; 3]; 2]) {
    let scale = 0.4 * index as f64;
    (
        [[0.0; 3], [scale, -1.0, 0.3 * scale]],
        [[0.05, -0.02, 0.1 * scale], [0.2 * scale, 0.01, -0.3]],
    )
}

fn single(index: usize) -> RigidBodyEngine {
    let row = state(index);
    let vector = |offset: usize| -> [[f64; 3]; 2] {
        std::array::from_fn(|body| std::array::from_fn(|axis| row[13 * body + offset + axis]))
    };
    RigidBodyEngine::new(
        MU,
        PARAMETERS.dt_ns,
        vector(0),
        vector(3),
        std::array::from_fn(|body| std::array::from_fn(|axis| row[13 * body + 6 + axis])),
        vector(10),
        PARAMETERS.masses_kg,
        PARAMETERS.inertias_kg_m2,
        PARAMETERS.force_limits_n,
        PARAMETERS.torque_limits_nm,
    )
    .unwrap()
}

fn states(batch: &RigidBodyBatch) -> Vec<f64> {
    let mut out = vec![0.0; batch.len() * STATE_WIDTH];
    batch.write_states(&mut out).unwrap();
    out
}

fn run(count: usize, threads: usize, ticks: usize) -> Vec<f64> {
    let rows: Vec<_> = (0..count).map(state).collect();
    let mut batch = RigidBodyBatch::new(PARAMETERS, &rows, threads).unwrap();
    let (forces, torques): (Vec<_>, Vec<_>) = (0..count).map(commands).unzip();
    for _ in 0..ticks {
        let errors = batch
            .step(&forces, &torques, &vec![true; count], 1)
            .unwrap();
        assert!(errors.iter().all(Option::is_none));
    }
    states(&batch)
}

#[test]
fn batch_matches_independent_engines_bitwise() {
    let count = 7;
    let batched = run(count, 3, 37);
    for index in 0..count {
        let (force, torque) = commands(index);
        let mut engine = single(index);
        for _ in 0..37 {
            engine.step(force, torque, 1).unwrap();
        }
        let snapshot = engine.snapshot();
        let row = &batched[index * STATE_WIDTH..(index + 1) * STATE_WIDTH];
        for body in 0..2 {
            let values = &row[13 * body..13 * (body + 1)];
            assert_eq!(values[..3], snapshot.positions_m[body]);
            assert_eq!(values[3..6], snapshot.velocities_mps[body]);
            assert_eq!(values[6..10], snapshot.quaternions_xyzw[body]);
            assert_eq!(values[10..], snapshot.angular_velocities_body_radps[body]);
        }
    }
}

#[test]
fn thread_count_does_not_change_results() {
    assert_eq!(run(64, 1, 20), run(64, 4, 20));
}

#[test]
fn a_failed_environment_does_not_stop_the_others() {
    let rows: Vec<_> = (0..3).map(state).collect();
    let mut batch = RigidBodyBatch::new(PARAMETERS, &rows, 2).unwrap();
    let before = states(&batch);
    let (mut forces, torques): (Vec<_>, Vec<_>) = (0..3).map(commands).unzip();
    forces[1][1][0] = f64::NAN;
    let errors = batch.step(&forces, &torques, &[true; 3], 5).unwrap();
    assert!(matches!(errors[1], Some(OrbitError::InvalidInput(_))));
    assert!(errors[0].is_none() && errors[2].is_none());
    let after = states(&batch);
    let row = |values: &[f64], index: usize| {
        values[index * STATE_WIDTH..(index + 1) * STATE_WIDTH].to_vec()
    };
    assert_eq!(row(&after, 1), row(&before, 1));
    assert_ne!(row(&after, 0), row(&before, 0));
    assert_ne!(row(&after, 2), row(&before, 2));
}

#[test]
fn inactive_environments_keep_their_state_and_reset_restarts_one() {
    let rows: Vec<_> = (0..3).map(state).collect();
    let mut batch = RigidBodyBatch::new(PARAMETERS, &rows, 2).unwrap();
    let before = states(&batch);
    let (forces, torques): (Vec<_>, Vec<_>) = (0..3).map(commands).unzip();
    batch
        .step(&forces, &torques, &[true, false, true], 3)
        .unwrap();
    let after = states(&batch);
    assert_eq!(
        after[STATE_WIDTH..2 * STATE_WIDTH],
        before[STATE_WIDTH..2 * STATE_WIDTH]
    );
    batch.reset(0, &state(0)).unwrap();
    assert_eq!(states(&batch)[..STATE_WIDTH], before[..STATE_WIDTH]);
    assert!(batch.reset(3, &state(0)).is_err());
}

#[test]
fn mismatched_command_rows_are_rejected() {
    let rows: Vec<_> = (0..2).map(state).collect();
    let mut batch = RigidBodyBatch::new(PARAMETERS, &rows, 1).unwrap();
    let (forces, torques): (Vec<_>, Vec<_>) = (0..1).map(commands).unzip();
    assert!(batch.step(&forces, &torques, &[true], 1).is_err());
    assert!(batch.write_states(&mut [0.0; 3]).is_err());
}
