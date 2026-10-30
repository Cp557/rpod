use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rpod_core::{OrbitError, RigidBodyEngine, RigidBodySnapshot};

const ZERO: [[f64; 3]; 2] = [[0.0; 3]; 2];
const IDENTITY: [[f64; 4]; 2] = [[0.0, 0.0, 0.0, 1.0]; 2];
const POSITIONS: [[f64; 3]; 2] = [[1.0e6, 0.0, 0.0], [1.0e6, 100.0, 0.0]];
const INERTIA: [[f64; 3]; 2] = [[2.0, 3.0, 4.0]; 2];

fn engine(
    dt_ns: u64,
    quaternions: [[f64; 4]; 2],
    angular_velocities: [[f64; 3]; 2],
) -> RigidBodyEngine {
    RigidBodyEngine::new(
        1.0,
        dt_ns,
        POSITIONS,
        ZERO,
        quaternions,
        angular_velocities,
        [2.0; 2],
        INERTIA,
        [10.0; 2],
        [10.0; 2],
    )
    .unwrap()
}

fn attitude(snapshot: &RigidBodySnapshot, body: usize) -> UnitQuaternion<f64> {
    let [x, y, z, w] = snapshot.quaternions_xyzw[body];
    UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z))
}

fn rotational_invariants(snapshot: &RigidBodySnapshot) -> (f64, Vector3<f64>) {
    let omega = Vector3::from(snapshot.angular_velocities_body_radps[0]);
    let momentum_body = Vector3::from(INERTIA[0]).component_mul(&omega);
    (
        0.5 * omega.dot(&momentum_body),
        attitude(snapshot, 0).transform_vector(&momentum_body),
    )
}

#[test]
fn torque_free_unequal_inertia_conserves_energy_and_inertial_momentum() {
    let initial_attitude = UnitQuaternion::from_euler_angles(0.2, -0.4, 0.3);
    let mut quaternions = IDENTITY;
    quaternions[0] = (initial_attitude.quaternion().coords * 3.0).into();
    let mut simulation = engine(10_000_000, quaternions, [[0.4, -0.3, 0.25], [0.0; 3]]);
    let initial = simulation.snapshot();
    let (energy, momentum) = rotational_invariants(&initial);
    let mut max_energy_error = 0.0_f64;
    let mut max_momentum_error = 0.0_f64;
    for tick in 1..=10_000 {
        let state = simulation.step(ZERO, ZERO, 1).unwrap();
        let (current_energy, current_momentum) = rotational_invariants(&state);
        max_energy_error = max_energy_error.max((current_energy / energy - 1.0).abs());
        max_momentum_error =
            max_momentum_error.max((current_momentum - momentum).norm() / momentum.norm());
        for quaternion in state.quaternions_xyzw {
            let norm = quaternion
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1.0e-14);
        }
        if tick == 100 {
            assert!(attitude(&initial, 0).angle_to(&attitude(&state, 0)) > 0.1);
        }
    }
    assert!(
        max_energy_error < 1.0e-9,
        "energy error: {max_energy_error}"
    );
    assert!(
        max_momentum_error < 1.0e-9,
        "momentum error: {max_momentum_error}"
    );
}

#[test]
fn principal_axis_spin_and_constant_torque_match_analytic_attitude() {
    let initial_attitude = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.4);
    let mut quaternions = IDENTITY;
    quaternions[0] = initial_attitude.quaternion().coords.into();
    let duration_s = 4.0;
    for torque in [0.0, 0.6] {
        let mut simulation = engine(5_000_000, quaternions, [[0.0, 0.0, 0.3], [0.0; 3]]);
        let state = simulation
            .step(ZERO, [[0.0, 0.0, torque], [0.0; 3]], 800)
            .unwrap();
        let acceleration = torque / INERTIA[0][2];
        let expected_rate = 0.3 + acceleration * duration_s;
        let angle = 0.3 * duration_s + 0.5 * acceleration * duration_s * duration_s;
        let expected_attitude =
            initial_attitude * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle);
        assert!((state.angular_velocities_body_radps[0][2] - expected_rate).abs() < 1.0e-12);
        assert!(attitude(&state, 0).angle_to(&expected_attitude) < 1.0e-10);
    }
}

#[test]
fn thrust_is_rotated_with_each_rk_stage_attitude() {
    let mut quaternions = IDENTITY;
    quaternions[1] =
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2)
            .quaternion()
            .coords
            .into();
    let mut simulation = engine(10_000_000, quaternions, [[0.0, 0.0, 1.0], [0.0; 3]]);
    let state = simulation.step([[4.0, 0.0, 0.0]; 2], ZERO, 200).unwrap();
    let duration_s = 2.0_f64;
    let acceleration = 2.0;
    let expected_velocity = Vector3::new(
        acceleration * duration_s.sin(),
        acceleration * (1.0 - duration_s.cos()),
        0.0,
    );
    let expected_position = Vector3::from(POSITIONS[0])
        + Vector3::new(
            acceleration * (1.0 - duration_s.cos()),
            acceleration * (duration_s - duration_s.sin()),
            0.0,
        );
    // At this radius mu=1 contributes less than 1e-11 over the test interval.
    assert!((Vector3::from(state.velocities_mps[0]) - expected_velocity).norm() < 1.0e-8);
    assert!((Vector3::from(state.positions_m[0]) - expected_position).norm() < 1.0e-7);
    assert!((state.velocities_mps[1][1] - acceleration * duration_s).abs() < 1.0e-10);
    assert!(state.velocities_mps[1][0].abs() < 1.0e-10);
}

#[test]
fn norm_limits_noop_invalid_commands_and_reset_preserve_contracts() {
    let mut simulation = RigidBodyEngine::new(
        1.0,
        10_000_000,
        POSITIONS,
        ZERO,
        [[0.0, 0.0, 0.0, 2.0]; 2],
        ZERO,
        [2.0; 2],
        INERTIA,
        [2.0, 0.0],
        [0.5, 0.0],
    )
    .unwrap();
    let initial = simulation.snapshot();
    assert_eq!(initial.quaternions_xyzw, IDENTITY);
    let state = simulation
        .step(
            [[3.0, 4.0, 0.0], [f64::MAX, f64::MAX, 0.0]],
            [[0.0, 0.0, 4.0]; 2],
            1,
        )
        .unwrap();
    let force = Vector3::from(state.applied_forces_body_n[0]);
    assert!((force - Vector3::new(1.2, 1.6, 0.0)).norm() < 1.0e-14);
    assert_eq!(state.applied_forces_body_n[1], [0.0; 3]);
    assert_eq!(state.applied_torques_body_nm, [[0.0, 0.0, 0.5], [0.0; 3]]);
    assert_eq!(simulation.step(ZERO, ZERO, 0).unwrap(), state);
    assert!(simulation.step([[f64::NAN; 3]; 2], ZERO, 1).is_err());
    assert_eq!(simulation.snapshot(), state);
    assert_eq!(initial.tick, 0);
    assert_eq!(state.tick, 1);
    assert_eq!(state.time_ns, 10_000_000);
    assert_eq!(simulation.reset(), initial);
}

#[test]
fn invalid_body_parameters_and_attitudes_are_rejected() {
    let build = |masses, inertias, quaternions, angular_velocities, force_limits, torque_limits| {
        RigidBodyEngine::new(
            1.0,
            10_000_000,
            POSITIONS,
            ZERO,
            quaternions,
            angular_velocities,
            masses,
            inertias,
            force_limits,
            torque_limits,
        )
    };
    for mass in [0.0, -1.0, f64::NAN] {
        assert!(build([mass, 2.0], INERTIA, IDENTITY, ZERO, [1.0; 2], [1.0; 2]).is_err());
    }
    assert!(
        build(
            [2.0; 2],
            [[0.0, 3.0, 4.0]; 2],
            IDENTITY,
            ZERO,
            [1.0; 2],
            [1.0; 2]
        )
        .is_err()
    );
    assert!(build([2.0; 2], INERTIA, [[0.0; 4]; 2], ZERO, [1.0; 2], [1.0; 2]).is_err());
    assert!(
        build(
            [2.0; 2],
            INERTIA,
            IDENTITY,
            [[f64::INFINITY; 3]; 2],
            [1.0; 2],
            [1.0; 2]
        )
        .is_err()
    );
    assert!(build([2.0; 2], INERTIA, IDENTITY, ZERO, [-1.0; 2], [1.0; 2]).is_err());
    assert!(build([2.0; 2], INERTIA, IDENTITY, ZERO, [1.0; 2], [f64::NAN; 2]).is_err());
}

#[test]
fn overflow_and_failed_ticks_do_not_commit_states_or_commands() {
    let mut overflow = engine(u64::MAX, IDENTITY, ZERO);
    let initial = overflow.snapshot();
    assert_eq!(overflow.step(ZERO, ZERO, 2), Err(OrbitError::ClockOverflow));
    assert_eq!(overflow.snapshot(), initial);
    let mut singular = RigidBodyEngine::new(
        1.0,
        2_000_000_000,
        [[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0; 3], [-1.0, 0.0, 0.0]],
        IDENTITY,
        ZERO,
        [2.0; 2],
        INERTIA,
        [10.0; 2],
        [10.0; 2],
    )
    .unwrap();
    let initial = singular.snapshot();
    assert!(
        singular
            .step([[1.0, 0.0, 0.0], [0.0; 3]], [[0.0, 0.0, 1.0], [0.0; 3]], 1)
            .is_err()
    );
    assert_eq!(singular.snapshot(), initial);
}
