use nalgebra::Vector3;
use rpod_core::{OrbitEngine, OrbitError, Snapshot};

const MU: f64 = 3.986_004_418e14;
const RADIUS_M: f64 = 6_778_137.0;

fn circular_state(time_s: f64, phase: f64, inclination: f64) -> ([f64; 3], [f64; 3]) {
    let rate = (MU / RADIUS_M.powi(3)).sqrt();
    let angle = phase + rate * time_s;
    let (sin, cos) = angle.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    (
        [
            RADIUS_M * cos,
            RADIUS_M * sin * cos_i,
            RADIUS_M * sin * sin_i,
        ],
        [
            -RADIUS_M * rate * sin,
            RADIUS_M * rate * cos * cos_i,
            RADIUS_M * rate * cos * sin_i,
        ],
    )
}

fn circular_engine(dt_ns: u64, inclination: f64) -> OrbitEngine {
    let (target_r, target_v) = circular_state(0.0, 0.0, inclination);
    let (chaser_r, chaser_v) = circular_state(0.0, -100.0 / RADIUS_M, inclination);
    OrbitEngine::new(MU, dt_ns, [target_r, chaser_r], [target_v, chaser_v]).unwrap()
}

fn vector_error(actual: [f64; 3], expected: [f64; 3]) -> f64 {
    (Vector3::from(actual) - Vector3::from(expected)).norm()
}

#[test]
fn circular_orbits_match_analytic_states_in_equatorial_and_inclined_planes() {
    for inclination in [0.0, 30.0_f64.to_radians()] {
        let mut engine = circular_engine(1_000_000_000, inclination);
        let mut max_position_error = 0.0_f64;
        let mut max_velocity_error = 0.0_f64;
        for tick in 1..=6_000 {
            let state = engine.step(1).unwrap();
            for (index, phase) in [0.0, -100.0 / RADIUS_M].into_iter().enumerate() {
                let (position, velocity) = circular_state(f64::from(tick), phase, inclination);
                max_position_error =
                    max_position_error.max(vector_error(state.positions_m[index], position));
                max_velocity_error =
                    max_velocity_error.max(vector_error(state.velocities_mps[index], velocity));
            }
        }
        assert!(
            max_position_error < 1.0e-3,
            "maximum position error: {max_position_error}"
        );
        assert!(
            max_velocity_error < 1.0e-6,
            "maximum velocity error: {max_velocity_error}"
        );
    }
}

#[test]
fn halving_the_step_exhibits_fourth_order_convergence() {
    let (reference, _) = circular_state(600.0, 0.0, 30.0_f64.to_radians());
    let errors = [40_u64, 20, 10].map(|dt_s| {
        let mut engine = circular_engine(dt_s * 1_000_000_000, 30.0_f64.to_radians());
        let state = engine.step(600 / dt_s).unwrap();
        vector_error(state.positions_m[0], reference)
    });
    for adjacent in errors.windows(2) {
        let ratio = adjacent[0] / adjacent[1];
        assert!((12.0..=20.0).contains(&ratio), "error ratio: {ratio}");
    }
}

#[test]
fn stepping_and_reset_preserve_clock_and_snapshot_contracts() {
    let mut engine = circular_engine(100_000_000, 0.0);
    let initial = engine.snapshot();
    assert_eq!(engine.step(0).unwrap(), initial);
    let later = engine.step(17).unwrap();
    assert_eq!(later.tick, 17);
    assert_eq!(later.time_ns, 1_700_000_000);
    assert_eq!(initial.tick, 0);
    assert_ne!(later.positions_m, initial.positions_m);
    assert_eq!(engine.reset(), initial);
    for _ in 0..17 {
        engine.step(1).unwrap();
    }
    assert_eq!(engine.snapshot(), later);
}

#[test]
fn identical_initial_states_propagate_identically() {
    let (position, velocity) = circular_state(0.0, 0.5, 0.4);
    let mut engine = OrbitEngine::new(MU, 1_000_000_000, [position; 2], [velocity; 2]).unwrap();
    let state = engine.step(400).unwrap();
    assert_eq!(state.positions_m[0], state.positions_m[1]);
    assert_eq!(state.velocities_mps[0], state.velocities_mps[1]);
}

#[test]
fn invalid_inputs_are_rejected() {
    let (position, velocity) = circular_state(0.0, 0.0, 0.0);
    for mu in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(OrbitEngine::new(mu, 1, [position; 2], [velocity; 2]).is_err());
    }
    assert!(OrbitEngine::new(MU, 0, [position; 2], [velocity; 2]).is_err());
    assert!(OrbitEngine::new(MU, 1, [[0.0; 3]; 2], [velocity; 2]).is_err());
    assert!(OrbitEngine::new(MU, 1, [position; 2], [[f64::NAN; 3]; 2]).is_err());
}

#[test]
fn clock_overflow_is_detected_before_any_step() {
    let mut engine = circular_engine(u64::MAX, 0.0);
    let before = engine.snapshot();
    assert_eq!(engine.step(2), Err(OrbitError::ClockOverflow));
    assert_eq!(engine.snapshot(), before);
}

#[test]
fn a_failed_second_body_does_not_commit_either_body() {
    let mut engine = OrbitEngine::new(
        1.0,
        2_000_000_000,
        [[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.3, 0.0], [-1.0, 0.0, 0.0]],
    )
    .unwrap();
    let before: Snapshot = engine.snapshot();
    assert!(matches!(engine.step(1), Err(OrbitError::InvalidState(_))));
    assert_eq!(engine.snapshot(), before);
}
