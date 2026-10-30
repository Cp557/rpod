use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use crate::{Derivative, OrbitError, OrbitalState};

/// Owned states and last applied body-frame commands, ordered target then chaser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBodySnapshot {
    pub tick: u64,
    pub time_ns: u64,
    pub positions_m: [[f64; 3]; 2],
    pub velocities_mps: [[f64; 3]; 2],
    /// XYZW quaternions rotating body-frame vectors into the inertial frame.
    pub quaternions_xyzw: [[f64; 4]; 2],
    pub angular_velocities_body_radps: [[f64; 3]; 2],
    pub applied_forces_body_n: [[f64; 3]; 2],
    pub applied_torques_body_nm: [[f64; 3]; 2],
}

#[derive(Clone, Copy)]
struct BodyParameters {
    mass_kg: f64,
    inertia_kg_m2: Vector3<f64>,
    force_limit_n: f64,
    torque_limit_nm: f64,
}

#[derive(Clone, Copy)]
struct BodyState {
    orbit: OrbitalState,
    attitude: UnitQuaternion<f64>,
    angular_velocity: Vector3<f64>,
}

#[derive(Clone, Copy)]
struct BodyDerivative {
    orbit: Derivative,
    attitude: Quaternion<f64>,
    angular_velocity: Vector3<f64>,
}

#[derive(Clone, Copy, Default)]
struct BodyWrench {
    force: Vector3<f64>,
    torque: Vector3<f64>,
}

fn normalized_attitude(quaternion: Quaternion<f64>) -> Result<UnitQuaternion<f64>, OrbitError> {
    let norm = quaternion.norm();
    if !quaternion.coords.iter().all(|value| value.is_finite()) || !norm.is_finite() || norm == 0.0
    {
        return Err(OrbitError::InvalidState(
            "quaternion must have finite components and a finite, nonzero norm",
        ));
    }
    Ok(UnitQuaternion::from_quaternion(quaternion))
}

impl BodyState {
    fn validate(&self) -> Result<(), OrbitError> {
        self.orbit.radius()?;
        if !self.angular_velocity.iter().all(|value| value.is_finite()) {
            return Err(OrbitError::InvalidState(
                "angular velocity components must be finite",
            ));
        }
        Ok(())
    }

    fn shifted(self, derivative: BodyDerivative, duration_s: f64) -> Result<Self, OrbitError> {
        Ok(Self {
            orbit: self.orbit.shifted(derivative.orbit, duration_s),
            attitude: normalized_attitude(
                self.attitude.into_inner() + derivative.attitude * duration_s,
            )?,
            angular_velocity: self.angular_velocity + derivative.angular_velocity * duration_s,
        })
    }
}

fn derivative(
    state: BodyState,
    parameters: BodyParameters,
    wrench: BodyWrench,
    mu: f64,
) -> Result<BodyDerivative, OrbitError> {
    state.validate()?;
    let mut orbit = crate::derivative(state.orbit, mu)?;
    orbit.velocity += state.attitude.transform_vector(&wrench.force) / parameters.mass_kg;
    let angular_momentum = parameters
        .inertia_kg_m2
        .component_mul(&state.angular_velocity);
    let angular_velocity = (wrench.torque - state.angular_velocity.cross(&angular_momentum))
        .component_div(&parameters.inertia_kg_m2);
    let attitude =
        state.attitude.into_inner() * Quaternion::from_parts(0.0, state.angular_velocity) * 0.5;
    if !orbit.velocity.iter().all(|value| value.is_finite())
        || !angular_velocity.iter().all(|value| value.is_finite())
        || !attitude.coords.iter().all(|value| value.is_finite())
    {
        return Err(OrbitError::InvalidState("state derivatives must be finite"));
    }
    Ok(BodyDerivative {
        orbit,
        attitude,
        angular_velocity,
    })
}

fn rk4(
    state: BodyState,
    parameters: BodyParameters,
    wrench: BodyWrench,
    mu: f64,
    duration_s: f64,
) -> Result<BodyState, OrbitError> {
    let k1 = derivative(state, parameters, wrench, mu)?;
    let k2 = derivative(state.shifted(k1, duration_s / 2.0)?, parameters, wrench, mu)?;
    let k3 = derivative(state.shifted(k2, duration_s / 2.0)?, parameters, wrench, mu)?;
    let k4 = derivative(state.shifted(k3, duration_s)?, parameters, wrench, mu)?;
    let weighted = BodyDerivative {
        orbit: Derivative {
            position: k1.orbit.position
                + k2.orbit.position * 2.0
                + k3.orbit.position * 2.0
                + k4.orbit.position,
            velocity: k1.orbit.velocity
                + k2.orbit.velocity * 2.0
                + k3.orbit.velocity * 2.0
                + k4.orbit.velocity,
        },
        attitude: k1.attitude + k2.attitude * 2.0 + k3.attitude * 2.0 + k4.attitude,
        angular_velocity: k1.angular_velocity
            + k2.angular_velocity * 2.0
            + k3.angular_velocity * 2.0
            + k4.angular_velocity,
    };
    let next = state.shifted(weighted, duration_s / 6.0)?;
    next.validate()?;
    Ok(next)
}

fn bounded_vector(command: [f64; 3], limit: f64) -> Result<Vector3<f64>, OrbitError> {
    let vector = Vector3::from(command);
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(OrbitError::InvalidInput(
            "command components must be finite",
        ));
    }
    let largest = vector.amax();
    if largest == 0.0 || limit == 0.0 {
        return Ok(Vector3::zeros());
    }
    // Scale before taking a norm so even very large finite requests can be bounded.
    let scaled = vector / largest;
    let scaled_norm = scaled.norm();
    if largest > limit / scaled_norm {
        Ok(scaled * (limit / scaled_norm))
    } else {
        Ok(vector)
    }
}

pub struct RigidBodyEngine {
    mu_m3_s2: f64,
    dt_ns: u64,
    tick: u64,
    parameters: [BodyParameters; 2],
    initial: [BodyState; 2],
    current: [BodyState; 2],
    applied: [BodyWrench; 2],
}

impl RigidBodyEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Result<Self, OrbitError> {
        if !mu_m3_s2.is_finite() || mu_m3_s2 <= 0.0 {
            return Err(OrbitError::InvalidInput(
                "mu_m3_s2 must be finite and positive",
            ));
        }
        if dt_ns == 0 {
            return Err(OrbitError::InvalidInput("dt_ns must be positive"));
        }
        if !masses_kg
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            || !inertias_kg_m2
                .iter()
                .flatten()
                .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(OrbitError::InvalidInput(
                "masses and principal inertias must be finite and positive",
            ));
        }
        if !force_limits_n
            .iter()
            .chain(&torque_limits_nm)
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(OrbitError::InvalidInput(
                "force and torque limits must be finite and nonnegative",
            ));
        }
        let parameters = std::array::from_fn(|index| BodyParameters {
            mass_kg: masses_kg[index],
            inertia_kg_m2: Vector3::from(inertias_kg_m2[index]),
            force_limit_n: force_limits_n[index],
            torque_limit_nm: torque_limits_nm[index],
        });
        let make_state = |index: usize| -> Result<BodyState, OrbitError> {
            let [x, y, z, w] = quaternions_xyzw[index];
            let state = BodyState {
                orbit: OrbitalState {
                    position: Vector3::from(positions_m[index]),
                    velocity: Vector3::from(velocities_mps[index]),
                },
                attitude: normalized_attitude(Quaternion::new(w, x, y, z))?,
                angular_velocity: Vector3::from(angular_velocities_body_radps[index]),
            };
            derivative(state, parameters[index], BodyWrench::default(), mu_m3_s2)?;
            Ok(state)
        };
        let initial = [make_state(0)?, make_state(1)?];
        Ok(Self {
            mu_m3_s2,
            dt_ns,
            tick: 0,
            parameters,
            initial,
            current: initial,
            applied: [BodyWrench::default(); 2],
        })
    }

    pub fn snapshot(&self) -> RigidBodySnapshot {
        RigidBodySnapshot {
            tick: self.tick,
            time_ns: self.tick * self.dt_ns,
            positions_m: self.current.map(|state| state.orbit.position.into()),
            velocities_mps: self.current.map(|state| state.orbit.velocity.into()),
            quaternions_xyzw: self
                .current
                .map(|state| state.attitude.quaternion().coords.into()),
            angular_velocities_body_radps: self.current.map(|state| state.angular_velocity.into()),
            applied_forces_body_n: self.applied.map(|wrench| wrench.force.into()),
            applied_torques_body_nm: self.applied.map(|wrench| wrench.torque.into()),
        }
    }

    /// Hold bounded body-frame commands constant across the requested physics ticks.
    pub fn step(
        &mut self,
        forces_body_n: [[f64; 3]; 2],
        torques_body_nm: [[f64; 3]; 2],
        steps: u64,
    ) -> Result<RigidBodySnapshot, OrbitError> {
        self.tick
            .checked_add(steps)
            .and_then(|tick| tick.checked_mul(self.dt_ns))
            .ok_or(OrbitError::ClockOverflow)?;
        let bound_wrench = |index: usize| -> Result<BodyWrench, OrbitError> {
            Ok(BodyWrench {
                force: bounded_vector(forces_body_n[index], self.parameters[index].force_limit_n)?,
                torque: bounded_vector(
                    torques_body_nm[index],
                    self.parameters[index].torque_limit_nm,
                )?,
            })
        };
        let applied = [bound_wrench(0)?, bound_wrench(1)?];
        let duration_s = self.dt_ns as f64 * 1.0e-9;
        for _ in 0..steps {
            let next = [
                rk4(
                    self.current[0],
                    self.parameters[0],
                    applied[0],
                    self.mu_m3_s2,
                    duration_s,
                )?,
                rk4(
                    self.current[1],
                    self.parameters[1],
                    applied[1],
                    self.mu_m3_s2,
                    duration_s,
                )?,
            ];
            self.current = next;
            self.applied = applied;
            self.tick += 1;
        }
        Ok(self.snapshot())
    }

    pub fn reset(&mut self) -> RigidBodySnapshot {
        self.current = self.initial;
        self.applied = [BodyWrench::default(); 2];
        self.tick = 0;
        self.snapshot()
    }
}
