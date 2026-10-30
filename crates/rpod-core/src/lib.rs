//! Point-mass and rigid-body spacecraft propagation in a fixed central gravity field.

#![forbid(unsafe_code)]

use std::fmt;

use nalgebra::Vector3;

mod rigid_body;

pub use rigid_body::{RigidBodyEngine, RigidBodySnapshot};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbitError {
    InvalidInput(&'static str),
    InvalidState(&'static str),
    ClockOverflow,
}

impl fmt::Display for OrbitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::InvalidState(message) => {
                formatter.write_str(message)
            }
            Self::ClockOverflow => formatter.write_str("simulation clock would overflow u64"),
        }
    }
}

impl std::error::Error for OrbitError {}

/// An owned copy of both spacecraft states, ordered target then chaser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    pub tick: u64,
    pub time_ns: u64,
    pub positions_m: [[f64; 3]; 2],
    pub velocities_mps: [[f64; 3]; 2],
}

#[derive(Clone, Copy, Debug)]
struct OrbitalState {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
}

impl OrbitalState {
    fn radius(&self) -> Result<f64, OrbitError> {
        if !self.position.iter().all(|value| value.is_finite())
            || !self.velocity.iter().all(|value| value.is_finite())
        {
            return Err(OrbitError::InvalidState("state components must be finite"));
        }
        let radius = self.position.norm();
        if !radius.is_finite() || radius == 0.0 {
            return Err(OrbitError::InvalidState(
                "position must have a finite, nonzero magnitude",
            ));
        }
        Ok(radius)
    }

    fn shifted(self, derivative: Derivative, duration_s: f64) -> Self {
        Self {
            position: self.position + derivative.position * duration_s,
            velocity: self.velocity + derivative.velocity * duration_s,
        }
    }
}

#[derive(Clone, Copy)]
struct Derivative {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
}

fn derivative(state: OrbitalState, mu: f64) -> Result<Derivative, OrbitError> {
    let radius = state.radius()?;
    let acceleration = -(mu / radius / radius) * (state.position / radius);
    if !acceleration.iter().all(|value| value.is_finite()) {
        return Err(OrbitError::InvalidState(
            "gravitational acceleration must be finite",
        ));
    }
    Ok(Derivative {
        position: state.velocity,
        velocity: acceleration,
    })
}

fn rk4(state: OrbitalState, mu: f64, duration_s: f64) -> Result<OrbitalState, OrbitError> {
    let k1 = derivative(state, mu)?;
    let k2 = derivative(state.shifted(k1, duration_s / 2.0), mu)?;
    let k3 = derivative(state.shifted(k2, duration_s / 2.0), mu)?;
    let k4 = derivative(state.shifted(k3, duration_s), mu)?;
    let next = OrbitalState {
        position: state.position
            + (k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position)
                * (duration_s / 6.0),
        velocity: state.velocity
            + (k1.velocity + 2.0 * k2.velocity + 2.0 * k3.velocity + k4.velocity)
                * (duration_s / 6.0),
    };
    next.radius()?;
    Ok(next)
}

pub struct OrbitEngine {
    mu_m3_s2: f64,
    dt_ns: u64,
    tick: u64,
    initial: [OrbitalState; 2],
    current: [OrbitalState; 2],
}

impl OrbitEngine {
    pub fn new(
        mu_m3_s2: f64,
        dt_ns: u64,
        positions_m: [[f64; 3]; 2],
        velocities_mps: [[f64; 3]; 2],
    ) -> Result<Self, OrbitError> {
        if !mu_m3_s2.is_finite() || mu_m3_s2 <= 0.0 {
            return Err(OrbitError::InvalidInput(
                "mu_m3_s2 must be finite and positive",
            ));
        }
        if dt_ns == 0 {
            return Err(OrbitError::InvalidInput("dt_ns must be positive"));
        }
        let initial = std::array::from_fn(|index| OrbitalState {
            position: Vector3::from(positions_m[index]),
            velocity: Vector3::from(velocities_mps[index]),
        });
        for state in initial {
            derivative(state, mu_m3_s2)?;
        }
        Ok(Self {
            mu_m3_s2,
            dt_ns,
            tick: 0,
            initial,
            current: initial,
        })
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            tick: self.tick,
            // Every tick is checked before propagation, so this product cannot overflow.
            time_ns: self.tick * self.dt_ns,
            positions_m: self.current.map(|state| state.position.into()),
            velocities_mps: self.current.map(|state| state.velocity.into()),
        }
    }

    /// Advance complete physics ticks, preserving the last successful tick on failure.
    pub fn step(&mut self, steps: u64) -> Result<Snapshot, OrbitError> {
        self.tick
            .checked_add(steps)
            .and_then(|tick| tick.checked_mul(self.dt_ns))
            .ok_or(OrbitError::ClockOverflow)?;
        let duration_s = self.dt_ns as f64 * 1.0e-9;
        for _ in 0..steps {
            let next = [
                rk4(self.current[0], self.mu_m3_s2, duration_s)?,
                rk4(self.current[1], self.mu_m3_s2, duration_s)?,
            ];
            self.current = next;
            self.tick += 1;
        }
        Ok(self.snapshot())
    }

    pub fn reset(&mut self) -> Snapshot {
        self.current = self.initial;
        self.tick = 0;
        self.snapshot()
    }
}
