# RPOD

RPOD is a project to build efficient, reproducible simulation environments for autonomous spacecraft rendezvous, proximity operations, and docking.

It begins as a portfolio project, with the longer-term intention of exploring whether the work could become the foundation of a startup.

The current direction is a custom Rust simulation core, Python for experiments and autonomy, Rerun for engineering visualization, and Blender for spacecraft assets and camera imagery.

This document describes the project's purpose and intended direction. The implementation includes orbital motion, rigid-body rotation, bounded actuation, and truth-state approach control. It renders calibrated camera images and uses delayed visual navigation for approach, retreat, and tracking-loss handling, first from cooperative markers and then, on an unmarked tumbling satellite, from keypoints found by a trained network. A simulated flash lidar feeds the same navigation filter; on the unmarked satellite, matching its scans to the satellite's shape carries most of the navigation. A realistic scene renders NASA spacecraft models under space lighting with an imaged Earth. A batched engine runs Monte Carlo campaigns of 100,000 episodes, in which a learned approach controller is trained and evaluated behind a safety filter. The chaser also docks with that tumbling satellite: it is captured at contact inside a stated envelope by a momentum-conserving latch, then detumbles the joined pair. Broader task environments, individual thrusters, and contact dynamics remain future work. [REFERENCE.md](REFERENCE.md) records implemented behavior and qualification results; this overview is not an implementation specification.

## End goal

Build an experimentation platform for developing, training, evaluating, and understanding autonomous spacecraft behavior. We should be able to define an encounter, supply a controller or autonomy system, run repeatable experiments, and inspect why an approach succeeded or failed.

The project has three core objectives: spacecraft autonomy, perception, and engine efficiency. Progress should demonstrate all three while preserving trustworthy physical and numerical behavior.

The platform should support both conventional guidance, navigation, and control methods and learned policies. Different approaches should be evaluated under the same stated physical assumptions, sensor access, actuator limits, and task criteria.

Perception is a central part of the project. The long-term goal includes autonomy that uses camera images and other sensor observations to estimate relative motion and make decisions. Exact-state control remains useful as an explicitly identified development baseline.

Engine efficiency is a core objective alongside autonomy and perception. We want control over the execution, memory use, and batching of simulation environments, with compute energy measured where practical. Improvements must preserve the declared model fidelity and numerical accuracy.

## Portfolio and startup potential

The near-term goal is a substantial portfolio project that demonstrates spacecraft simulation, Rust systems engineering, perception, autonomous control, and quantitative performance evaluation. It should produce understandable demonstrations, reproducible experiments, measured benchmarks, and clear documentation that make the work easy to assess.

The longer-term goal is to explore whether these capabilities solve a specific problem well enough to support a startup. That possibility should be informed by technical results and evidence of useful applications. Target users, product scope, business model, and commercial viability have not been established. The portfolio project should remain valuable as a technical achievement while those possibilities are explored.

## Intended environment families

These are intended capabilities, not existing environments.

| Environment | Intended behavior |
| --- | --- |
| Rendezvous | Move from an initial relative orbit or separation to a defined proximity-operations region or hold point. |
| Station keeping | Maintain a relative position and pointing condition within specified bounds. |
| Flyaround | Move between viewpoints or hold points while respecting visibility and separation constraints. |
| Terminal approach | Align with a docking interface and approach a pre-contact target condition. |
| Abort and retreat | Respond to an unsafe condition or fault and execute a defined retreat maneuver. |
| Docking contact | A possible later extension for physical contact, capture, and latching with a specified docking mechanism. |

The environments should eventually support seeded variation, disturbances, sensor degradation, actuator faults, and batch evaluation. Physical contact is a separate modeling stage from reaching a pre-contact approach envelope.

## Why a custom Rust core

We want to explore how a simulator designed specifically for these environments can use computation and memory efficiently. Owning the core gives us control over state layout, repeated stepping, reset behavior, memory reuse, and the execution of many independent environments.

Rust is the working language choice for this core. Its ownership model is useful for managing simulation state, but language choice alone does not establish speed, energy efficiency, or physical correctness. Those properties must be designed and measured.

The cost of this choice is responsibility for the dynamics, numerical methods, model assumptions, and verification. We will start with a narrow physical model and add detail when an experiment requires it.

Basilisk remains a useful reference for spacecraft models and selected comparisons. It is not part of the proposed runtime foundation. Analytical solutions and small independent calculations should provide verification without requiring a second maintained simulator.

## Division of responsibilities

| Layer | Intended responsibility |
| --- | --- |
| Rust core | Own physical state and simulation time; propagate spacecraft motion; apply physical actuator commands; compute geometry and physical diagnostics. |
| Python experiment layer | Define scenarios, initialization, controllers, estimators, sensor schedules, task phases, safety supervision, outcomes, evaluation, and recording. |
| Python bindings | Expose a small, explicit interface between the Rust core and Python. |
| Rerun | Display spacecraft geometry, frames, trajectories, images, estimates, and telemetry on a shared simulation timeline. |
| Blender | Create visual assets and render camera imagery from supplied spacecraft, camera, and lighting states. |

The Rust core should run without a viewer or renderer. Visual tools consume its timestamped outputs. Rendering time on the computer must not implicitly advance physical simulation time.

The setup includes nalgebra for dynamics mathematics, PyO3 and maturin for the Python bridge and packaging, NumPy and Matplotlib for experiments, OpenCV for image-based pose estimation, and Rerun for replay. Blender runs as a separate persistent renderer. Parry's double-precision geometry answers mesh ray casts and clearance queries; it remains a candidate for collision work. Installed versions and setup commands are recorded in [REFERENCE.md](REFERENCE.md). A general plugin system or interchangeable-engine framework is not an initial requirement.

## Foundation rules

### Physical state, frames, and timing

- Use SI units and double precision for physical state and dynamics calculations.
- Define coordinate frames, transform directions, quaternion conventions, and port offsets explicitly.
- Maintain one authoritative physical state and simulation clock. Derive relative and port states from the spacecraft states.
- Include rotational lever-arm velocity when computing offset docking-port or camera motion.
- Give control updates, sensor capture, observation delivery, and actuator changes explicit timing.
- Keep render coordinates near the encounter while preserving the physical state's declared reference frame.

Exact conventions and interfaces belong in [REFERENCE.md](REFERENCE.md) as they are implemented. The rough handoff's equations and thresholds must be checked before adoption.

### Truth, observations, and estimates

Keep three concepts separate throughout the system:

- **Truth:** the physical state represented by the simulation.
- **Observations:** the measurements and images available to an autonomy system.
- **Estimates:** the state inferred from those observations.

Truth may be used for scoring and debugging without being available to the controller. A truth-based baseline, noisy-state surrogate, and image-based estimator must be identified accurately in experiment records.

### Events and outcomes

Distinguish physical events, safety-monitor detections, supervisor transitions, and final task outcomes. A recoverable warning can trigger retreat while simulation continues. A collision or unrecoverable numerical error can end an episode.

State the accuracy of collision and event detection. Checks at discrete physics ticks must not be presented as exact continuous-time detection. Define dwell requirements and retreat-success criteria with their sampling and evaluation assumptions.

### Recording and reproducibility

Record the resolved configuration, seed information, model and software versions, timestamps, states, requested and applied commands, events, and outcomes. When perception participates in a run, preserve the actual observations and estimates used by the autonomy system.

Numerical records should remain usable independently of the viewer. Replay should explain an experiment without rerunning its physics or silently regenerating different sensor images.

Define the scope of reproducibility, including the build and platform. Numerical agreement across implementations is a separate claim from repeating a run with the same software.

## First build: a visible, verified orbital experiment

The first implemented experiment connects Rust, Python, recording, and Rerun:

- Two simple spacecraft in a circular Earth orbit, initially separated along track.
- Gravity-driven translation, with prescribed orientations at first.
- A Python script that advances the Rust simulation and records timestamped state.
- A Rerun view showing the vehicles, coordinate axes, relative trajectory, and separation.

Success means the experiment runs without a viewer, agrees with a suitable analytical orbital case within a declared tolerance, shows numerical convergence as the timestep is reduced, and can be replayed from saved records. A small headless performance benchmark establishes a baseline.

This first experiment is a foundation check. It does not yet demonstrate autonomous approach or physical docking.

## Development path

Progress through small, inspectable experiments rather than implementing the entire model catalog at once.

1. **Orbital foundation and visualization.** Build the first experiment above and establish the implemented numerical and timing conventions.
2. **Attitude, actuation, and terminal approach.** Add rigid-body rotation and idealized bounded force/torque commands. Establish a baseline approach controller and nominal, perturbed, and explainable failure runs.
3. **Calibrated imagery and visual approach.** Verify Blender projection and rendered-pixel pose estimates, then close the loop using delayed observations. The implemented baseline uses known cooperative markers, simple lighting, and ideal own-spacecraft navigation; richer imagery and natural features require separate qualification.
4. **Task suite and evaluation.** Add the other RPOD tasks, seeded variation, faults, and batch comparisons as their definitions become concrete.
5. **Measured scaling and model extensions.** Improve batching, memory reuse, and parallel execution from profiling results. Add detailed actuators, disturbances, or contact models when required by an experiment.

Performance measurement begins with the first core. Optimization and verification continue throughout development. Blender can initially render recorded trajectories; reactive image-based autonomy requires rendering observations from the states reached by the feedback loop. That loop may run slower than real time.

## Performance and verification

Evaluate efficiency at a stated model fidelity, accuracy, and workload. Track:

- Headless physics throughput and complete-environment throughput separately.
- Reset latency, memory per environment, allocations, and scaling with environment count.
- Recording overhead, camera rendering, image transfer, and perception inference separately.
- Compute energy per defined workload when an actual measurement is available.

Early-terminated episodes or simplified physics must not be presented as equivalent workload speedups.

Verification should include analytical motion cases, frame and port-kinematics checks, actuator limits, nontrivial rotational dynamics, numerical convergence, event behavior, and reproducible initialization. Camera projection and sensor timing need their own checks. Attractive imagery alone does not establish a valid sensor model.

## Decisions still open

- Target users, useful applications, and product scope for potential commercialization.
- Physical parameters and task thresholds beyond the current fixed suites.
- Targets of unknown shape or inertia, and the inspection and synchronization that precede a tumbling-target approach.
- When ideal force/torque actuation should give way to individual thruster models.
- Camera, illumination, noise, latency, and sensor models beyond the calibrated pinhole baseline, and perception validated on real imagery.
- Controller and navigation alternatives beyond the reference law and the evolution-strategies policy.
- The accuracy and operating limits required of collision, dwell, and retreat checks.
- Scaling the rendered feedback loop and acceptable runtime for larger experiments.
- The docking mechanism and contact solver, if physical capture becomes a requirement.

Game engines, Isaac Sim, a contact engine, GPU execution, and a large-scale learning stack are not prerequisites for the initial build. They remain future choices to assess against a concrete need and available hardware.

## Project documents

- [README.md](README.md): a summary of results, architecture, and quickstart.
- [CLAUDE.md](CLAUDE.md): instructions for working in this repository.
- [OVERVIEW.md](OVERVIEW.md): purpose, intended direction, proposed scope, and open decisions.
- [REFERENCE.md](REFERENCE.md): the canonical record of implemented architecture, interfaces, invariants, and operational behavior.
- [docs](docs): milestone write-ups with method, results, limitations, and reproduction commands.
