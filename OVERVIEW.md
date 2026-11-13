# RPOD

RPOD is a project to build efficient, reproducible simulation environments for autonomous spacecraft rendezvous, proximity operations, and docking.

The stack is a custom Rust simulation core, Python for experiments and autonomy, Rerun for engineering visualization, and Blender for spacecraft assets and camera imagery.

This document describes the project's purpose and direction; it is not an implementation specification. [REFERENCE.md](REFERENCE.md) records implemented behavior and qualification results, and [README.md](README.md) summarizes them by milestone. Work proceeds in small, inspectable milestones rather than implementing a whole model catalog at once.

## Where it stands

A chaser flies itself to a target satellite on delayed sensor observations. The target was first cooperative, carrying markers and retroreflectors; later it is NASA's QuikSCAT, stripped of them and tumbling freely. Blender renders camera images from the states the feedback loop reaches, and a simulated flash lidar feeds the same navigation filter. On the unmarked satellite, matching lidar scans to its shape carries most of the navigation, and a trained network finds keypoints in the images. A batched engine runs Monte Carlo campaigns of 100,000 episodes, in which a learned approach controller was trained and evaluated behind a safety filter. The chaser docks with the tumbling satellite: a momentum-conserving latch captures it at contact inside a stated envelope, and the chaser then detumbles the joined pair. The same docking also flies on 16 thrusters, one of which fails mid-approach and is isolated. Far-range rendezvous is next; its first stage measured what that needs.

## End goal

Build an experimentation platform for developing, training, evaluating, and understanding autonomous spacecraft behavior. We should be able to define an encounter, supply a controller or autonomy system, run repeatable experiments, and inspect why an approach succeeded or failed.

The project has three core objectives: spacecraft autonomy, perception, and engine efficiency. Progress should demonstrate all three while preserving trustworthy physical and numerical behavior.

The platform should support both conventional guidance, navigation, and control methods and learned policies. Different approaches should be evaluated under the same stated physical assumptions, sensor access, actuator limits, and task criteria.

Perception is a central part of the project. The long-term goal includes autonomy that uses camera images and other sensor observations to estimate relative motion and make decisions. Exact-state control remains useful as an explicitly identified development baseline.

Engine efficiency is a core objective alongside autonomy and perception. We want control over the execution, memory use, and batching of simulation environments, with compute energy measured where practical. Improvements must preserve the declared model fidelity and numerical accuracy.

## Environment families

| Environment | Behavior | Status |
| --- | --- | --- |
| Rendezvous | Move from an initial relative orbit or separation to a defined proximity-operations region or hold point. | Not built: episodes start tens of meters from the target. M15 stage 0 measured what a far-range approach needs ([docs/m15-rendezvous.md](docs/m15-rendezvous.md)). |
| Station keeping | Maintain a relative position and pointing condition within specified bounds. | Partly: holds at 30 m and 5 m with dwell checks, not long-duration station keeping. |
| Flyaround | Move between viewpoints or hold points while respecting visibility and separation constraints. | Not built. |
| Terminal approach | Align with a docking interface and approach a pre-contact target condition. | Built, for the cooperative target (M2–M10) and the unmarked, tumbling one (M11). |
| Abort and retreat | Respond to an unsafe condition or fault and execute a defined retreat maneuver. | Built: aborts retreat to 30 m, lost tracking brakes, and a navigation loss triggers a collision-avoidance burn (M12). |
| Docking contact | Physical contact, capture, and latching with a specified docking mechanism. | Partly: capture inside a stated envelope by a momentum-conserving latch, then a detumble (M13); no contact dynamics. |

Seeded variation, initial-state disturbances, sensor degradation and faults, actuator faults (a thruster stuck on or off, M14), and batch evaluation exist.

## Why a custom Rust core

We want to explore how a simulator designed specifically for these environments can use computation and memory efficiently. Owning the core gives us control over state layout, repeated stepping, reset behavior, memory reuse, and the execution of many independent environments.

Rust is the working language choice for this core. Its ownership model is useful for managing simulation state, but language choice alone does not establish speed, energy efficiency, or physical correctness. Those properties must be designed and measured.

The cost of this choice is responsibility for the dynamics, numerical methods, model assumptions, and verification. We keep the physical model narrow and add detail when an experiment requires it.

Basilisk remains a useful reference for spacecraft models and selected comparisons. It is not part of the runtime. Analytical solutions and small independent calculations should provide verification without requiring a second maintained simulator.

## Division of responsibilities

| Layer | Responsibility |
| --- | --- |
| Rust core | Own physical state and simulation time; propagate spacecraft motion; apply physical actuator commands; compute geometry and physical diagnostics. |
| Python experiment layer | Define scenarios, initialization, controllers, estimators, sensor schedules, task phases, safety supervision, outcomes, evaluation, and recording. |
| Python bindings | Expose a small, explicit interface between the Rust core and Python. |
| Rerun | Display spacecraft geometry, frames, trajectories, images, estimates, and telemetry on a shared simulation timeline. |
| Blender | Create visual assets and render camera imagery from supplied spacecraft, camera, and lighting states. |

The Rust core should run without a viewer or renderer. Visual tools consume its timestamped outputs. Rendering time on the computer must not implicitly advance physical simulation time.

The setup includes nalgebra for dynamics mathematics, PyO3 and maturin for the Python bridge and packaging, NumPy and Matplotlib for experiments, OpenCV for image-based pose estimation, and Rerun for replay. Blender runs as a separate persistent renderer. Parry's double-precision geometry answers mesh ray casts and clearance queries; it remains a candidate for collision work. Installed versions and setup commands are recorded in [REFERENCE.md](REFERENCE.md). A general plugin system or interchangeable-engine framework is not a requirement.

## Foundation rules

### Physical state, frames, and timing

- Use SI units and double precision for physical state and dynamics calculations.
- Define coordinate frames, transform directions, quaternion conventions, and port offsets explicitly.
- Maintain one authoritative physical state and simulation clock. Derive relative and port states from the spacecraft states.
- Include rotational lever-arm velocity when computing offset docking-port or camera motion.
- Give control updates, sensor capture, observation delivery, and actuator changes explicit timing.
- Keep render coordinates near the encounter while preserving the physical state's declared reference frame.

Exact conventions and interfaces belong in [REFERENCE.md](REFERENCE.md).

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

## Performance and verification

Evaluate efficiency at a stated model fidelity, accuracy, and workload. Track:

- Headless physics throughput and complete-environment throughput separately.
- Reset latency, memory per environment, allocations, and scaling with environment count.
- Recording overhead, camera rendering, image transfer, and perception inference separately.
- Compute energy per defined workload when an actual measurement is available.

Early-terminated episodes or simplified physics must not be presented as equivalent workload speedups.

Verification should include analytical motion cases, frame and port-kinematics checks, actuator limits, nontrivial rotational dynamics, numerical convergence, event behavior, and reproducible initialization. Camera projection and sensor timing need their own checks. Attractive imagery alone does not establish a valid sensor model.

## Decisions still open

- Physical parameters and task thresholds beyond the current fixed suites.
- Targets of unknown shape or inertia, and the inspection and synchronization that precede a tumbling-target approach.
- Whether other suites should move from ideal force/torque actuation to M14's thrusters, and propellant mass depletion.
- Camera, illumination, noise, latency, and sensor models beyond the calibrated pinhole baseline, and perception validated on real imagery.
- Controller and navigation alternatives beyond the reference law and the evolution-strategies policy.
- The accuracy and operating limits required of collision, dwell, and retreat checks.
- Scaling the rendered feedback loop and acceptable runtime for larger experiments.
- A contact solver and docking-mechanism dynamics beyond M13's capture envelope and latch.

Game engines, Isaac Sim, a contact engine, GPU physics, and a large-scale learning stack have not been needed so far. They remain future choices to assess against a concrete need and available hardware.

## Project documents

- [README.md](README.md): a summary of results, architecture, and quickstart.
- [CLAUDE.md](CLAUDE.md): instructions for AI coding agents working in this repository.
- [OVERVIEW.md](OVERVIEW.md): purpose, direction, status by environment family, and open decisions.
- [REFERENCE.md](REFERENCE.md): the canonical record of implemented architecture, interfaces, invariants, and operational behavior.
- [docs](docs): milestone write-ups with method, results, limitations, and reproduction commands.
- [benchmarks](benchmarks): the result files that REFERENCE.md cites.
