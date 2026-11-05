# RPOD

Simulation, perception and autonomy for spacecraft rendezvous, proximity operations and docking, on a Rust physics core.

![The chaser following the unmarked QuikSCAT's tumbling docking axis in to a 5 m hold](docs/showcase.gif)

*Episode 16 of the M11 suite, sped up 22×: the chaser follows QuikSCAT's docking axis as the target tumbles at 1°/s, navigating by lidar shape matching and camera keypoints. The inset is the docking camera. `rpod showcase` rendered it from the run's saved record.*

A chaser spacecraft flies itself to a 5 m hold point at a satellite using only what its sensors deliver: camera images that Blender renders from the simulated state, and a simulated flash lidar, arriving late and sometimes dropped or degraded. Physics runs in Rust. Guidance, navigation, perception and evaluation run in Python and never see the true state. Every result below comes from a frozen, hash-checked evaluation suite or a seeded Monte Carlo campaign, and every run saves records that replay without rerunning the physics.

## Results

| Milestone | Adds | Result |
| --- | --- | --- |
| M1 | RK4 orbital engine in Rust, Python bindings, Rerun replay | 2 µm from the analytic orbit over 6,000 s; 8.2 million physics ticks/s |
| M2 | Rigid-body attitude, bounded force and torque, supervised truth-state approach | nominal, disturbed and retreat runs end inside their envelopes |
| M3 | Calibrated Blender camera, marker pose estimation | pose gates pass on held-out renders |
| M4 | Closed-loop approach on delayed image observations | all six qualification cases pass |
| M5 | Reusable `reset`/`step` environment, frozen 40-episode suite | 28/28 operational successes; aborts, dropped frames and lost tracking get their designed responses |
| M6 | Flash lidar fused in the same navigation filter | dropped frames and extended camera loss no longer force a brake |
| M7–M8 | Realistic scene: NASA's QuikSCAT, imaged Earth, space lighting, GPU rendering | 28/28; the camera accepted 8,420 of 8,432 images |
| M9 | Batched multithreaded engine, vectorized autonomy, validated navigation surrogate | 15.7 million environment ticks/s on 12 threads; 99.932% success over 100,000 episodes (95% interval 99.914–99.947%); agrees with Basilisk to 1.5e-8 m |
| M10 | Learned approach policy behind a runtime safety filter | same success rate as the reference, median completion 171 s instead of 210 s for 17% more Δv, no filter interventions; in the rendered suite, 28/28 operational successes 18% faster, and one abort ended in a false completion at the retreat tolerance |
| M11 | Unmarked, tumbling target: lidar shape matching and a keypoint network | lidar-only and camera-plus-lidar suites both pass every gate: 28/28 operational successes at the oracle's pace, both aborts retreated, target position error 1.2 cm median; the surrogate does not validate, so no population estimate |

28/28 is a fixed-suite result; population claims come from the Monte Carlo campaigns, which M11 does not have. [REFERENCE.md](REFERENCE.md) records every gate, metric and caveat.

## How it works

```mermaid
flowchart LR
    subgraph rust [Rust]
        engine[Orbit and rigid-body engines] --- batch[Batched engine]
        mesh[Mesh ray casts and closest points]
    end
    subgraph python [Python]
        env[Environment: timing, faults, scoring]
        perception[Perception: markers, ICP, keypoints]
        nav[Navigation filter]
        control[Supervisor and controllers]
    end
    blender[Blender renderer]
    env -- state --> blender -- images --> perception
    env -- scans --> perception --> nav --> control -- force and torque --> env
    env <-- PyO3 --> engine
    perception <-- PyO3 --> mesh
```

- **Rust core** (`crates/rpod-core`) owns physical state and the simulation clock: fixed-step RK4 for orbits and rigid bodies, atomic ticks, and a batch of independent environments stepped in parallel. Physics has no renderer dependency.
- **Python** (`src/rpod`) defines scenarios, sensors, perception, navigation, guidance, supervision and evaluation. Truth, observations and estimates stay separate: controllers see only delivered observations and their own estimates.
- **Blender** runs as a separate persistent worker that renders camera images from the states the feedback loop reaches. Rendering time never advances simulation time.
- **Records** hold configuration, seeds, software hashes, states, commands, observations, estimates and events, and replay in [Rerun](https://rerun.io) without rerunning physics.

## Quickstart

Verified on Windows x86_64 with [uv](https://docs.astral.sh/uv/) and stable Rust (MSVC build tools). From the repository root:

```powershell
uv sync --locked
uv run --locked maturin develop --release --locked
.\.venv\Scripts\python.exe -m unittest discover -s tests

.\.venv\Scripts\rpod.exe run --scenario drift --output runs/drift
.\.venv\Scripts\rpod.exe replay runs/drift
.\.venv\Scripts\rpod.exe monte-carlo --episodes 10000 --output runs/campaign
```

Rendered experiments need Blender 5.2.3, at `.tools/blender-5.2.3-windows-x64/blender.exe` or on `PATH`. The realistic scene downloads NASA imagery and models (about 205 MB, hash-checked) on first use and is practical only on a CUDA GPU. [REFERENCE.md](REFERENCE.md#environment-and-commands) lists every command.

## Documentation

- [OVERVIEW.md](OVERVIEW.md): purpose, direction and open decisions.
- [REFERENCE.md](REFERENCE.md): the canonical specification of implemented behavior, interfaces and results.
- Milestone write-ups: [M9, batched engine and Monte Carlo](docs/m9-batched-engine.md), [M10, learned control](docs/m10-learned-control.md), [M11, tumbling target](docs/m11-tumbling-target.md).

## Limitations

- Actuation is ideal bounded force and torque: no individual thrusters, plume effects or fuel depletion.
- There is no contact model. Episodes end at a 5 m pre-contact hold.
- Perception is trained and tested on renders only; transfer to real imagery is untested.
- Reproducibility covers the same code, dependencies, build and platform. GPU rendering is not bitwise repeatable, so rendered reruns reproduce outcomes, not every pixel.

## License

Apache-2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) credits the NASA and ESA data downloaded at run time.
