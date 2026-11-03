# Technical reference

This document records implemented behavior. [OVERVIEW.md](OVERVIEW.md) describes the project's purpose and future direction.

## Current implementation

RPOD 0.1.0 implements M1 orbital experiments, M2 truth-state controlled pre-contact approach, M3 calibrated cooperative-marker perception, M4 image-based closed-loop approach, the M5 reusable visual environment (`reset`/`step`) with a frozen 40-episode evaluation suite, M6 optional simulated flash lidar fused with the camera in the same navigation filter, and an M7 optional realistic render scene (space lighting, Earth, GPU rendering), upgraded in M8 to NASA's QuikSCAT model as the target bus, a NASA-imaged rotating Earth with an atmosphere, and a raised Sun. The rendered-pixel M3 gates (in the simple and realistic scenes), all six fixed M4 qualification cases, and the M5, M6 lidar and M8 realistic suite gates pass. The M7 realistic suite had failed its fault gate when one abort episode's target entered Earth's shadow; in M8 no episode reaches sunset. A presentation-only `showcase` command renders a saved run as a realistic outside-view movie. Individual thrusters, fuel/mass depletion, environmental disturbances, natural-feature perception, and collision/contact dynamics are not implemented. Simple rendered geometry does not define a contact model.

The Rust `rpod-core` crate owns state and time. `rpod-python` exposes it through PyO3. Python defines scenarios, collects snapshots, derives relative state and diagnostics, and exports recordings. Physics has no renderer dependency.

## Environment and commands

Verified on Windows x86_64 with CPython 3.15.1, uv 0.13.5, Rust/Cargo 1.99.1, Visual Studio Build Tools 2022/MSVC 14.44.35207, and Windows SDK 10.0.26100.0. Dependencies include nalgebra 0.35.0, PyO3 0.30.1, maturin 1.15.0, NumPy 2.5.4, Matplotlib 3.11.2, Rerun 0.40.0, and development dependency psutil 7.2.2.

From the repository root in PowerShell:

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
uv sync --locked
uv run --locked maturin develop --release --locked

.\.venv\Scripts\rpod.exe run --scenario circular --output runs/circular
.\.venv\Scripts\rpod.exe run --scenario drift --output runs/drift
.\.venv\Scripts\rpod.exe replay runs/drift
.\.venv\Scripts\rpod.exe replay runs/drift --save-only
.\.venv\Scripts\rpod.exe benchmark --output benchmarks/m1.json
```

`python -m rpod` provides the same commands. `run` defaults to `circular` and never launches a viewer. Output must be a new or empty directory. `replay` regenerates `replay.rrd` from numerical records and opens the native viewer unless `--save-only` is supplied. It never constructs an engine or propagates physics. Benchmark JSON files must not already exist.

Use the environment's executables after the release build. A later `uv sync` can rebuild the editable extension in debug mode; rebuild release before benchmarking. The benchmark rejects extensions with debug assertions enabled. For an already synchronized environment, this offline build avoids another dependency installation:

```powershell
$env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
$env:PYO3_PYTHON = (Resolve-Path '.venv\Scripts\python.exe').Path
.\.venv\Scripts\maturin.exe develop --release --locked --offline --skip-install
```

`.python-version` selects Python 3.15.1. `rust-toolchain.toml` tracks stable with Clippy/rustfmt. Rust 1.99.1 is verified; the declared minimum Rust 1.89 is not independently tested. Cargo.lock and uv.lock record dependency resolution. Reproducibility applies to the same code, dependencies, build, and platform; cross-platform bitwise identity is not promised.

## Physics and Python interface

M1's `OrbitEngine` obeys `dr/dt = v`, `dv/dt = -mu*r/|r|^3`, integrated using classical fixed-step RK4. Both bodies share a central gravitational parameter and timestep; there is no mutual spacecraft gravity. Physical vectors use `f64`, meters, and meters per second. This API and the original orbital scenarios remain available unchanged.

```python
from rpod import OrbitEngine, Snapshot

engine = OrbitEngine(mu_m3_s2, dt_ns, positions_m, velocities_mps)
initial = engine.snapshot()
later = engine.step(steps=1)
restored = engine.reset()
```

- Constructor arrays have shape `(2, 3)`, always target then chaser. Inputs are copied.
- `Snapshot` has read-only `tick`, `time_ns`, `positions_m`, and `velocities_mps`. Vectors are copied tuples of tuples; old snapshots remain unchanged as the engine advances.
- `reset()` restores constructor state and tick zero. `step(0)` returns the current state. Bulk stepping runs inside Rust and releases the Python GIL.
- The authoritative clock is an unsigned 64-bit tick with an unsigned 64-bit positive `dt_ns`; elapsed time is the checked product `tick * dt_ns`. Conversion to floating-point seconds occurs for numerical integration and derived calculations.
- Invalid physical inputs, nonfinite states, zero/invalid radii, and nonfinite accelerations raise `ValueError`. Shape/type errors are rejected by the binding. Unsigned integer conversion and clock overflow raise `OverflowError`.
- Clock overflow is checked before a multi-tick call advances anything. A numerical failure leaves the last complete tick available: neither body's candidate state is committed until both succeed. Earlier successful ticks in a multi-tick call remain committed.

`rpod._core` also exposes native `__version__`, `__rustc_version__`, and `__debug_assertions__` provenance. The top-level package re-exports both engine/snapshot pairs and the version.

## Scenarios and frames

`rpod.scenarios.make_scenario(name)` returns fresh resolved configuration for `circular`, `drift`, and the three M2 scenarios described below. The following orbital defaults describe M1.

| Parameter | Value |
| --- | --- |
| Gravitational parameter | 3.986004418e14 m³/s² |
| Reference Earth radius | 6,378,137 m |
| Orbit altitude / radius | 400,000 m / 6,778,137 m |
| Timestep | 1,000,000,000 ns |
| Duration | 600 physics ticks, 600 seconds |
| Samples | 601, including tick zero |

The inertial frame `N` has Earth at its origin, target initially on +X, and initial target velocity along +Y. Axes remain fixed; no epoch or named astronomical frame is implied. Earth turns only in rendering (`earth_rotation_ne`); the physics has no Earth rotation.

The target starts at `[a, 0, 0]` with velocity `[0, sqrt(mu/a), 0]`. The chaser starts at phase `-100/a` radians on the same circle, with correspondingly rotated tangential velocity. The 100 m separation is an arc length, so the initial local radial offset is approximately -0.000738 m rather than exactly zero. `drift` adds `[0, 0.05, 0.02] m/s` to the chaser in the initial target-local basis. There are no stochastic effects in M1.

Target local frame `L` uses +X radial outward, +Z along `r_target × v_target`, and +Y = Z × X. Matrix `R_NL` has these local axes as columns in inertial coordinates. `rpod.frames.relative_state` derives:

```text
rho_L     = R_NL.T @ (r_chaser - r_target)
omega_L   = [0, 0, |r_target × v_target| / |r_target|²]
rho_dot_L = R_NL.T @ (v_chaser - v_target) - omega_L × rho_L
```

It returns local position, rotating-frame velocity, and `R_NL`, accepting matching `(..., 2, 3)` arrays. Zero target radius or angular momentum makes the frame undefined and raises `ValueError`. Relative state is derived, never separately integrated. Displayed relative speed is `|rho_dot_L|`, not the norm of the inertial velocity difference.

For M1 only, both displayed body frames are prescribed to align with target LVLH. M2 instead records and displays actual propagated attitudes. The local-frame velocity formulas above assume an unforced target in central gravity; they do not include RTN-plane precession from an out-of-plane target force. M2 keeps target translational force zero and uses its circular motion as the guidance reference.

## M2 rigid-body dynamics

```python
from rpod import RigidBodyEngine

engine = RigidBodyEngine(
    mu_m3_s2, dt_ns, positions_m, velocities_mps,
    quaternions_xyzw, angular_velocities_body_radps,
    masses_kg, inertias_kg_m2, force_limits_n, torque_limits_nm,
)
state = engine.step(forces_body_n, torques_body_nm, steps=1)
state = engine.snapshot()
initial = engine.reset()
```

Body order is target then chaser. Position, velocity, angular velocity, inertia, and wrench arrays have shape `(2, 3)`; quaternions `(2, 4)`; masses and force/torque limits `(2,)`. Mass is constant. Inertia is diagonal in principal body axes, supplied as positive principal moments in kg m². Limits are nonnegative Euclidean vector-norm bounds per body, independently for force (N) and torque (N m); they are not per-axis bounds. Zero disables that actuation type. No thruster allocation or propellant model is implied.

Quaternions use Hamilton multiplication and XYZW storage and rotate body vectors into inertial coordinates (`R_NB`). Angular velocity and commanded force/torque are expressed in body coordinates. Torque acts about the center of mass. Dynamics are:

```text
dv_N/dt = -mu*r_N/|r_N|³ + R_NB*F_B/m
domega_B/dt = I_B^-1 * (tau_B - omega_B × (I_B*omega_B))
dq_NB/dt = 0.5 * q_NB HamiltonProduct [omega_B, 0]
```

The coupled RK4 solver rotates body force using each intermediate attitude, not just the tick's starting attitude. Finite nonzero input quaternions are normalized; norms that are zero/nonfinite are rejected. Intermediate and completed attitudes are normalized. There is no gravity-gradient torque. Commands are held in body axes across the requested ticks; their inertial direction changes as the body rotates.

`RigidBodySnapshot` contains the M1 clock/translation fields plus frozen copied tuples `quaternions_xyzw`, `angular_velocities_body_radps`, `applied_forces_body_n`, and `applied_torques_body_nm`. Applied wrench is the bounded input of the last completed tick. Reset restores initial state and zero applied wrench. Step-zero calls validate inputs but do not change state/wrench/time. Overflow checks precede propagation, and each complete tick commits both bodies, applied wrench, and clock together. Numerical failure preserves the last complete tick. Input/state errors map to `ValueError`; unsigned conversion/clock overflow to `OverflowError`.

## M2 guidance, control, and supervision

```powershell
.\.venv\Scripts\rpod.exe run --scenario approach --output runs/m2-approach
.\.venv\Scripts\rpod.exe run --scenario approach-disturbed --output runs/m2-disturbed
.\.venv\Scripts\rpod.exe run --scenario approach-retreat --output runs/m2-retreat
.\.venv\Scripts\rpod.exe replay runs/m2-disturbed
```

All three use a 0.1 s physics tick and 1 s control period, with at most 1,200 simulated seconds. Observations are exact current truth with zero latency; there is no camera, sensor scheduling pipeline, or estimator. Guidance, controller, and supervisor live in Python, with physical bounds enforced again in Rust.

The target is 200 kg with principal inertia `[100,120,140] kg m²`; the chaser 100 kg with `[20,25,30] kg m²`. Both have 2 N force and 0.5 N m torque norm limits. Target force requests stay zero. Both bodies physically rotate to track target LVLH, initially with orbital angular rate. Full relative-body attitude alignment is constrained; roll is not a free docking-axis degree of freedom in this baseline.

Port offsets are target `[0,-1,0] m`, chaser `[0,0.5,0] m`, in each body's frame, with outward normals -Y and +Y respectively. `frames.port_relative_state` computes actual point positions `r + R_NB*p_B` and velocities `v + R_NB*(omega_B × p_B)`, then transforms their difference and derivative into COM-defined target LVLH, including rotating-frame transport. It returns `(port_r_L, port_v_L, R_NL)`. These geometric ports have no capture/contact behavior.

Axial gap is `-port_r_L.y`, distinct from Euclidean port distance and COM range. The initial 100 m COM arc separation gives approximately 98.5 m port gap. Aligned COM range is approximately port gap +1.5 m.

The hold transfer moves the gap reference from its initial value to 30 m. A rest-to-rest trapezoidal profile limits reference speed to 0.15 m/s and acceleration to 0.005 m/s²; short transfers use a triangular profile. After a valid ten-second hold, a new profile moves from 30 m to a final 5 m staging gap. The COM reference accounts for aligned port offsets. Position/rate PD uses gains 0.0016 s^-2 and 0.08 s^-1, plus exact differential gravity and circular-LVLH Coriolis/centrifugal compensation. Actual offset-port state determines acceptance and safety, rather than substituting COM state for port measurements.

Attitude PD uses the shortest quaternion rotation vector, desired orbital rate transformed into current body axes, rate-transport feedforward, and gyroscopic compensation. Angular-acceleration gains are 0.16 s^-2 and 0.8 s^-1, converted through each body's inertia before torque limiting.

At every physics tick, hold/final/retreat dwell requires port-position error <=0.2 m, rotating-frame port speed <=0.02 m/s, full relative-body attitude error <=2 degrees, and relative inertial angular-rate magnitude <=0.002 rad/s. All conditions must hold uninterrupted for ten seconds measured from the first valid sample; invalid samples reset dwell. Acceptance is tick-sampled, not continuous-time certification.

Supervisor phase codes are `0=hold`, `1=approach`, `2=retreat`, `3=complete` (any terminal outcome). Each tick evaluates hard bounds, soft abort conditions, scheduled requests, then dwell/outcome; phase transitions immediately recompute commands. Other commands update every ten ticks and hold through each following physics interval. Supervisor reactions do not wait for the next one-second controller boundary.

Hard bounds apply in every active phase: COM range <=3 m or axial port gap <=2 m ends the episode as failed. These are discrete experiment boundaries, not geometry-based collision detection. Before retreat, closing speed >0.4 m/s triggers retreat; at gaps <=35 m, lateral port offset >5 m or alignment >15 degrees also triggers retreat. Soft abort checks do not repeatedly restart a retreat. Hard bounds remain active during retreat. The first abort cause is retained.

`approach-disturbed` starts with an additional `[2,0,1] m` position offset, `[0,0.03,0] m/s` velocity offset, 12-degree roll error, and body-rate perturbation. `approach-retreat` requests a planned abort at the first approach tick with actual gap <=20 m. This is explicitly an `abort_request`, not a manufactured fault. Retreat starts from the actual gap at that tick toward 60 m, resetting reference velocity to zero for braking; it is not globally velocity-continuous across the abort. A valid ten-second endpoint yields `retreated`, never approach success. This endpoint does not guarantee future collision avoidance.

Outcomes are `success` for final staging dwell, `retreated` for retreat dwell, or `failed` for hard-bound violations, numerical failure, or time limit. There is no docking/capture-success outcome.

## Records and replay

`rpod.recording.run_scenario(name, output)` saves the following M1 schema-1 files:

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Schema version 1, scenario, body order, constants, timestep, step count, initial state, perturbation, units/frame/model descriptions, software/compiler/build provenance |
| `trajectory.npz` | Uncompressed arrays: `tick` uint64 `(T,)`; `positions_m` and `velocities_mps` float64 `(T, 2, 3)` |
| `summary.json` | Sample count, elapsed time, range statistics, final relative state, maximum local position change/speed, fractional specific-energy and angular-momentum-vector changes for each body |
| `replay.rrd` | Derived Rerun geometry, telemetry, timeline, and default blueprint |

Only the circular summary contains analytical position/velocity errors. There is no circular analytical claim for the perturbed chaser. Conservation changes are relative to each body's initial state.

M2 uses schema 2 and adds `events.json`. State arrays have `T` rows: the original translation/tick fields, `quaternions_xyzw` float64 `(T,2,4)`, `angular_velocities_body_radps` float64 `(T,2,3)`, `phase` uint8 `(T,)`, and `desired_gap_m` float64 `(T,)`. Interval arrays have `T-1` rows: requested/applied `forces_body_n` and `torques_body_nm` float64 `(T-1,2,3)`, and `control_update` bool `(T-1,)`. Array names are prefixed `requested_` or `applied_`, e.g. `applied_forces_body_n`. Interval row i applies over `[tick_i,tick_i+1)`; there is no invented command at the final state.

The M2 manifest includes resolved body, port, guidance, controller, supervisor, acceptance, timing, and observation assumptions. `max_steps` preserves the horizon and `num_steps` records actual completed ticks. Summaries add outcome, first abort reason, final port errors/rates, required/achieved final dwell, control-update count, maximum applied wrench norms, and integrals of wrench norms. Failed episodes report zero achieved final dwell. Orbital energy/momentum changes under applied force are diagnostics, not conservation-error claims; wrench integrals are not measured fuel or compute energy.

Each event records integer `tick`, matching `time_ns`, `kind`, `reason`, and phase code. Kinds distinguish `abort_request`, `safety_detection`, `supervisor_transition`, `physical_boundary`, and `outcome`. Events share the physical clock and remain ordered within a tick. `load_events` validates their order/timestamps and final outcome. Numerical files plus events fully drive M2 replay, with no controller or dynamics rerun.

`collect_trajectory(engine, num_steps)` records M1's current snapshot and every subsequent tick into new arrays. `load_run(directory)` supports schemas 1, 2, and 3 independently of the viewer; it validates schema/scenario agreement, body order, constants, unsigned timing bounds, dimensions/dtypes, finite values, contiguous ticks from zero, nonzero radii, and agreement with manifest initial states. M2 additionally validates unit quaternions, phase/interval arrays, body parameters, and applied norm limits. Schema 3 adds the visual sidecar checks described below. NumPy loading uses `allow_pickle=False`. Unsupported or corrupt records fail explicitly. The original M1 replay format remains supported.

Rerun uses duration timeline `sim_time` with integer nanosecond timestamps. Its signed 64-bit timeline has a smaller range than the engine clock; oversized replay times are rejected. Records are flushed and disconnected before export returns.

The orbit overview uses kilometers, a reference-radius Earth sphere, enlarged markers, and complete recorded paths. The encounter view uses meters and target LVLH coordinates, with a 2 m target cube, 1 m chaser cube, axes, and relative trajectory. These sizes do not define a collision model. Local coordinates are computed in double precision before renderer conversion. Full paths are static visual context; moving geometry and telemetry follow the selected tick. Explicit camera placement frames both bodies and their recorded paths. Charts show the complete record, with a minimum 1 m separation-axis span and a zero-baseline speed axis of at least 0.001 m/s so nominal roundoff does not appear as substantial motion.

M2 encounter geometry uses each recorded body rotation in target LVLH and offset port markers. The gray port-gap line is a measurement annotation, not a physical connection. Additional panels show desired/actual gap, COM range, port velocity, relative attitude, requested/applied wrench norms, phase, and timestamped events. Held commands and phase use step interpolation. M1 retains its original prescribed-orientation view.

## Verification

```powershell
cargo test -p rpod-core --locked
cargo fmt --all -- --check
$env:PYO3_PYTHON = (Resolve-Path '.venv\Scripts\python.exe').Path
cargo clippy --workspace --all-targets --locked -- -D warnings
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m rerun_cli rrd verify runs/circular/replay.rrd runs/drift/replay.rrd
```

Seven Rust tests cover analytical maximum error at every tick through 6,000 seconds for both spacecraft in equatorial and 30-degree inclined planes; fourth-order convergence; state/time contracts; identical-body determinism; invalid inputs; overflow; and atomic failed ticks. Seven Python tests cover binding contracts, long-run local-frame/conservation behavior, drift initialization, both save/replay pipelines without physics during replay, record corruption, and clean CLI errors.

Acceptance tolerances: maximum position error <1 mm and velocity error <1 μm/s over 6,000 seconds at 1 s timestep; RK4 error ratios 12–20 for 40/20/10 s steps over 600 s; circular local-position change <10 μm and local speed <10 nm/s. Conservation drift is checked below 1e-10 fractionally. Convergence uses coarse steps because roundoff dominates at very small steps.

Measured on this build: circular maximum position error approximately 2.01 μm and velocity error 2.12 nm/s over 6,000 seconds; local-position change 0.336 μm and local speed 0.158 nm/s. The 600 s drift scenario ends at approximately `[19.594, -79.006, 11.099] m` local position and 82.153 m range.

Both M1 recordings passed Rerun file verification and native rendering on this hardware. Screenshots are saved as `runs/circular/preview.png` and `runs/drift/preview.png`. The native viewer was opened with both recordings. Automated interaction could not proceed because computer-use app approval timed out; manual playback-control acceptance remains outstanding. Sandbox launches did not expose a window; native rendering and opening were verified outside the agent sandbox.

M2 adds six Rust checks for unequal-inertia torque-free motion, analytic spin/constant torque, rotating body thrust, norm limits/reset, invalid body parameters, and overflow/failed-tick atomicity. Over 100 s at 0.01 s steps, torque-free energy and inertial angular-momentum vector changes stay below 1e-9 fractionally and quaternion norm error below 1e-14. Python adds four focused checks covering independent offset-port finite differences, full nominal/disturbed dwell and bounded commands, v2 replay/corruption, retreat timing/safety, and dwell reset boundaries.

The combined baseline passes 13 Rust tests and 11 Python tests, along with workspace formatting and Clippy with warnings denied. Python tests also pass with resource warnings treated as errors. Tests cover automatic closing-speed retreat, immediate command updates, retained first cause, and hard boundaries while retreating without adding full-duration runs for those boundary cases.

Verified M2 demonstrations:

| Scenario | Outcome | Simulated time | Final axial port gap | Final port speed |
| --- | --- | --- | --- | --- |
| approach | success | 703.4 s | 4.99575 m | 0.001707 m/s |
| approach-disturbed | success | 703.5 s | 4.99653 m | 0.001425 m/s |
| approach-retreat | retreated | 885.2 s | 60.0058 m | 0.002402 m/s |

The retreat request occurs at tick 5784 (578.4 s), immediately updates control, and preserves `scheduled_abort_at_20m` through its terminal outcome. All three RRD files pass file verification. Native screenshots of disturbed approach and retreat are saved under their run directories; actual attitude, ports, guidance, wrench, and phase telemetry were visually inspected. Interactive playback controls have not been re-certified.

## Benchmark baseline

`rpod benchmark` uses one warmup and five measured repetitions per workload, resetting outside stepping timings. Counts are checked after each repetition. A physics tick advances both spacecraft; it is not a complete autonomy environment step.

| Workload | Fixed work | Initial median throughput |
| --- | --- | --- |
| Native bulk stepping | One `step(1_000_000)` call, including its returned snapshot | 8.16 million physics ticks/s |
| Python record collection | 10,000 steps and NumPy collection, including initial sample and array allocation | 413,490 physics ticks/s |
| Python reset calls | 10,000 calls, including returned snapshots and Python loop | 13.46 million calls/s |

The initial report is `benchmarks/m1.json`, measured on Windows/Intel Family 6 Model 158 with the verified release compiler. The desktop was not isolated from other workloads; these are local baselines, not hardware-independent guarantees or performance targets.

Reports include raw timings, medians, warmup durations, completed counts, resolved scenario, and software/hardware metadata. Disk writes, Rerun export, and the viewer are excluded from timing. Process RSS was approximately 32.3 MB before and 32.2 MB after this run; these snapshots are neither peak memory nor memory per environment. No compute energy, per-engine allocation, batching, or scaling results have been established.

An additional M2 release measurement, `benchmarks/m2-native.json`, records approximately 2.34 million rigid-body physics ticks/s for two spacecraft with zero body wrench, 0.1 s timestep, and one bulk call per 100,000-tick repetition (one warmup, five timed repetitions; reset excluded). This is native propagation only, not closed-loop Python throughput or an equivalent-work speed comparison against M1. No energy result is inferred.

## Visual perception and navigation implementation

The Python environment includes `opencv-python-headless==4.14.0.94` (runtime OpenCV 4.14.0), verified on standard CPython 3.15.1 with ArUco, PnP, and PNG encoding. Portable Blender 5.2.3 LTS is installed at `.tools/blender-5.2.3-windows-x64/blender.exe`; its archive was checked against the official SHA256 manifest. It runs a separate bundled-Python worker and does not import RPOD's native extension. Cycles CPU rendering works in background mode on this machine.

The immutable observation records are `OwnNavigation`, `ImageObservation`, `PoseEstimate`, `NavigationEstimate`, and `ChaserInputs`. Image observations carry actual pixels and capture/delivery times. Pose estimates map target-body coordinates into optical camera coordinates. Own navigation supplies only the chaser's position, velocity, body-to-inertial quaternion, and body angular rate. `ChaserInputs` has optional target estimate and an accepted/invalid/no-delivery flag; it contains no physical two-body snapshot or target initialization.

`CameraModel` defaults to 1024×1024 pixels, horizontal FOV 35 degrees, square pixels, principal point `(511.5,511.5)`, and zero distortion. `R_BC=[[1,0,0],[0,0,1],[0,-1,0]]` maps optical right/down/forward axes to chaser body axes. The mount is `[0.2,0.55,0.2] m`. Camera pose follows the physical mounting without look-at tracking or zoom. Physical poses stay double precision; the Blender scene uses relative camera-centered coordinates, with its camera-axis conversion `diag(1,-1,-1)`.

The shared target definition has four 0.55 m ArUco `DICT_4X4_50` markers with IDs 0–3, 0.08 m quiet zones, centers `(-.55,-1,+.55)`, `(+.55,-1.3,+.55)`, `(-.55,-1.3,-.55)`, `(+.55,-1,-.55)`, and normals -Y in target body coordinates. This is a cooperative marked-target model. The marker pads are rendering geometry, not a new contact model. Texture generation, mesh placement, and the estimator's ordered corner map use this definition.

`BlenderRenderer` maintains one background process with JSON-line request/reply messages. The default simple scene uses Cycles CPU, six threads, 16 samples, fixed Standard view transform/exposure, and no denoising, depth of field, or motion blur; the optional realistic scene is described under M7. Metadata records settings, texture/worker hashes, Blender version, render timings, and Blender-process RSS sampled every 0.1 s (not an OS high-water measurement). Simulation time never comes from renderer wall time.

`estimate_pose(ImageObservation, CameraModel)` reads RGB pixels only. It enlarges detector input 2× with cubic interpolation and uses AprilTag subpixel corner refinement with the ArUco 4×4 dictionary, maps corners back using `(pixel+0.5)/2-0.5`, then solves EPNP/RANSAC and refines inliers. Physical resolution, camera intrinsics, and field of view remain unchanged. Validity requires at least eight inlier corners spanning both marker-depth groups, minimum original-image marker edge 20 pixels, finite positive-depth geometry, front-face visibility, and reprojection RMS <=1 pixel. Missing/unusable images return explicit invalid reasons. No projected truth landmarks are estimator inputs.

```powershell
.\.venv\Scripts\rpod.exe camera-check --output datasets/camera-check
.\.venv\Scripts\rpod.exe perception-eval --suite m3 --output datasets/m3
.\.venv\Scripts\rpod.exe perception-eval --source runs/m2-disturbed --output datasets/m2-images
.\.venv\Scripts\rpod.exe visual-run --case nominal --seed 0 --calibration datasets/m3/calibration.json --output runs/my-visual-approach
.\.venv\Scripts\rpod.exe replay runs/m4-nominal
.\.venv\Scripts\rpod.exe replay datasets/m3 --save-only
```

The M3 fixture suite fixes six port gaps (4.5,5,10,20,30,32 m), six body poses, and three lit-face Sun directions. It has 108 positive renders, with 36 calibration and 72 held-out poses, degraded copies (Gaussian noise 2/255 and blur 0.5 pixels), and 24 negative renders. Partition membership is fixed before estimator evaluation. PNGs, hashes, feature detections, and estimates are in `observations.jsonl`; truth poses and scoring labels are kept separately under `truth/`. `dataset.json` records camera, source hashes, assets, and implementation hashes. `report.json`, `calibration.json`, and `performance.json` preserve measurements and qualification state.

`camera-check` compares numerical and Blender projection and independently measures colored rendered calibration-dot locations. The measured maximum discrepancies are approximately 0.000153 pixels for projection and 0.0625 pixels for raster centroids, passing the 0.05/0.5 pixel limits. After calibration-only corner-localization tuning, M3 passes every held-out gate on the original PNGs: 72/72 clean and 72/72 degraded positives accepted, all 24 negatives rejected, and no gross accepted errors. `datasets/m3/report-initial.json` preserves the initial failed baseline; `report.json` records final qualification. M4 refuses unqualified calibration.

| Axial gap | Clean P95 translation / rotation | Degraded P95 translation / rotation |
| --- | --- | --- |
| 4.5 m | 0.00283 m / 0.0675° | 0.00293 m / 0.0752° |
| 5 m | 0.00337 m / 0.0439° | 0.00334 m / 0.0582° |
| 10 m | 0.01482 m / 0.0930° | 0.01546 m / 0.0851° |
| 20 m | 0.05452 m / 0.1880° | 0.05514 m / 0.2253° |
| 30 m | 0.12401 m / 0.4385° | 0.12318 m / 0.4795° |
| 32 m | 0.19486 m / 0.5779° | 0.20160 m / 0.6096° |

These are fixed-fixture statistics, not population reliability estimates. The original 132-render/240-image dataset took approximately 217 seconds on this unisolated desktop, before the final estimator was re-evaluated on saved PNGs. Startup was 1.006 s; median warm render time was 0.878 s; final estimator median/P95 were 0.227/0.755 s. A separate sequential saved-PNG read/decode pass measured median/P95 0.031/0.052 s. Worker render timings include PNG writes; request timings also include message overhead. Sampled peak Blender RSS was approximately 266 MiB. Raw timings, scope, and sampled memory are in `performance.json`; no compute-energy measurement is inferred. `report.json` lists rejected positives and worst accepted cases as well as percentiles. Qualification rejects missing, duplicate, or relabelled frozen fixtures.

`VisualNavigation` implements a 12-error-state EKF for target inertial position/velocity, body-to-inertial quaternion, and inertial angular velocity. Translation uses central gravity RK4, attitude a constant spatial-rate model. Its left quaternion propagation differs from the Rust body's body-rate propagation. Camera pose covariance is transformed into the matching inertial residual frame. Covariance uses continuous white-acceleration integrated blocks, including the declared higher angular noise for acquisition/reacquisition. The filter keeps a two-second lag buffer, corrects delayed measurements at capture time using capture-time own navigation, and predicts causally to delivery/control time. Initialization uses visual position/attitude, own velocity with 0.1 m/s uncertainty, and zero angular rate with 0.005 rad/s uncertainty. Six-dimensional innovation gating rejects outliers without refreshing measurement age.

The visual runner uses 0.1 s physics, 1 Hz capture, 0.2 s simulated delivery delay, and control at ticks 2 modulo 10. Acquisition needs three consecutive accepted images within ten seconds. An invalid/missing expected delivery inhibits approach and resets dwell; two accepted images allow resumption. Predicted-state braking is limited to three seconds since the last accepted capture. Target truth controls the environment target and independent physical scoring only. The chaser receives an immutable `MissionSpec` whitelist and `ChaserInputs`.

`visual-run` supports `nominal`, `disturbed`, `hidden`, `abort`, `short-loss`, and `long-loss`, with explicit `--seed`, `--calibration`, and `--output`. The abort request uses the estimated 20 m gap and retreats to 30 m. Short loss removes capture 60; long loss schedules black captures 60–65 and ends when accepted-capture age exceeds three seconds. A run can terminate before all scheduled faults occur. Successful completion requires fresh estimates and uninterrupted ten-second onboard dwell within 0.1 m, 0.01 m/s, 1 degree, and 0.001 rad/s. Independent physical scoring applies the M2 0.2 m, 0.02 m/s, 2 degree, and 0.002 rad/s limits over ten seconds. Incorrect onboard completion becomes `false_completion`. A hard physical boundary can end the experiment but supplies no corrective chaser command.

The runner creates physical initialization from an M2 30 m handoff, then adds hidden seeded target position/velocity/attitude errors. These values stay outside autonomy. Its schema-3 records preserve physical/own-navigation states, requested/applied interval commands, images and delivery metadata, actual used estimates, controller events, independent scoring, and a frozen calibration copy.

All six actual-Blender qualification runs pass:

| Case / seed | Outcome | Simulated duration | Paired oracle duration |
| --- | --- | --- | --- |
| nominal / 0 | success | 208.4 s | 207.6 s |
| disturbed / 1 | success | 212.5 s | 211.7 s |
| hidden / 2 | success | 208.5 s | 207.6 s |
| abort / 0 | retreated to 30 m | 228.2 s | — |
| short-loss / 1 | success after reacquisition | 227.7 s | — |
| long-loss / 2 | navigation_lost | 62.1 s | — |

The three uninterrupted approaches finish within 1.005× their paired oracle times. Nominal ends at a 4.9961 m physical port gap. Short-loss braking begins at 60.2 s and approach resumes after two valid deliveries at 62.2 s. Long-loss ends when the last accepted capture becomes 3.1 s old. All runs respect the physical separation and actuator bounds, preserve causal timing, and produce zero false completions. Successful endpoints satisfy both onboard dwell and independent physical dwell. The aggregate `benchmarks/m3-m4-qualification.json` includes individual gates, artifact hashes, performance data, and independently recomputed final ten-second physical maxima. These are fixed-suite results, not a statistical reliability claim.

The official nominal feedback run took 362.2 wall-clock seconds for 208.4 simulated seconds, excluding its separately timed paired oracle. Its sampled Blender peak RSS was 266.2 MiB. Qualification cases ran in pairs on the same desktop, so these wall times describe that workload rather than an isolated renderer or real-time guarantee. Each run's summary preserves startup, per-render/request timing, sampled RSS, and total feedback-loop wall time.

The current fast suite passes 49 Python tests and 13 Rust tests, with Python resource warnings treated as errors, Rust formatting, and Clippy warnings denied. Actual saved Blender images exercise the image-to-command boundary: hidden scoring labels cannot affect commands, changed valid image evidence changes estimates and commands, and black images never acquire. Runtime-pipeline-disabled replay tests and real saved-run exports pass. Native screenshots of the M3 and M4 replays were inspected; interactive playback controls were not re-certified. `benchmarks/visual-approach.png` shows three actual nominal observations and the recorded port-gap history.

### Schema 3 visual records

Schema 3 preserves the M2 state/interval arrays and adds per-tick own-navigation arrays: `own_position_m`, `own_velocity_mps`, `own_quaternion_xyzw`, and `own_angular_velocity_body_radps`. The manifest records camera mounting/calibration, mission thresholds, filter configuration, loss schedule, Sun angle and inertial Sun direction, `scene` (realistic runs only), chaser disturbance, image degradation, renderer/assets, calibration checksum, and startup hashes of the executed implementation (`visual_control.py`, `navigation.py`, `visual_agent.py`, `visual_env.py`, `visual_run.py`, and since M6 `lidar.py`). Target initialization remains evaluation/environment metadata and is never passed to `VisualController`. Summaries also record per-image perception time and per-tick autonomy time under `performance`.

The qualified execution sources are archived under `benchmarks/m4-qualified-source/`. After qualification, three inherited M2 descriptions were corrected in manifest serialization: the retreat goal is 30 m, abort is requested at an estimated 20 m gap in the abort case, and the fixed inertial axes persist through the rotated handoff. Original manifests are preserved as `manifest.before-metadata-correction.json`. `record_metadata_correction` and a separate writer hash document this correction; execution hashes and all numerical/sensor records remain unchanged. The `mission` whitelist specifies onboard behavior.

| Artifact | Contents |
| --- | --- |
| `images/*.png` | Actual captured RGB, streamed to disk, including deliberately black observations |
| `observations.jsonl` | Capture ID, capture/scheduled/actual delivery nanoseconds, image path/checksum or missing-capture status, actual detected pose and rejection reason, filter acceptance |
| `estimates.jsonl` | Every tick's predicted target state, 12×12 covariance, last accepted capture time, health, delivery result, controller-use flag, and onboard metrics |
| `events.json` | Onboard acquisition, braking, reacquisition, abort and completion transitions; independent physical boundary/outcome events |
| `calibration.json` | Frozen M3 measurement calibration used by this run |
| `summary.json` | Physical outcome and dwell, bounds, paired oracle time where applicable, measured renderer performance, individual qualification gates |
| `lidar/*.npz` | Lidar runs only: each scan's float32 `ranges_m` (NaN without a return) and boolean `retroreflective` image |
| `lidar.jsonl` | Lidar runs only: scan ID, capture/scheduled/actual delivery nanoseconds, scan path/checksum, return counts, reflector-derived pose and rejection reason, filter acceptance |

A missing capture records delivery of the missing-observation notice at its deadline with no fabricated image. A capture whose scheduled delivery falls after termination retains a null actual-delivery time. Command interval semantics remain unchanged. Public `load_run`, `load_events`, and `replay_run` dispatch schema 3 to its dedicated validators. They check image hashes and confined paths, contiguous state/estimate timing, covariance shape and validity, capture/delivery causality, own-navigation agreement, and command/event dimensions.

Replay consumes saved PNGs, detections, estimates, commands, and events. It does not invoke Blender, PnP, the navigation filter, or physics. Its views add camera imagery and detected corners, estimated target geometry, navigation health/age, and estimation errors alongside physical telemetry. Fast tests use a small committed pair of actual Blender images for the image-to-command boundary check; fresh Blender qualification is an explicit CLI workload rather than a default unit-test dependency.

## Visual environment interface

`VisualEnv` (`visual_env.py`) owns the physical encounter: engine, target attitude controller, camera captures and faults, Blender rendering, image degradation, `estimate_pose`, hard boundaries, physical dwell scoring, termination, and environment-side records. A client never receives target truth, hidden initialization, scoring state, or outcome.

| Contract | Fields |
| --- | --- |
| `EnvObservation` | `time_ns`, `own: OwnNavigation`, `deliveries: tuple[Delivery, ...]`, `lidar: tuple[LidarDelivery, ...]` (empty without lidar) |
| `Delivery` | `image: ImageObservation` (actual pixels, or `None` for a missing capture), `pose: PoseEstimate`, `own_at_capture: OwnNavigation` |
| `LidarDelivery` | `scan: LidarScan` (range and retroreflector images), `pose: LidarPose`, `own_at_capture: OwnNavigation` |
| `ChaserCommand` | `force_body_n`, `torque_body_nm` (chaser body frame), optional `claim` in `success`, `retreated`, `navigation_lost`, `acquisition_failed` |

`reset(initial, output)` builds the engine from a resolved manifest, starts one Blender worker for the episode, and returns the tick-0 observation. `step(command)` holds the wrench for exactly one 0.1 s physics tick and returns the next observation, or `None` when the episode ends; `done`, `outcome`, `failure_reason`, and `physical_dwell_ns` are then available to the runner. The environment is a context manager that closes the renderer. Non-finite or wrong-shape commands and unknown claims raise `ValueError` without advancing. The Rust engine saturates force/torque norms; requested and applied values are both recorded.

Ordering matches M4: each tick records state, checks the hard boundaries (3 m COM, 2 m port gap), captures at ticks 0 mod 10, and delivers due images. On `step`, physical dwell is updated, then a claim or the time limit ends the episode without stepping physics, then the target torque updates at ticks 2 mod 10 and the engine steps. Physical dwell is tracked for both mission goals (`final_gap_m` 5 m for `success`, `retreat_gap_m` 30 m for `retreated`); an unsupported completion claim becomes `false_completion`. This equals M4's reset-on-goal-change rule because no valid 30 m dwell can span an abort below 30 m.

`VisualAgent` (`visual_agent.py`) is the reference client: it feeds each delivery to the frozen `VisualNavigation`, predicts to the tick, calls the frozen `VisualController`, signals reacquisition back to its own filter, and returns `ChaserCommand(force, torque, claim=decision.outcome)`. It keeps the onboard telemetry that schema 3 records (phase, desired gap, control updates, events, estimates, filter acceptance). `visual_control.py` and `navigation.py` remain byte-identical to `benchmarks/m4-qualified-source`; M6 lidar fusion is an optional constructor argument (see below) and leaves camera-only behavior unchanged.

`run_episode` (`visual_run.py`) drives one client through one environment episode and writes the schema-3 record; `run_visual` is the M4 wrapper that adds the fixed-case qualification gates. `initial_state(scenario, seed, ...)` resolves an episode: M4 hidden target errors scaled by `error_scale`, optional chaser disturbance, Sun angle about chaser body Z from the M4 lit face (M3's ±30° light convention), image degradation (blur then seeded noise per capture, `degrade_image`), and loss schedule. The six M4 cases are fixed condition sets.

M4 preservation: `tests/fixtures/m4_trace.json` holds digests of every record from a short stub-rendered M4 run (acquisition, missing and black frames, braking, reacquisition, navigation loss) captured before the refactor; the refactored path reproduces them exactly. Actual-Blender reruns of `nominal`/0 and `long-loss`/2 reproduce `runs/m4-nominal` and `runs/m4-long-loss` exactly in trajectory, events, estimates, pose detections, summaries, and decoded pixels (209/209 and 63/63 images). Blender PNG file bytes differ between renders of identical pixels, so image checksums are per-record integrity checks, not cross-run identity.

## M5 evaluation suite

```powershell
.\.venv\Scripts\rpod.exe m5-suite --output runs/m5 --calibration datasets/m3/calibration.json
```

`m5_suite.py` resolves all 40 episodes from `M5_SUITE_SEED = 5` and writes them to `suite.json` with the calibration hash, calibration range bins, and hashes of the frozen execution files before any rendering. Initialization seeds are 1000+index (M4 used 0–2). Each episode's conditions, faults, and per-capture image noise use separate recorded streams `[suite seed, index, 1|2|3]`.

| Category | Episodes | Conditions | Expected |
| --- | --- | --- | --- |
| Operational | 28 | M4 hidden-init distribution; Sun U[−30°, 30°]; M3 degradation (2 DN noise, 0.5 px blur) on even indices; chaser disturbance within M4 `disturbed` magnitudes (±[0.5, 0.5, 0.25] m LVLH, ±0.01 m/s, ≤3° about a random body axis) | success |
| Fault | 2 abort, 2 dropped frame, 2 extended loss | Operational conditions plus: abort at estimated gap U[10, 25] m; one missing capture at U[20, 150] s; 2–6 consecutive missing (even index) or black (odd) captures at U[20, 150] s | retreated, success, navigation_lost |
| Stress | 6 | One factor each from nominal M4 conditions: 2× init errors; 4 DN / 1.0 px degradation; oblique Sun ±[45°, 60°]; 2× chaser disturbance; plus two combinations (qualified degradation + dropped frame + Sun ±[30°, 45°]; 1.5× errors, 3 DN / 0.75 px, Sun ±[30°, 45°]) | none |

One lost capture lets the estimate age reach 2.1 s before the next accepted delivery; two consecutive losses reach 3.1 s, beyond the 3 s limit, so `navigation_lost` is the expected response. Operational episodes also run the paired truth-state oracle.

Episodes are written to `episodes/<id>.partial` and renamed on completion. Rerunning the command removes partial directories, runs missing episodes, and refuses to continue if `suite.json` differs from the current specs, calibration, or implementation hashes. The CLI copies the final report to `benchmarks/m5-evaluation.json`.

`suite_report` scores the suite from saved records only, without physics, rendering, or estimation. Each episode record must pass the schema-3 loader. The report gives outcomes, failed episodes with replayable paths, completion-time ratio, final pre-contact state, separation and actuator maxima, filtered target position/attitude error against truth during approach, retreat, and braking, accepted image-pose error by calibration range bin, fault response timing, and runtime (episode wall time, simulated/wall ratio, render, perception, autonomy step, Blender startup, and sampled Blender RSS).

M5 passes only if: all 28 operational episodes succeed with no false completion or boundary violation, actuator bounds respected, onboard and physical dwell met, and completion within 1.5× the paired oracle; all 6 fault episodes reach their expected outcome with braking at the first lost delivery (and reacquisition after a dropped frame, or final estimate age in (3.0, 3.1] s after extended loss, or an estimated-gap abort request); and every record is valid with uniform implementation and calibration hashes. Stress episodes have no outcome gate; any stress false completion or boundary violation is listed under `stress_findings`.

### M5 results

`runs/m5` passes all three gates; its report is copied to `benchmarks/m5-evaluation.json`. All 40 records are valid with uniform hashes, no episode failed, and there are no stress findings. Recomputing the report from the saved records reproduces it exactly. Replays of `00-operational-approach` and `30-fault-dropped-frame` export and pass `rrd verify`. 28/28 is a fixed-suite result, not a population reliability estimate.

| Category | Outcomes | Measured |
| --- | --- | --- |
| Operational | 28/28 success | 205–214 s; completion 1.004× the paired oracle (median; max 1.005×); final gap 4.999 m median; final port speed ≤ 0.0026 m/s; minimum port gap 4.96 m |
| Fault: abort | 2/2 retreated | abort requests at 23.2 and 13.7 m estimated gap; final gap 29.8 m at 323 and 280 s |
| Fault: dropped frame | 2/2 success | brake at the lost delivery, reacquisition 2.0 s later; success at 271 and 265 s; minimum port gap 4.46 and 4.57 m |
| Fault: extended loss | 2/2 navigation_lost | at 55.1 and 42.1 s, final estimate age 3.1 s |
| Stress | 6/6 success | slowest: qualified degradation + dropped frame + oblique Sun (249 s, minimum port gap 4.76 m); heavy degradation (234 s, 3 of 234 deliveries rejected) |

Over operational episodes, filtered target position error has median 0.036 m, P95 0.121 m, and max 0.189 m; attitude error has median 0.066°, P95 0.30°, and max 1.96°. Accepted image-pose translation error grows with range: median 3.8 mm at 5.7–8.45 m, 15 mm at 8.45–15.95 m, 50 mm at 15.95–25.95 m, and 106 mm at 25.95–31.95 m. Maximum chaser force is 1.71 N and maximum torque 0.34 N·m.

A dropped frame late in the approach (captures 113–138, 11–15 m gap, closing at 0.15 m/s) triggers braking. After reacquisition, the approach passes the 5 m hold point by up to 0.54 m around 210 s and completes 55–60 s later than operational episodes. M4's `short-loss` drop at 60 s reached a minimum port gap of 4.94 m. The fault gate does not bound this overshoot, which stays far from the 2 m hard limit.

The suite took 14,390 s (4.0 h) on an unisolated desktop. Another CPU-heavy workload ran during roughly episodes 4–16 and about doubled their wall time. Episode wall time has median 335 s; per-image render median is 0.93 s, perception 0.44 s, and the autonomy step 1.0 ms; sampled Blender RSS is about 266 MiB. Perception on degraded images (median per-episode mean 0.63 s per image) is about 8× slower than on clean images (0.08 s).

## Lidar sensor and fusion (M6)

```powershell
.\.venv\Scripts\rpod.exe m5-suite --lidar --output runs/m6-lidar --calibration datasets/m3/calibration.json
```

**Sensor model** (`lidar.py`). `LidarModel` is a pinhole flash lidar: 256×256 pixels, 30° field of view (0.117° per pixel), the camera's optical axes and `R_BC`, mounted at `[-0.2, 0.55, 0.2] m` on the chaser body. Scans run at 1 Hz on ticks 5 mod 10, half a period after camera captures, and are delivered 0.2 s later. `scan_target` ray-casts the rendered geometry: the 2 m target cube and the four 0.71 m marker pads, two of which float 0.3 m in front of the docking face. It also includes four retroreflectors that exist only in the lidar model, not in camera images: 0.06 m radius discs 2 mm proud of the −Y face at `(0,−1,0.85)`, `(0.8,−1,0)`, `(0,−1,−0.45)` and `(−0.6,−1,0)` in target body coordinates, with distinct pairwise spacings. Every return gets Gaussian range noise of σ 0.02 m, seeded by `[episode lidar seed, scan ID]`, and the maximum range is 100 m. A reflector reads as retroreflective only within 45° of its normal; outside that cone it returns like the face. The model has no intensity falloff, beam divergence, multipath, motion distortion, solar background or random dropouts.

**Pose estimate.** `estimate_lidar_pose` groups retroreflective pixels into 8-connected blobs and takes the mean 3D point of each blob. Each centroid's variance is `(σ_range² + (pixel angle · range)²/12) / hits + (2 mm)²`. With three or four blobs, a weighted Kabsch fit tries every assignment to the reflector layout and keeps the lowest RMS. Fewer than three or more than four blobs gives `reflector_count`, and an RMS above 5 cm gives `reflector_mismatch`. The 6×6 covariance (position, then left attitude, in lidar axes) is the inverse of the fit's linearized information. Because no prior pose is needed, lidar can acquire on its own. A Monte Carlo of 60 scans per gap, with ±2° target attitude and lateral offsets, found every scan valid:

| Port gap | Hits per reflector | Translation error median / max | Rotation error median / max | Mean NEES (6 = consistent) |
| --- | --- | --- | --- | --- |
| 5 m | ~105 | 0.33 / 0.65 cm | 0.12 / 0.36° | 4.1 |
| 10 m | ~26 | 0.59 / 1.22 cm | 0.32 / 0.64° | 3.3 |
| 20 m | ~7 | 1.09 / 2.74 cm | 0.64 / 1.44° | 3.2 |
| 30 m | ~2–3 | 1.61 / 4.73 cm | 0.98 / 2.08° | 2.8 |

The covariance is mildly conservative. Lidar position is far more accurate than the camera at long range (M3 30 m translation P95 is 12 cm), while the camera's attitude is better. The filter weights each sensor by its covariance.

**Fusion.** `LidarNavigation` subclasses the frozen `VisualNavigation` and only adds a measurement path for `LidarPose`:
- It places the pose using the lidar mount and capture-time own navigation, and rotates the covariance to inertial.
- The delayed-update replay, two-second lag buffer, Mahalanobis² gate of 30, and capture-age health are unchanged.
- Lidar capture times (x.5 s) never collide with camera captures (whole seconds).

`VisualAgent(mission, camera, calibration, lidar)` feeds both sensors to that filter. In a tick, any accepted delivery reports an accepted update. A failed delivery reports a failure, which makes `VisualController` brake, only if no capture from either sensor was accepted in the previous 1 s; otherwise the tick reports no update. Without lidar, a failed camera delivery always arrives 1.2 s after the previous capture, so behavior equals M4: the M4 trace digest still matches exactly. Acquisition still needs three accepted updates, which can now come from either sensor.

**Records and replay.** Lidar episodes add the following to the record:
- The manifest gets a `lidar` block (model and noise seed) and lidar entries in `navigation_configuration`.
- Each scan is saved as `lidar/*.npz`, with a matching row in `lidar.jsonl`.
- The summary gets `lidar_scans_delivered`, `lidar_accepted_observations` and `performance.lidar_perception_seconds`.

The schema-3 loader validates lidar scan timing, file checksums, image shapes and delivered poses. It counts lidar acceptances as observations that estimates may use. With lidar, a failed delivery may be recorded as no update. Replay shows each delivered scan in target LVLH: every fourth return plus all reflector returns, with lidar status lines. NPZ files embed zip timestamps, so like PNGs their checksums are per-record integrity checks.

**Evaluation.** `m5-suite --lidar` runs the same 40 frozen episodes (seeds, conditions and faults) with a lidar noise stream `[5, index, 4]` and `"sensors": ["camera", "lidar"]` in `suite.json`. Extended camera loss is expected to succeed. The gates are:
- Operational and abort episodes use the M5 rules.
- Dropped-frame and extended-loss episodes must succeed without any brake, because lidar captures cover the lost camera deliveries.
- Records must be valid and uniform.

The report adds per-episode lidar counts and pose errors, `lidar_pose_by_range` per category, and `lidar_perception_seconds`. The CLI copies it to `benchmarks/m6-lidar-evaluation.json`.

**Verification.** Six new fast tests cover:
- scan geometry, the acceptance cone and the range limit;
- covariance consistency at 5 and 30 m;
- rejection with too few reflectors;
- inertial placement of a lidar measurement and rejection of a singular covariance;
- a stub-rendered closed-loop episode with 40 s of black camera frames, where lidar alone acquires, never brakes, tracks within 0.1 m, and writes a record the loader accepts;
- lidar suite specs that differ from M5 only in noise streams and the extended-loss expectation.

The report-recompute test also covers a lidar suite. With M6 code, recomputing the M5 report from `runs/m5` gives an identical report apart from `report_implementation_sha256`, and all gates still pass. A lidar replay exports and passes `rrd verify`, and its scan points land on the target: within ±1 m in body x and z, at y ≈ −1.0 m (face) and −1.3 m (floating pads). In numpy, a scan takes about 0.11 s to ray-cast and about 0.02 s to estimate.

### M6 results

`runs/m6-lidar` passes all three gates, and its report is copied to `benchmarks/m6-lidar-evaluation.json`.
- **Records:** all 40 are valid with uniform hashes. No episode failed, and there are no stress findings. Recomputing the report from the saved records reproduces it exactly.
- **Replay:** the replay of `32-fault-extended-loss` exports and passes `rrd verify`.
- **Acceptance:** every one of the 8,384 delivered lidar scans was accepted. The camera accepted 8,375 of 8,396 delivered images.
- **Resume:** the run was stopped after episode 14 and resumed with the same command, and the frozen-suite check accepted it.

Outcomes, against the camera-only M5 suite on the same episodes:

| Category | M5 (camera only) | M6 (camera + lidar) |
| --- | --- | --- |
| Operational | 28/28 success; completion 1.004× the paired oracle | 28/28 success at 204–212 s; completion 0.995× the oracle (median; max 0.997×); final port speed ≤ 0.0031 m/s; minimum port gap 4.92 m |
| Fault: abort | 2/2 retreated | 2/2 retreated; final gap 29.9 m at 209 and 278 s |
| Fault: dropped frame | brake, reacquisition 2.0 s later; success at 271 and 265 s; minimum port gap 4.46 and 4.57 m | no brake; success at 204 and 209 s; minimum port gap 4.93 and 4.94 m |
| Fault: extended loss | `navigation_lost` at 55.1 and 42.1 s | no brake; success at 203 and 212 s |
| Stress | 6/6 success | 6/6 success; the degraded + dropped-frame + oblique-Sun case no longer brakes (205 s, minimum port gap 4.97 m) |

Lidar removes the M5 dropped-frame overshoot of the 5 m hold point. Chaser actuator maxima match M5 (1.72 N, 0.34 N·m).

Filtered target error over operational episodes:

| Error | M5 median / P95 / max | M6 median / P95 / max |
| --- | --- | --- |
| Position | 3.6 / 12.1 / 18.9 cm | 0.52 / 1.36 / 5.98 cm |
| Attitude | 0.066 / 0.30 / 1.96° | 0.110 / 0.34 / 2.52° |

Per-delivery pose error by camera range (operational median, translation / rotation):

| Camera range | Camera | Lidar |
| --- | --- | --- |
| 5.7–8.45 m | 3.8 mm / 0.05° | 3.5 mm / 0.15° |
| 8.45–15.95 m | 15 mm / 0.10° | 6.2 mm / 0.30° |
| 15.95–25.95 m | 50 mm / 0.17° | 11 mm / 0.57° |
| 25.95–31.95 m | 106 mm / 0.26° | 17 mm / 0.85° |

Fusion makes filtered attitude slightly worse, even though the filter weights each sensor by its covariance. The lidar attitude is 3–4× noisier than the camera's at every range. This suggests the lidar attitude covariance is optimistic in closed loop, or its errors are correlated between scans. The cause has not been investigated.

Runtime, on an unisolated desktop:
- **Total:** episode wall times sum to 15,839 s (4.4 h).
- **Contention:** episodes 0–14 ran alongside another CPU-heavy workload (median 626 s each). Episodes 15–39 ran alone after the resume (median 291 s).
- **Per image:** render median 1.06 s, perception 0.38 s.
- **Per scan:** lidar pose estimation 8.8 ms median. The scan ray-cast is not timed separately.

## Realistic scene (M7, upgraded in M8)

```powershell
.\.venv\Scripts\rpod.exe perception-eval --suite m3 --scene realistic --earth vbar --output datasets/m8-vbar
.\.venv\Scripts\rpod.exe perception-eval --suite m3 --scene realistic --earth rbar --output datasets/m8-rbar
.\.venv\Scripts\rpod.exe m5-suite --scene realistic --output runs/m8-realistic --calibration datasets/m8-vbar/calibration.json
```

M8 replaced the M7 realistic scene in place, raised its Sun, and reran the suite. The M7 code is preserved in git commit `a03d4f0`. `runs/m7-realistic`, `datasets/m7-*` and `benchmarks/m7-realistic-evaluation.json` were produced by that commit; the current code cannot resume them, because the frozen hashes differ.

**Scene** (`realistic_scene()` in `rendering.py`, built by `blender_worker.py`). The simple scene is unchanged: re-rendered M3 fixtures are pixel-identical to `datasets/m3`. The realistic scene keeps the calibrated camera, the 2 m target body and the four marker pads, and adds:
- **Docking face (as in M7):**
  - a white painted −Y face;
  - a 0.17 m aluminum docking ring with a dark hatch;
  - standoff posts under the two floating pads;
  - gold handrails beside the face.

  The four lidar retroreflectors appear as glossy dark discs at their lidar positions.

  In M8 the 2 m body is drawn only as a 5 cm docking plate, from y = −1 to −0.95.
- **Bus and wings (M8):** NASA 3D Resources' QuikSCAT model (glTF), mounted behind the plate:
  - scaled 1.1× so its 1.8 m bus spans the 2 m body, and moved (0, 0.08, −0.05) m;
  - its scatterometer dish below, and its two three-panel solar wings, about 8 m tip to tip;
  - wings on a drive about body X through the model's hinge axis (model z = 0.7 m), turned each render so the cell side faces the Sun.

  The bus is about 2.5 m tall and the dish hangs about 2.3 m below. Physics and the lidar still model only the 2 m body, pads and reflectors.
- **Lighting:**
  - Space is black, with no ambient term.
  - The Sun has strength 3, so a white diffuse face lit head-on reads 0.95, and a 0.53° disc.
  - Earth is a 6,378,137 m sphere placed each frame from the camera's inertial pose. It can shadow the Sun.
- **Earth (M8):** `earth_pose` turns Earth about inertial +Z at the sidereal rate (`earth_rotation_ne`). The simulation has no epoch, so the angle is chosen: −37.8° at t = 0, which puts the approaches over southern India.
  - **Surface:** NASA imagery looked up in the shader by equirectangular coordinates, lighting both camera rays and earthshine:
    - Blue Marble Next Generation (July 2004, topography and bathymetry, 21600×10800), with water darkened through a blue-minus-red mask;
    - Blue Marble clouds (8192×4096).
  - **Atmosphere:** a transparent 60 km shell seen only by camera rays. Its glow follows each ray's path length through the air and the Sun's elevation at the path's midpoint. The full model is under Showcase movie.
  - **Emitters:** Earth and air are not sampled as lights.
  - **No per-render material edits:** the Earth and air shaders read the Sun from `sun_frame`, an Earth-centred empty whose +Z each render turns toward the Sun; the shell is parented to it. A first M8 attempt instead edited shader node values every render. In 5 of 38 episodes, the four white marker pads then intermittently rendered pure black mid-run, and navigation was lost. A fresh renderer drew the same frames correctly, and the fault could not be reproduced on demand. That attempt was discarded.
  - **Not in the camera scene:** city lights and stars. A camera exposed for the sunlit target would not record them; only the showcase adds them.
- **Assets** (`scene_assets()` in `rendering.py`):
  - four NASA images and two NASA 3D Resources models (QuikSCAT for the target, IBEX for the showcase chaser), about 205 MB, downloaded to `.tools/showcase-assets/` on first use;
  - each checked against its SHA-256 once per process when a realistic renderer starts;
  - recorded as `scene_assets_sha256` in the render metadata.
- **Rendering:**
  - Cycles on CUDA (GTX 1060 6 GB), 16 samples, two bounces (one diffuse, one glossy), indirect clamp 10, no denoising. OptiX fails to compile its kernel on this GPU and driver.
  - Repeated CUDA renders differ by at most 1 DN in fewer than ten pixels per frame (CPU + CUDA hybrid rendering differed by up to 14 DN). Realistic re-runs are therefore not bitwise repeatable, but records keep the actual images and replay is exact.
  - Render metadata adds the scene, device, devices, bounce limit and Sun size.

**Geometry consequences:**
- **Earth out of view:** on the along-track (V-bar) approach, Earth's center lies about 90° from the optical axis and its disk never enters the image; a test checks that no border ray meets it. Earth's only effect there is reflected light.
- **Sun convention:** M5's Sun angle turns the Sun within the orbit plane. With Earth present, angles below −19.8° (the horizon dip at 400 km) would put 8 of the 40 suite episodes in Earth's shadow. The realistic scene therefore applies the same angle about chaser body X, the handoff's local vertical (`sunlight_direction`). Episode specs are otherwise identical.
  - **M7:** the Sun started level with the approach axis. It lies behind the chaser, against the direction of orbital motion, so it sets at up to the orbital rate (0.065°/s). It started 17.5–22.5° above Earth's limb and would reach the limb after 287–528 s. Episode `29` ran past its sunset.
  - **M8:** the Sun is first raised `REALISTIC_SUN_ELEVATION_DEG` = 30° above the approach axis toward the local vertical, then turned. It then stays above the limb for at least the 600 s step limit. A test checks every episode: the start elevation, minus the orbital rate times 600 s, must stay above the limb plus the Sun's radius. Earth below the spacecraft is now in daylight.
- **Fixture Earth placement:**
  - `--earth vbar` places Earth 6,778 km below the image (fixture −Z), where M3's ±30° fixture Sun stays on the horizon.
  - `--earth rbar` places Earth beyond the target along the line of sight, as when approaching from above. The Sun is overhead, so the camera sees lit Earth and its sun glint directly behind the target.

**Qualification results.** Both realistic fixture datasets pass every M3 gate:
- 72/72 clean and 72/72 degraded positives accepted;
- all 24 negatives rejected, including the R-bar frames that show only Earth;
- no gross errors;
- the camera geometry check reproduces 0.00015 px projection and 0.0625 px raster error on CUDA.

Clean P95 translation / rotation errors (M8, `datasets/m8-*`) match the simple scene within a few percent:

| Axial gap | M3 simple | Realistic V-bar | Realistic R-bar |
| --- | --- | --- | --- |
| 5 m | 3.4 mm / 0.044° | 3.3 mm / 0.044° | 3.3 mm / 0.044° |
| 10 m | 14.8 mm / 0.093° | 14.8 mm / 0.089° | 14.8 mm / 0.087° |
| 20 m | 54.5 mm / 0.188° | 54.4 mm / 0.193° | 54.6 mm / 0.200° |
| 30 m | 124 mm / 0.438° | 124 mm / 0.446° | 122 mm / 0.443° |
| 32 m | 195 mm / 0.578° | 194 mm / 0.545° | 194 mm / 0.556° |

This realism does not break the marker pipeline. The matte, high-contrast markers on a sunlit white face dominate detection, and the clutter (the QuikSCAT bus and wings, clouds, glint) never decodes as a valid marker. M8 median render times were 0.71 s per fixture for V-bar and 1.58 s for R-bar, where Earth fills the frame. Median estimation times were 0.13 s and 0.43 s.

`calibration.json` records `scene` and `earth_geometry`, and `m5-suite --scene realistic` refuses a calibration qualified on another scene. The realistic suite puts `scene: realistic` in every episode spec and in `suite.json`, adds `scene` to the report, and copies the report to `benchmarks/m8-realistic-evaluation.json` (`m7-realistic-evaluation.json` at commit `a03d4f0`).

### M7 results

`runs/m7-realistic` passes the operational and integrity gates and fails the fault gate. Its report is copied to `benchmarks/m7-realistic-evaluation.json`.
- **Records:** all 40 are valid with uniform hashes. Recomputing the report from the saved records reproduces it exactly.
- **Replays:** `00-operational-approach` and `29-fault-abort` export and pass `rrd verify`.
- **Acceptance:** the camera accepted 8,287 of 8,297 delivered images (M5: 8,444 of 8,449).

Outcomes, against the simple-scene M5 suite on the same episodes:

| Category | M5 (simple scene) | M7 (realistic scene) |
| --- | --- | --- |
| Operational | 28/28 success; completion 1.004× the paired oracle (median; max 1.005×); minimum port gap 4.96 m | 28/28 success at 205–224 s; completion 1.004× the paired oracle (median; max 1.061×); final port speed ≤ 0.0026 m/s; minimum port gap 4.96 m |
| Fault: abort | 2/2 retreated, at 323 and 280 s | 1/2: `28` retreated at 309 s; `29` ended `navigation_lost` at 335 s |
| Fault: dropped frame | success at 271 and 265 s; minimum port gap 4.46 and 4.57 m | the same |
| Fault: extended loss | `navigation_lost` at 55.1 and 42.1 s | the same |
| Stress | 6/6 success | 5/6 success; heavy degradation ended `navigation_lost` at 21.1 s. No false completion or boundary violation. |

The slowest operational episode, `16`, rejected one three-marker estimate at 17 s. It braked, reacquired 2 s later, and docked 12 s later than in M5.

**Episode `29-fault-abort`** failed for two reasons:
- **Slow retreat confirmation.** The abort request came at 131.4 s, as in M5. At the 30 m hold, the estimated port speed crossed the 0.01 m/s onboard dwell limit in 56% of ticks, which kept resetting the 10 s dwell. This was already slow in M5: confirming the retreat took 48 s of hold for this episode and 202 s for episode `28`.
- **Eclipse.** At 324.9 s the Sun's disc reached Earth's limb. Target brightness then fell with the Sun's height, and from capture 333 (fully black at 334) no markers were found. Two consecutive failed deliveries gave `navigation_lost` at 335.1 s, the filter's designed safe response. There was no false completion or boundary violation.

Every other episode ended before its sunset; `28` finished with the Sun 0.9° above the limb. The simple scene has no Earth to cast a shadow, so M5 never exposed this.

**Episode `35-stress-heavy-degradation`** (4 DN noise, 1.0 px blur, 30 m) often decoded only two or three markers. It braked at 5.2 and 9.2 s and reacquired each time. At 19–20 s the navigation filter rejected two consecutive three-marker estimates, ending in `navigation_lost`. M5 survived the same case with nearly identical detections, braking three times, so this stress case is marginal in both scenes.

Filtered target error over operational episodes:

| Error | M5 median / P95 / max | M7 median / P95 / max |
| --- | --- | --- |
| Position | 3.6 / 12.1 / 18.9 cm | 3.6 / 12.0 / 18.9 cm |
| Attitude | 0.066 / 0.30 / 1.96° | 0.068 / 0.42 / 1.93° |

Accepted image-pose translation error by camera range matches M5 within about 1 mm: medians of 3.8, 15, 51 and 105 mm in the same four range bins. Rotation-error P95 is higher at close range: 0.16° against 0.12° at 5.7–8.45 m, 0.34° against 0.26° at 8.45–15.95 m, and 0.48° against 0.40° at 15.95–25.95 m. Chaser actuator maxima match M5.

Runtime, on an unisolated desktop, with the suite paused while another heavy workload ran:
- **Total:** episode wall times sum to 5,826 s (1.6 h), against 14,390 s for M5.
- **Per episode:** median 142 s. Episodes 0 and 1 took 473 and 414 s, and no other episode took more than 200 s. The slow start was not investigated.
- **Per image:** GPU render median 0.37 s (P95 0.73 s), perception 0.095 s. Renderer startup median is 1.5 s.
- **Memory:** sampled Blender RSS is about 516 MiB, against 266 MiB for the simple scene.

### M8 results

`runs/m8-realistic` passes all three gates. Its report is copied to `benchmarks/m8-realistic-evaluation.json`.
- **Records:** all 40 are valid with uniform hashes. Recomputing the report from the saved records reproduces it exactly.
- **Replays:** `00-operational-approach` and `29-fault-abort` export and pass `rrd verify`.
- **Acceptance:** the camera accepted 8,420 of 8,432 delivered images (M7: 8,287 of 8,297).
- **No black pads:** apart from the injected black frames of `33`, every image processed during an episode produced a valid pose estimate, so the black-pad fault of the first M8 attempt did not recur.

Outcomes, against M7 on the same episodes:

| Category | M7 | M8 |
| --- | --- | --- |
| Operational | 28/28 success at 205–224 s; completion 1.004× the paired oracle (median; max 1.061×); final port speed ≤ 0.0026 m/s; minimum port gap 4.96 m | 28/28 success at 205–234 s; completion 1.004× (median; max 1.107×); final port speed ≤ 0.0028 m/s; minimum port gap 4.95 m |
| Fault: abort | 1/2: `28` retreated at 309 s; `29` ended `navigation_lost` at 335 s | 2/2 retreated, at 243 and 280 s |
| Fault: dropped frame | success at 271 and 265 s; minimum port gap 4.46 and 4.57 m | the same |
| Fault: extended loss | `navigation_lost` at 55.1 and 42.1 s | the same |
| Stress | 5/6 success; heavy degradation ended `navigation_lost` at 21.1 s | 6/6 success; heavy degradation braked three times and docked at 237 s |

No episode had a false completion, boundary violation or actuator-bound breach. The two slowest operational episodes braked after rejected estimates and reacquired: `12` at 14 and 34 s (docked at 234 s) and `24` at 24 s (227 s). Episode `29` now retreats at 280 s, as in M5, with the Sun well above the limb. `35` remains marginal: the same stress case survived in M5, failed in M7 and survives in M8.

Filtered target error over operational episodes:

| Error | M7 median / P95 / max | M8 median / P95 / max |
| --- | --- | --- |
| Position | 3.6 / 12.0 / 18.9 cm | 3.5 / 11.8 / 18.6 cm |
| Attitude | 0.068 / 0.42 / 1.93° | 0.065 / 0.33 / 1.98° |

Accepted image-pose translation error medians are 3.7, 15, 49 and 104 mm in the four range bins (M7: 3.8, 15, 51, 105). Rotation-error P95 is back near M5: 0.11°, 0.27°, 0.42° and 0.72° (M7: 0.16°, 0.34°, 0.48°, 0.72°). Chaser actuator maxima match M7.

Runtime, on an unisolated desktop:
- **Total:** episode wall times sum to 9,240 s (2.6 h), against 5,826 s for M7.
- **Per episode:** median 218 s, maximum 410 s.
- **Per image:** GPU render median 0.67 s (P95 1.39 s), perception 0.15 s. Renderer startup median is 2.3 s.
- **Memory:** sampled Blender RSS is about 1.8 GiB, mostly the 21600×10800 Earth image and the QuikSCAT model.

## Showcase movie

`rpod showcase RUN_DIR --output OUT_DIR` renders a saved schema-3 visual run as a realistic outside-view movie. It is presentation-only. It reads the validated record (`load_visual_run`), never constructs an engine or autonomy, and is not part of any suite or its frozen hashes.

**Renderer.** `showcase_worker.py` runs inside Blender and `showcase.py` drives it through a `BlenderRenderer` subclass that launches that worker. The worker subclasses the realistic `SceneRenderer`, so the target, Earth, atmosphere and Sun are exactly the docking camera's (see Realistic scene). It adds the following. All of it is visual only.
- **Chaser (1 m):** NASA 3D Resources' IBEX model (glTF), scaled 0.72× to about 1.1 m across.
  - It is turned so its solar deck faces aft (−Y) and its base sits at y = 0.45 m.
  - On the base are a docking ring on an adapter, with guide petals, and the camera and lidar heads at their mounts.
- **Thruster plumes:** four corner pods on the hull, each with nozzles along both ±Y directions and its two outward lateral axes. For each body axis, the recorded `applied_forces_body_n` of the chaser, divided by its 2 N force limit, sets the throttle of the nozzles whose exhaust opposes that force (`plume_throttles`).
  - Each fired nozzle shows a faint transparent 0.5 m cone. It glows 0.6 × throttle, brightest at the nozzle.
  - Torques do not fire plumes.
- **Presentation extras**, passed as scene options that the docking camera leaves off:
  - **City lights:** Black Marble 2016 (13500×6750) on the night side, at strength 0.05, dimmed under clouds.
  - **Stars:** NASA SVS Deep Star Maps 2020 (8K EXR, celestial coordinates), shown at strength 0.15 on a 5×10⁷ m sphere around the camera. Only camera rays see it. Each render turns the sphere to inertial axes.
- **Atmosphere model**, shared with the docking camera: a transparent shell 60 km above the surface, seen only by camera rays.
  - Each front face emits the light scattered along its ray's path L through the shell.
  - **Strength:** `0.6 × (L / L_max)²`, so the horizon is brightest and the view straight down stays clear.
  - **Colour:** follows the Sun's elevation at the path midpoint. It is blue by day, orange to red through twilight, and black at night.
- **Image settings:**
  - AgX view with the Medium High Contrast look, exposure +0.5, and Sun strength 4;
  - 8 samples, then CPU OpenImageDenoise at fast quality and prefilter, on every CPU thread. Denoised, 8 samples differ from 16 by 39 dB PSNR over the frame and 33 dB on the target, which is not visible, at half the GPU time;
  - no adaptive sampling: Cycles tests convergence only every 16 samples, so it never acts at these counts (with it on, 16-sample frames changed by at most 1 DN);
  - up to 6 bounces: 2 diffuse and 3 glossy.
- **Every run uses this scene,** including simple-scene runs. A run from before M8 is filmed under its own recorded Sun, but its docking-camera inset shows the older images.

`showcase.json` records the asset URLs, hashes and credits.

**Camera rules.** Both shots use LVLH +radial as up and a 50° horizontal field of view at 1920×1080. Both take their normal side from the Sun: the sign of the toward-Sun normal component, so the camera sees the lit sides. In every suite episode the Sun is on the −along-track side.
- **Wide shot** (`chase_camera`), establishing both spacecraft over Earth for the first 30 s of the run (3 s of movie):
  - **Aim:** the midpoint of the two centers of mass.
  - **Direction from that midpoint**, in target LVLH: `normalize(0.25, −0.2, ±0.95)` (radial, along-track, normal). That is slightly above and behind the chaser, looking across the approach.
  - **Distance:** `max(1.5 × separation, 12 m)`.
- **Close shot** (`close_camera`), for the rest of the run, retreats included. In the wide shot the 1.1 m chaser is only 50–110 px across beside the 7 m target; here, 5.7 m from the camera, it stays about 400 px across at every gap:
  - **Position:** over the chaser's shoulder, at LVLH `(1.2, −5.0, ±2.4)` m from its center of mass.
  - **Aim:** midway between the directions to the chaser and to the target's docking port. Between 30 m and 5 m of port gap, the chaser's center stays near the lower-left third of the frame and the port near the upper-right third.

Each frame places the target, chaser and Earth from recorded truth, with the Sun along `sunlight_direction_inertial`. Fast tests check:
- the wide shot keeps both centers of mass inside the image with a 10% margin, at 30 m and 6.5 m separation;
- the wide shot is on the sunlit normal side for both signs of the Sun's normal component, and Earth's center is below the image;
- the close shot keeps the chaser and the port inside the image with a 10% margin at 30 m and 1 m gaps, the chaser nearer, with Earth's center below the image;
- plumes fire against the applied force.

**Rendering:**
- Every third tick is rendered, so each frame covers 0.3 s of simulated time. Rendering uses CUDA.
- A second thread blooms, overlays and encodes each frame while Blender renders the next.
- The video is 30 fps (9× real time) MPEG-4 Part 2 (`mp4v`). This OpenCV build cannot open an H.264 encoder.
- **Cost** at 1080p (GTX 1060 6 GB, i7-8700K): 2.0–2.8 s per frame, against 3.3 s with the earlier 16 samples.
  - The cost grows as the target fills the close shot.
  - Run `29` (933 frames, never nearer than 13 m) took 31 min.
  - Run `00` (694 frames, to 5 m) took 28 min, against 39 min before.
  - A run `29` frame splits into:
    - about 0.9 s of GPU sampling, two thirds of it Earth's surface and the atmosphere shell;
    - about 0.8 s for the CPU denoiser on all 12 threads (1.0 s on the base worker's 6);
    - about 0.4 s for the BVH update and the PNG save.
- **Tried without gain:**
  - the OptiX denoiser (no faster);
  - half-size Earth textures (no faster);
  - fewer bounces (−1%);
  - two Blender workers rendering alternate frames (−11%, at 4.9 of the GPU's 6 GB).

**Overlay** (drawn by OpenCV on each clean render; sizes scale with the image height from a 720-line base):
- **Bloom:** a soft glow on highlights: threshold 190, σ 9 px, gain 0.8.
- **Inset:** the run's most recent docking-camera image captured at or before that tick, or a "no image" box for a missing capture. It is 405 px wide, at the bottom centre in the wide shot and the bottom right in the close shot, clear of the spacecraft.
- **Labels:** run name, elapsed time, port gap and supervisor phase.

**Outputs:**
- `frames/NNNNNN.png`: clean renders, without bloom or overlay.
- `showcase.mp4`.
- `showcase.json`: source manifest hash, camera rule, asset URLs, hashes and credits, render settings, and timing.
- The marker assets.

**Limitations:**
- City lights, stars and plumes are brighter than a camera exposed for the sunlit spacecraft would record. Real cold-gas and monopropellant plumes are nearly invisible.
- The atmosphere uses single scattering in a uniform 60 km shell. There is no airglow or cloud shadow.

## Source layout and generated output

- `crates/rpod-core`: native dynamics and Rust acceptance tests.
- `crates/rpod-python`: PyO3 interface and compiler provenance.
- `src/rpod`: scenarios, frames, M1/M2 approach, recording, visualization, benchmark, and CLI; camera, rendering (simple and realistic scenes, `blender_worker.py`), perception, and M3 evaluation; navigation filter and visual controller; visual environment, reference client, M4 runner, schema-3 records/replay, and the M5 suite; flash-lidar model, reflector pose estimate, and lidar filter measurement (`lidar.py`); presentation-only showcase movie (`showcase.py`, `showcase_worker.py`).
- `tests`: Python integration and contract checks; `tests/fixtures` holds two actual Blender images and the pre-refactor M4 trace digest.

Frozen-file hashes are taken over raw bytes, so `.gitattributes` checks text out with LF, except `blender_worker.py`. That file is stored byte-exact with CRLF line endings, as the M8 suite hashed it. `.gitignore` excludes the environment, workspace-local uv cache, target directory, compiled extensions, Python caches, portable tools, and generated `runs/`, `datasets/`, and `benchmarks/` outputs. Blender is installed locally for M3/M4 and remains separate from the physics runtime. Contact engines and learning frameworks are not installed.
