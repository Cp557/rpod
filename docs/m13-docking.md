# M13: docking with a tumbling satellite

Through M12 the chaser stopped at a 5 m hold off the tumbling satellite's docking port. M13 goes the rest of the way: it closes on the rotating port, is captured, and stops the joined pair's tumble. It does this with nothing but its own lidar against an unprepared target.

[REFERENCE.md](../REFERENCE.md#docking-with-the-tumbling-target-m13) is the full specification.

## What made the last five meters hard

The first attempts failed in instructive ways, each traced on episodes outside the suite and fixed for a stated reason.

- **The lidar loses the target's structure up close.** M11's 30-degree lidar sees only a patch of the flat docking face inside about 1.5 m. M13's chaser instead carries one 90-degree lidar on its docking axis, which keeps the whole face in view down to a meter.
- **A flat face can't pin down attitude.** Near the face, roll about the docking axis is unobservable, and a tilt about the target's center (a meter behind the face) looks like a sideways shift. Fitting all six degrees of freedom let attitude wander by degrees and dragged the position fit by 3–7 cm.
  - Inside a 2.5 m gap the tracker therefore holds the relative attitude, which the chaser's attitude loop keeps steady, and fits position only.
  - The navigation filter treats those attitudes as uninformative, so its attitude comes from the torque-free model.
  - Fusing the held attitude made it worse: the chaser turned with the filter's attitude, the held attitude turned with the chaser, and the estimate walked off by degrees.
- **Inside a meter, the fit itself becomes ambiguous.** A second minimum appears about 4.5 cm away, the depth of the docking ring. So the lidar reports nothing inside a 1.3 m gap. The chaser commits there and closes the last 1.24 m, about 31 s, on prediction.
- **Prediction is only as good as the velocity estimate.** M4's filter let the target's estimated velocity jitter by ±3 mm/s, which 31 s of coasting turns into centimeters.
  - The target's orbit is modeled exactly, so near the target the filter's translation process noise drops a thousandfold.
  - Far out it keeps M4's level. With the low noise everywhere, the filter locked onto early errors and refused good scans.
- **The calibration must match the approach.** A calibration at random attitudes underestimated the errors seen along the docking axis two- to sixfold. The final one samples the approach itself, with a floor at M11's level for the fits' heavier tails.

## Contact and capture

There is no contact model; capture is a stated envelope and a latch.
- **Contact:** an aligned chaser touches the hatch, 6.3 cm proud of the port.
- **The envelope:** at contact the environment checks lateral offset ≤ 5 cm, alignment ≤ 3°, closing speed 0.01–0.06 m/s, lateral speed ≤ 2 cm/s and relative rate ≤ 0.2°/s.
- **Inside it,** a soft-capture latch merges the two spacecraft into one rigid body, conserving linear and angular momentum. The engine keeps stepping the stack, and the records still show both spacecraft.
- **Outside it,** the episode fails.
- **After the latch,** the chaser's attitude control damps the stack's tumble at its 0.5 N·m limit.

## Results

All 28 suite episodes docked: the same conditions, tumbles and seeds as M11's operational episodes, at 0.2–1.0°/s. 27 of them never braked.

| At contact | Median | Worst | Capture envelope |
| --- | --- | --- | --- |
| Lateral offset | 0.6 cm | 1.1 cm | 5 cm |
| Alignment | 0.06° | 0.13° | 3° |
| Closing speed | 0.039 m/s | 0.039 m/s | 0.01–0.06 m/s |

- **After a 31 s blind coast,** the chaser's estimate of the port was still within 1.2 cm laterally and 1.5 cm along the axis.
- **Capture** came about 342 s after starting 30 m out. The detumble then took 11–20 s, longer for faster tumbles.
- **Fuel grows with the square of the tumble rate:** 0.4 m/s of delta-v at 0.2°/s, up to 2 m/s at 1°/s. Following a rotating docking axis means thrusting continually just to stay on it.

## Limitations

- **No contact dynamics.** Capture is an envelope check and a perfectly inelastic latch; there are no contact forces, mechanism dynamics or rebound.
- **The target's shape is known,** and the docking face is assumed to be a fixed, clean interface.
- **A fixed suite.** M13's evidence is 28 episodes. The batched engine can't carry a merged stack, so there is no Monte Carlo campaign.
- **The committed coast assumes the model.** Its last 31 s rely on exact orbital and torque-free prediction. On orbit, unmodeled forces and torques would set how close the chaser can commit.

## Reproduce

```bash
uv run --no-sync rpod m13-calibrate
uv run --no-sync rpod m13-suite --output runs/m13 --label m13
```

Both run on the CPU; the suite takes about half an hour.
