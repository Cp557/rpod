# M12: lidar faults and safety after a navigation loss

In M11 the lidar does nearly all the navigating near the tumbling target, yet the suite's faults only drop camera frames. M12 fails the lidar itself and asks two questions. Does the chaser recover? And when it cannot, does it stay clear of a satellite that keeps tumbling next to it?

[REFERENCE.md](../REFERENCE.md#lidar-faults-and-safety-after-a-navigation-loss-m12) is the full specification.

## A bug found on the way

Close-range tests of the lidar's shape matching showed a steady 23 cm error along the docking axis inside 3 m. The cause was in the Rust mesh query. parry 0.31 reports which *side* of a triangle a point projects onto, and puts the triangle's index elsewhere. Our code read that side flag as the triangle index, so almost every correspondence got the normal of triangle 0 or 1. The "point-to-plane" fit M11 described was not what ran.

M11's lidar still reached 1–1.5 cm at 5–35 m, carried by edge and vertex correspondences, but close to a flat face nothing constrained the axial direction. The fix is one line and a test that probes every face of a cube; the old test probed only the face that happened to work.

On the corrected core, the M11 lidar-only suite passes again with the same 40 outcomes:

| | M11 | Corrected |
| --- | --- | --- |
| Target position error (median, P95, max) | 1.3, 2.3, 4.1 cm | 0.9, 1.3, 2.5 cm |
| Target attitude error (median, P95) | 0.09, 0.27° | 0.04, 0.16° |
| Scans accepted | 8,525 of 8,526 | 8,476 of 8,476 |

The suites hash every Python file but not the native core, which is how the fix changed behavior without changing a hash. M12's suite also records the core's hash.

## Method

**Faults.** The environment fails the lidar once per episode. The fault starts when the true port gap first falls to 25, 12 or 5.1 m, and is one of four kinds:
- one dropped scan;
- one blank scan, which also makes M11's tracker drop its lock;
- three scans biased 0.25 m long, which only the filter's innovation gate can catch;
- a 4 s outage.

Each of the 12 episodes reuses one of M11's operational episodes, so it pairs with an unfaulted record.

**A burn after a navigation loss.** M11's chaser declares navigation lost when no measurement has been accepted for 3 s, and the episode ends. That leaves the question of what happens next unanswered. Now it first burns away from its last estimate of the target. The burn leaves it moving 0.05 m/s away from the target's center plus 0.05 m/s along-track, because a purely radial separation would bring it back after one orbit.

**Coast checks.** Both spacecraft coast with no thrust for 6,000 s, longer than an orbit, in the batched engine. The check measures how close the chaser's hull comes to the target's mesh, exactly whenever it is within a metre. A coast is passively safe if the hull stays at least 0.5 m away.

**Two configurations.**
- `m11`: M11's settings.
- `robust`: the tracker keeps its lock through an empty scan, and navigation is declared lost after 10 s instead of 3 s. The target's torque-free motion is predicted so well that the longer coast is cheap.

Both configurations were chosen on episodes outside the suite.

## Results

Both configurations pass the safety gates:
- every episode ends in success or a declared loss, with no false completion or boundary violation;
- every loss is followed by a coast that stays clear.

| Fault | `m11` | `robust` |
| --- | --- | --- |
| One dropped scan | recovers | recovers |
| One blank scan | recovers | recovers |
| Three biased scans | lost | recovers |
| 4 s outage | lost | recovers |

- **Two bad scans in a row are enough to lose navigation with M11's settings, at any range.** The gate refused every biased scan, so the bias never reached the estimate; refusing it is what cost the lock.
- **The robust configuration recovers from all twelve.** While coasting through up to 4 s without a measurement, its estimate drifted by at most 0.7 cm and 0.008°.
- **Recovery is not free.** Every fault brakes. At 12 m and at the hold, the restarted approach assumes the chaser is at rest when it is still closing, so the chaser passes the hold by up to 0.53 m and finishes 60–77 s late. This is the reference supervisor's behavior, also seen with M8's dropped camera frames.
- **The burn matters.** All six of M11's losses ended clear of the target, the closest coast still 4.75 m away. Without the burn, the loss during the outage at the hold would have hit the satellite five minutes later.

**Passive safety.** Coasting from every 10 s of the 28 M11 approaches, as if all control stopped then:

| Port gap | Passively safe |
| --- | --- |
| below 6 m | 74 of 86 |
| 6–8 m | 17 of 35 |
| 8–12 m | 67 of 83 |
| 12–18 m | 104 of 112 |
| 18–25 m | 123 of 124 |
| 25 m and beyond | 157 of 157 |

The approach closes straight along the rotating docking axis, so inside about 12 m a chaser that stops thrusting often drifts into the tumbling satellite within a minute or two. Safety there rests on active control. The burn provides it when navigation fails; nothing does if all thrust fails.

## Limitations

- **One fault per episode, twelve episodes.** This is a fixed suite, not a population estimate. The robust configuration's 10 s limit was never actually reached; the longest coast was 5.1 s.
- **The burn is open-loop and assumes the thrusters still work.** It is planned from the last estimate and never re-checked against the target.
- **Coasting is passive physics only:** central gravity and a torque-free tumble. A real coast would also feel drag, solar pressure and the target's gravity-gradient torque.
- **The overshoot after a brake is unaddressed.** Fixing it means changing the reference supervisor, which every suite since M5 shares.

## Reproduce

```bash
uv run --no-sync rpod m11-suite --output runs/m11-lidar-normals --label m11-lidar-normals
uv run --no-sync rpod m12-suite --output runs/m12-lidar --label m12-lidar
uv run --no-sync rpod m12-suite --output runs/m12-lidar-robust --configuration robust --label m12-lidar-robust
uv run --no-sync rpod passive-safety runs/m11-lidar-normals --output benchmarks/m12-passive-safety.json
```

The lidar-only runs need no renderer and take minutes on a CPU.
