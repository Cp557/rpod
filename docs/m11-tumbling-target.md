# M11: an unmarked, tumbling target

Through M10 the target cooperated: it held its attitude, carried four ArUco markers and four lidar retroreflectors, and turned its solar wings to the Sun. A dead satellite does none of that. M11 removes every cooperative aid and lets QuikSCAT tumble freely, so the chaser must recognize the target by its shape, estimate how it is turning, and follow its rotating docking axis in to a 5 m hold in the port's frame.

[REFERENCE.md](../REFERENCE.md#unmarked-tumbling-target-m11) is the full specification.

## Method

**Scenario.** The target turns torque-free at 0.2–1.0°/s about a random axis. The chaser starts synchronized 30 m out on the rotating docking axis, co-rotating with it: the analogue of the M2 handoff, after an inspection and synchronization phase that is not simulated. The 40 episodes keep the M5 lidar suite's conditions and faults, with a seeded tumble each.

**Geometry.** A Blender worker removes the 20 cooperative aids (markers, pads, reflectors and their posts), stows the wings and exports the remaining 68,573 triangles in body axes. In Rust, a parry3d `TriMesh` answers ray casts for the lidar and closest-point queries for ICP and for a new hard boundary: no point of the chaser's hull may come within 0.5 m of the mesh.

**Lidar by shape.** The flash lidar's 256×256 rays now hit the mesh. `IcpTracker` estimates the target's pose from each scan by point-to-plane ICP and never sees truth or the filter. It acquires from 72 attitude hypotheses and rejects ambiguous fits, then tracks from its own last solution, rejecting poor fits. Its covariance comes from 900 calibration scans against truth, so the filter weights each pose by measured error rather than by the ICP residual, which M6 found optimistic. *Erratum:* a mesh-query bug, found and fixed in M12, gave most correspondences the wrong surface normal, so these M11 results come from a degraded fit; [M12](m12-lidar-faults.md) requalifies the lidar-only suite on the corrected core.

**Camera by keypoints.** A ResNet-18 heatmap network finds 20 keypoints on the satellite, 8 of them on the docking face so that the hold point stays covered, and EPnP with RANSAC turns them into a pose. A pose is kept only if the solver's predicted uncertainty is under 0.25 m and 1°; at long range, poses solved from a handful of keypoints were otherwise sometimes off by meters. The network trains on 9,000 renders at random attitudes, ranges, lighting and Earth backgrounds, is calibrated per range on 1,000 validation renders, and is judged on 1,500 held-out renders at Sun phase angles (70–100°) and degradation it never saw. Training runs in its own Python 3.13 project with PyTorch; the simulator runs the exported ONNX model through OpenCV.

**Rotating-frame guidance and navigation.** The reference supervisor and gains carry over, with metrics in the port frame and rotating-frame feedforward. The navigation filter needed one change. In the rotating frame, an error in the estimated rate is an apparent port velocity: 3e-4 rad/s at 30 m is 9 mm/s, near the 10 mm/s completion limit. The M4 filter predicts a constant rate, but a free tumble precesses. Predicting the torque-free motion exactly let the rotation process noise drop 400-fold, which cut the median rate error from 3e-4 to 4e-5 rad/s on episodes outside the suite.

**Two configurations.** Lidar only, with no camera and no rendering, and lidar fused with camera keypoints in realistic renders.

## Results

Both configurations pass all three gates of the 40-episode suite: every operational episode succeeds within 1.5× its paired oracle, every fault gets its expected response, and every record is valid.

| | Lidar only | Lidar and camera |
| --- | --- | --- |
| Operational | 28/28, median 209.3 s | 28/28, median 207.7 s |
| Completion against the paired oracle | median 1.002×, max 1.140× | median 0.996×, max 1.144× |
| Aborts | 2/2 retreated | 2/2 retreated |
| Dropped frame, extended loss | 4/4 success, no brake | 4/4 success, no brake |
| Stress | 6/6 success | 6/6 success |
| Target position error after the first 2 s (median, P95, max) | 1.3, 2.2, 4.1 cm | 1.2, 2.0, 3.8 cm |
| Closest hull approach to the mesh | 4.76 m | 4.76 m |
| Wall time | 15 min, CPU only | 3.3 h, rendering on the GPU |

- **Lidar by shape works on its own.** ICP accepted 8,525 of 8,526 scans in the lidar-only run, with median errors of 1.0–1.5 cm and about 0.2° per range bin.
- **The camera mostly speeds up acquisition.** Keypoint images count toward the three valid observations that start tracking, so the fused run acquires at 1.2 s instead of 2.7 s and finishes 1.6 s sooner. With the lidar healthy, accuracy barely changes. The half second after acquisition is worse (up to 18 cm), while the estimate rests on two camera poses from 31 m.
- **Near the hold the lidar navigates almost alone.** Within 7 m of the target's center the satellite overflows the camera frame, and only 25 of 719 images there gave a pose.
- **The faults test the lidar as the camera's backup, not the reverse.** They drop camera frames; the lidar carried navigation through every one without braking, as in M6.
- **One brake.** In episode 21 the filter refused one good lidar fit near the hold; the approach braked, resumed 2 s later, and finished at 1.14× the oracle.

**No Monte Carlo claim.** The M9 navigation surrogate, fitted to the fused suite, reproduces its estimate errors (within 4–9%) but not its timing. It fails validation: completion time falls inside the surrogate's 95% band for 6 of 28 episodes and minimum gap for 23, where 25 are needed. The surrogate has one delivery stream at the camera's cadence, so it acquires when the truth-navigation oracle does. The fused agent updates twice a second on camera and lidar and acquires a second earlier, and its one brake came at the hold rather than mid-approach. A two-stream surrogate would be fitted and judged on the same 28 episodes, so M11's evidence stays the two 40-episode suites.

## Limitations

- **The tumble is exactly torque-free with a known inertia**, and the filter predicts it with that model. On orbit, gravity-gradient torque (about 4e-7 rad/s² for this inertia) and inertia uncertainty would set the process noise.
- **The chaser starts synchronized.** Inspection, shape estimation and the synchronization maneuver are not simulated.
- **The target's shape is known.** The lidar matches the known mesh and the camera the known keypoints; an unknown or damaged target would need a model built on the spot.
- **Renders only.** The keypoint network is trained and tested on renders of the same model as the scene; transfer to real imagery is untested.
- **No population estimate.** The surrogate did not validate, so 28/28 is a fixed-suite result only.
- **No contact.** Episodes end at the 5 m hold.

## Reproduce

```bash
uv run --no-sync rpod m11-assets
uv run --no-sync rpod keypoint-dataset --output datasets/m11/keypoints
uv run --project tools/keypoints python tools/keypoints/train.py datasets/m11/keypoints runs/m11-keypoints
uv run --no-sync rpod keypoint-calibrate --model runs/m11-keypoints/keypoints.onnx
uv run --no-sync rpod m11-suite --output runs/m11-lidar --label m11-lidar
uv run --no-sync rpod m11-suite --output runs/m11-fused --keypoint-model runs/m11-keypoints/keypoints.onnx --calibration models/m11-camera-calibration.json --label m11-fused
uv run --no-sync rpod fit-surrogate --suite runs/m11-fused --scenario tumbling --output benchmarks/m11-surrogate.json
```

`m11-assets` needs Blender and the NASA model; the lidar-only suite needs no renderer. The surrogate fit exits 1 because its validation fails.
