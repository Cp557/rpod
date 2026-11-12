"""M15 stage 0: the measurements that settle the far-range design (docs/m15-rendezvous.md).

- `engine_study`: the physics' cost and accuracy over a 3 h far-range coast.
- `pointing_study`: a 1 Hz attitude loop holding the camera on the target over that coast.
- `eclipse_study`: Earth's shadow on the M15 orbit against the cylindrical-shadow formula.
- `render_study`: the unmarked target rendered from 1 to 5 km by the wide camera and a narrow
  one, and Blender's soft shadow at Earth's limb against `sunlit_fraction`.
"""

from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
from scipy.integrate import solve_ivp

from .approach import make_approach_scenario, make_engine
from .camera import CameraModel, camera_pose, earth_pose, relative_target_pose
from .frames import matrix_quaternion, quaternion_matrices, quaternion_product
from .m5_suite import QUALIFIED_DEGRADATION
from .observations import OwnNavigation
from .recording import write_json
from .rendezvous import SUN_ANGULAR_RADIUS_RAD, sunlight_direction, sunlit_fraction
from .scenarios import EARTH_RADIUS_M, MU_M3_S2, ORBIT_RADIUS_M
from .vision_records import degrade_image, json_value, new_output, read_rgb, sha256_file

COAST_S = 10_800  # 3 h, about two orbits
# The chaser starts 3 km behind the target, 300 m below and 50 m out of plane, drifting closer.
FAR_OFFSET_LVLH_M = (-300.0, -3000.0, 50.0)
RATE_RADPS = np.sqrt(MU_M3_S2 / ORBIT_RADIUS_M**3)
BETAS_DEG = (0.0, 30.0, 60.0, 75.0)
WIDE_HFOV_DEG, NARROW_HFOV_DEG = 35.0, 5.0
RANGES_M = (1000.0, 2000.0, 3000.0, 5000.0)
PHASE_ANGLES_DEG = (30.0, 90.0, 150.0)
# Against Earth: the chaser above the target's orbit, looking 30 deg down past the target.
EARTH_CASES = ((1000.0, 30.0, 60.0), (3000.0, 30.0, 60.0))
# The Sun's centre this many Sun radii outside Earth's limb: lit, penumbra, umbra's edge.
PENUMBRA_STEPS = (2.0, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0)
PENUMBRA_SAMPLES = 256  # the suites' 16 samples per pixel leave the soft shadow noisy
NOISE_DN, BLUR_PX = QUALIFIED_DEGRADATION
TARGET_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])  # body axes are LVLH at t = 0


def far_state(offset_lvlh_m=FAR_OFFSET_LVLH_M):
    """The M2 bodies with the chaser at `offset_lvlh_m` on a circular CW drift.

    The target starts on +X moving +Y, so its LVLH and the inertial axes agree at t = 0.
    """
    manifest = make_approach_scenario("approach")
    target_r = np.array(manifest["initial_positions_m"][0])
    target_v = np.array(manifest["initial_velocities_mps"][0])
    offset = np.asarray(offset_lvlh_m, dtype=np.float64)
    drift = np.array([0.0, -1.5 * RATE_RADPS * offset[0], 0.0])
    chaser_v = target_v + drift + np.cross([0.0, 0.0, RATE_RADPS], offset)
    manifest["initial_positions_m"] = [target_r.tolist(), (target_r + offset).tolist()]
    manifest["initial_velocities_mps"] = [target_v.tolist(), chaser_v.tolist()]
    return manifest


def reference_offset(manifest, duration_s):
    """Chaser minus target position after `duration_s` of two-body coasting, integrated in
    relative form with DOP853 at tight tolerances."""
    (target_r, chaser_r), (target_v, chaser_v) = (
        np.asarray(manifest[key], dtype=np.float64)
        for key in ("initial_positions_m", "initial_velocities_mps"))
    mu = manifest["mu_m3_s2"]

    def derivative(_, state):
        target, offset = state[:3], state[6:9]
        gravity = -mu * target / np.linalg.norm(target) ** 3
        chaser = target + offset
        return np.r_[state[3:6], gravity, state[9:],
                     -mu * chaser / np.linalg.norm(chaser) ** 3 - gravity]

    solution = solve_ivp(derivative, (0.0, duration_s),
                         np.r_[target_r, target_v, chaser_r - target_r, chaser_v - target_v],
                         method="DOP853", rtol=1e-13, atol=1e-12)
    if not solution.success:
        raise RuntimeError(f"reference integration failed: {solution.message}")
    return solution.y[6:9, -1]


def engine_study(manifest, duration_s=COAST_S):
    """Wall time per physics tick at several ticks per call, and the relative-position error
    after `duration_s` at several tick sizes against the DOP853 reference."""
    zeros = np.zeros((2, 3))
    reference = reference_offset(manifest, duration_s)
    accuracy = []
    for tick_s in (0.1, 1.0, 10.0):
        engine = make_engine({**manifest, "dt_ns": round(tick_s * 1e9)})
        after = engine.step(zeros, zeros, round(duration_s / tick_s))
        offset = np.subtract(after.positions_m[1], after.positions_m[0])
        accuracy.append({"tick_s": tick_s,
                         "relative_position_error_m": float(np.linalg.norm(offset - reference))})
    ticks = round(duration_s / (manifest["dt_ns"] / 1e9))
    cost = []
    for per_call in (1, 10, 100, ticks):
        engine = make_engine(manifest)
        started = perf_counter()
        for _ in range(ticks // per_call):
            engine.step(zeros, zeros, per_call)
        seconds = perf_counter() - started
        cost.append({"ticks_per_call": per_call, "wall_seconds": seconds,
                     "microseconds_per_tick": 1e6 * seconds / ticks})
    start = np.subtract(manifest["initial_positions_m"][1], manifest["initial_positions_m"][0])
    return {"coast_s": duration_s, "physics_tick_s": manifest["dt_ns"] / 1e9, "ticks": ticks,
            "initial_offset_lvlh_m": start, "relative_motion_m": np.linalg.norm(reference - start),
            "reference": "DOP853 in relative form, rtol 1e-13, atol 1e-12",
            "accuracy": accuracy, "cost": cost}


def _pointing(snapshot, inertia, gains):
    """Torque turning the camera boresight, chaser body +Y, onto the target with body +Z
    toward the orbit normal; and the boresight's angle from the target in radians."""
    positions = np.asarray(snapshot.positions_m)
    sight = positions[0] - positions[1]
    sight /= np.linalg.norm(sight)
    momentum = np.cross(positions[0], snapshot.velocities_mps[0])
    rate = np.linalg.norm(momentum) / (positions[0] @ positions[0])
    normal = momentum / np.linalg.norm(momentum)
    side = normal - (normal @ sight) * sight
    side /= np.linalg.norm(side)
    quaternion = np.asarray(snapshot.quaternions_xyzw[1])
    omega = np.asarray(snapshot.angular_velocities_body_radps[1])
    rotation = quaternion_matrices(quaternion)
    desired = matrix_quaternion(np.column_stack((np.cross(sight, side), sight, side)))
    error = quaternion_product(quaternion * [-1, -1, -1, 1], desired)
    error = -error if error[3] < 0 else error
    length = np.linalg.norm(error[:3])
    turn = (2 * np.arctan2(length, error[3]) * error[:3] / length if length > 1e-12
            else 2 * error[:3])
    alpha = (gains["attitude_kp_s2"] * turn
             + gains["rate_kd_s"] * (rotation.T @ normal * rate - omega))
    torque = inertia * alpha + np.cross(omega, inertia * omega)
    return torque, float(np.arccos(np.clip(rotation[:, 1] @ sight, -1.0, 1.0)))


def pointing_study(manifest, duration_s=COAST_S, control_ticks=10):
    """Hold the camera on the target with one attitude update per `control_ticks` over a coast,
    from the M2 gains: Python time per update, the boresight error after the first minute, and
    when and how far from the target its worst came, beside the closest approach."""
    engine = make_engine(manifest)
    inertia = np.asarray(manifest["inertias_kg_m2"][1])
    period_s = control_ticks * manifest["dt_ns"] / 1e9
    updates = round(duration_s / period_s)
    zero = np.zeros(3)
    snapshot, errors, ranges = engine.snapshot(), [], []
    started = perf_counter()
    for _ in range(updates):
        torque, error = _pointing(snapshot, inertia, manifest["controller"])
        errors.append(error)
        ranges.append(np.linalg.norm(np.subtract(*snapshot.positions_m)))
        snapshot = engine.step([zero, zero], [zero, torque], control_ticks)
    seconds = perf_counter() - started
    errors, ranges = np.degrees(errors), np.asarray(ranges)
    first = round(60 / period_s)
    worst, closest = first + int(np.argmax(errors[first:])), int(np.argmin(ranges))
    return {"control_period_s": period_s, "updates": updates, "wall_seconds": seconds,
            "microseconds_per_update": 1e6 * seconds / updates,
            "initial_boresight_error_deg": errors[0],
            "boresight_error_after_60_s_deg": {"median": np.median(errors[first:]),
                                               "max": errors[worst]},
            "worst": {"time_s": worst * period_s, "range_m": ranges[worst]},
            "closest_approach": {"time_s": closest * period_s, "range_m": ranges[closest]}}


def eclipse_study(samples=200_000):
    """Shadow, umbra and penumbra time per M15 orbit at several Sun elevations above the orbit
    plane (beta), against the cylindrical-shadow duration. "Shadow" is less than half the Sun."""
    period_s = 2 * np.pi / RATE_RADPS
    angles = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    positions = ORBIT_RADIUS_M * np.stack((np.cos(angles), np.sin(angles), np.zeros(samples)),
                                          axis=-1)
    step_s = period_s / samples
    rows = []
    for beta in BETAS_DEG:
        fraction = sunlit_fraction(positions, sunlight_direction(beta, 0.0))
        cosine = np.sqrt(1 - (EARTH_RADIUS_M / ORBIT_RADIUS_M) ** 2) / np.cos(np.radians(beta))
        rows.append({
            "beta_deg": beta,
            "shadow_s": (fraction < 0.5).sum() * step_s,
            "cylindrical_shadow_s": 2 * np.arccos(cosine) / RATE_RADPS if cosine <= 1 else 0.0,
            "umbra_s": (fraction == 0).sum() * step_s,
            "penumbra_s": ((fraction > 0) & (fraction < 1)).sum() * step_s,
        })
    return {"orbit_period_s": period_s, "sample_s": step_s, "betas": rows}


def _pose(range_m, depression_deg):
    """Target at its t = 0 position; the chaser `range_m` behind it and `depression_deg` above
    its along-track line, the camera boresight (body +Y) on the target, body +Z the normal."""
    target = np.array([ORBIT_RADIUS_M, 0.0, 0.0])
    depression = np.radians(depression_deg)
    chaser = target + range_m * np.array([np.sin(depression), -np.cos(depression), 0.0])
    sight, side = (target - chaser) / range_m, np.array([0.0, 0.0, 1.0])
    attitude = matrix_quaternion(np.column_stack((np.cross(sight, side), sight, side)))
    return target, OwnNavigation(0, chaser, np.zeros(3), attitude, np.zeros(3))


def _sunlight(target, own, phase_deg):
    """Incident Sun ray at `phase_deg` between the target's directions to the camera and to
    the Sun, turned from the camera direction toward the local vertical."""
    toward_camera = own.position_m - target
    toward_camera /= np.linalg.norm(toward_camera)
    up = target / np.linalg.norm(target)
    up = up - (up @ toward_camera) * toward_camera
    up /= np.linalg.norm(up)
    phase = np.radians(phase_deg)
    return -(np.cos(phase) * toward_camera + np.sin(phase) * up)


def _gray(path):
    return read_rgb(path).astype(np.float64).mean(axis=-1)


def _linear(path):
    """Mean scene-linear value per pixel of an 8-bit sRGB render."""
    value = read_rgb(path).astype(np.float64) / 255
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4).mean(-1)


def _signal(image, empty, expected_px, focal_px, range_m):
    """The target's mark on the image: the render minus the target-free render, in 8-bit grey.

    Its footprint is where it changes the image by more than 3 sensor-noise sigmas; the SNR is
    its peak after the suites' blur over their noise; the bearing offset is its centroid's
    distance from the projected centre of mass.
    """
    signal = image - empty
    mask = np.abs(signal) > 3 * NOISE_DN
    if not mask.any():
        return {"footprint_px": 0, "snr": 0.0}
    weights = np.abs(signal) * mask
    rows, columns = np.indices(signal.shape)
    centroid = np.array([(weights * columns).sum(), (weights * rows).sum()]) / weights.sum()
    offset_px = float(np.linalg.norm(centroid - expected_px))
    peak = float(np.abs(cv2.GaussianBlur(signal, (0, 0), BLUR_PX)).max())
    return {"footprint_px": int(mask.sum()), "peak_contrast_dn": peak, "snr": peak / NOISE_DN,
            "background_dn": float(empty[mask].mean()), "centroid_offset_px": offset_px,
            "bearing_offset_mrad": 1e3 * offset_px / focal_px,
            "bearing_offset_m": offset_px / focal_px * range_m}


def _detect(pixels, seed, expected_px, footprint_px):
    """A global-threshold blob detector on the suites' degraded image: pixels 6 robust sigmas
    above the median grey, keeping the blob with the most excess; and whether it is the target."""
    gray = degrade_image(pixels, seed, NOISE_DN, BLUR_PX).astype(np.float64).mean(axis=-1)
    median = np.median(gray)
    sigma = max(1.4826 * np.median(np.abs(gray - median)), NOISE_DN / np.sqrt(3))
    mask = (gray > median + 6 * sigma).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return {"detected": False, "blobs": 0}
    excess = np.bincount(labels.ravel(), weights=(gray - median).ravel(), minlength=count)
    best = 1 + int(np.argmax(excess[1:]))
    offset = float(np.linalg.norm(centroids[best] - expected_px))
    return {"detected": offset <= max(5.0, 2 * np.sqrt(footprint_px / np.pi)),
            "blobs": count - 1, "blob_px": int(stats[best, cv2.CC_STAT_AREA]),
            "blob_offset_px": offset}


def _tile(path, expected_px, hfov_deg, label):
    """A 128 px crop around the target: native for the narrow camera, 4x for the wide one."""
    image = read_rgb(path)
    half = 64 if hfov_deg == NARROW_HFOV_DEG else 16
    column, row = np.rint(expected_px).astype(int)
    padded = np.pad(image, ((half, half), (half, half), (0, 0)))
    crop = padded[row:row + 2 * half, column:column + 2 * half]
    tile = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, label, (3, 123), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 200, 0), 1)
    return tile


def _penumbra(output):
    """Blender's soft shadow against `sunlit_fraction`, with the narrow camera at
    `PENUMBRA_SAMPLES`: the target 1 km ahead while the Sun sets behind the chaser, its centre
    `PENUMBRA_STEPS` Sun radii outside the limb. The rendered fraction is the target's summed
    scene-linear brightness over the fully lit render's, on pixels unsaturated when lit."""
    from .rendering import realistic_scene
    from .target_mesh import UnmarkedRenderer

    camera = CameraModel(hfov_deg=NARROW_HFOV_DEG)
    target, own = _pose(1000.0, 0.0)
    limb = np.arcsin(EARTH_RADIUS_M / np.linalg.norm(target))
    down, back = -target / np.linalg.norm(target), np.array([0.0, -1.0, 0.0])
    rotation_ct, translation_ct = relative_target_pose(own, target, TARGET_QUATERNION, camera)
    rows, lit = [], None
    with UnmarkedRenderer(Path(output) / "renderer-penumbra", camera=camera,
                          scene=realistic_scene(samples=PENUMBRA_SAMPLES)) as renderer:
        for steps in PENUMBRA_STEPS:
            angle = limb + steps * SUN_ANGULAR_RADIUS_RAD
            sunlight = -(np.cos(angle) * down + np.sin(angle) * back)
            path = Path(output) / "renders" / f"penumbra-{steps:+g}.png"
            renderer.render(rotation_ct, translation_ct, path,
                            light_direction_c=camera_pose(own, camera)[1].T @ sunlight,
                            earth=earth_pose(own, camera))
            linear = _linear(path)
            if lit is None:
                lit = linear
                usable = (lit > 0.002) & (read_rgb(path).max(axis=-1) < 255)
            rows.append({"sun_radii_outside_limb": steps,
                         "sunlit_fraction": float(sunlit_fraction(target, sunlight)),
                         "rendered_fraction": float(linear[usable].sum() / lit[usable].sum())})
    return {"samples": PENUMBRA_SAMPLES, "pixels": int(usable.sum()), "steps": rows}


def render_study(output):
    """Render the unmarked target with the wide and narrow cameras; measure its signal,
    bearing offset and a simple detector, and the penumbra with the narrow camera."""
    from .target_mesh import UnmarkedRenderer

    directory = Path(output) / "renders"
    cases = [(range_m, 0.0, phase) for range_m in RANGES_M for phase in PHASE_ANGLES_DEG]
    cases += EARTH_CASES
    rows, timing, tiles = [], {}, {}
    for hfov in (WIDE_HFOV_DEG, NARROW_HFOV_DEG):
        camera = CameraModel(hfov_deg=hfov)
        seconds = []
        with UnmarkedRenderer(Path(output) / f"renderer-{hfov:g}deg", camera=camera) as renderer:
            for index, (range_m, depression, phase) in enumerate(cases):
                target, own = _pose(range_m, depression)
                sunlight = _sunlight(target, own, phase)
                rotation_ct, translation_ct = relative_target_pose(
                    own, target, TARGET_QUATERNION, camera)
                request = {"light_direction_c": camera_pose(own, camera)[1].T @ sunlight,
                           "earth": earth_pose(own, camera)}
                name = f"{hfov:g}deg-{range_m / 1000:g}km-{depression:g}down-{phase:g}phase"
                paths = {variant: directory / f"{name}-{variant}.png"
                         for variant in ("target", "empty")}
                for variant, path in paths.items():
                    result = renderer.render(rotation_ct, translation_ct, path,
                                             variant=variant, **request)
                    if variant == "target":
                        seconds.append(result["render_seconds"])
                projected = camera.K @ translation_ct
                expected = projected[:2] / projected[2]
                signal = _signal(_gray(paths["target"]), _gray(paths["empty"]), expected,
                                 camera.K[0, 0], range_m)
                rows.append({"hfov_deg": hfov, "range_m": range_m,
                             "depression_deg": depression, "phase_deg": phase,
                             "sunlit_fraction": float(sunlit_fraction(target, sunlight)),
                             **signal, "detector": _detect(
                                 read_rgb(paths["target"]), [15, index, round(hfov)], expected,
                                 signal["footprint_px"])})
                tiles[hfov, index] = _tile(paths["target"], expected, hfov,
                                           f"{range_m / 1000:g}km {phase:g}"
                                           + (" Earth" if depression else ""))
            timing[f"{hfov:g}"] = {"startup_s": renderer.startup_seconds,
                                   "median_render_s": float(np.median(seconds))}
    sheet = np.vstack([np.hstack([tiles[hfov, index] for index in range(len(cases))])
                       for hfov in (WIDE_HFOV_DEG, NARROW_HFOV_DEG)])
    cv2.imwrite(str(Path(output) / "detectability.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    penumbra = _penumbra(output)
    shutil.rmtree(directory)  # the sheet and the report keep what the frames showed
    return {"cameras_hfov_deg": [WIDE_HFOV_DEG, NARROW_HFOV_DEG],
            "degradation": {"noise_dn": NOISE_DN, "blur_px": BLUR_PX},
            "cases": rows, "penumbra": penumbra, "timing": timing}


def run_stage0(output, renders=True):
    """Run every study and write `report.json`; `renders=False` skips the GPU work."""
    output = new_output(output)
    manifest = far_state()
    source = Path(__file__).parent
    report = {
        "implementation_sha256": {name: sha256_file(source / name)
                                  for name in ("m15_stage0.py", "rendezvous.py")},
        "engine": engine_study(manifest), "pointing": pointing_study(manifest),
        "eclipse": eclipse_study(),
    }
    if renders:
        report["renders"] = render_study(output)
    write_json(output / "report.json", json_value(report))
    return report
