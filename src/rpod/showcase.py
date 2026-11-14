"""Presentation-only movie of a saved visual run, filmed by an outside camera.

Reads a validated schema-3 record and renders it in the realistic scene, with a chaser,
city lights, stars and thruster plumes added, under the run's own Sun. One continuous shot
from above and behind the chaser, level with the orbit, keeps both spacecraft whole in frame
from the first tick to the last and closes in as they do. An M11 run is filmed around the
unmarked target. A docking run (M13, M14) is filmed faster through the approach than around
contact, easing between speeds. A run with
`thrusters.npz` (M14) fires each nozzle's plume from its recorded on-time. Nothing here
feeds physics, autonomy, or evaluation.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import CameraModel, earth_rotation_ne
from .frames import port_relative_state, quaternion_matrices, relative_state
from .rendering import ASSET_CREDITS, ASSETS, BlenderRenderer, realistic_scene
from .thrusters import SPECIFIC_IMPULSE_S, STANDARD_GRAVITY_MPS2, layout
from .tumble import tumble_metrics
from .vision_records import json_value, new_output, read_jsonl, read_rgb, sha256_file
from .visual_recording import load_visual_run
from .visualization import PHASE_NAMES


VIEW = CameraModel(width=1920, height=1080, hfov_deg=50.0, camera_id="showcase")
# The camera's direction from the spacecraft midpoint in target LVLH (radial, along-track,
# normal): above and behind the chaser, 29 degrees off the along-track axis, so for an
# approach from behind (M11-M14) it looks past the chaser at the target's docking face and
# Earth's limb. The normal component's sign follows the Sun, so the camera sees the lit sides.
OFFSET_LVLH = np.array([0.25, -1.0, -0.5]) / np.linalg.norm([0.25, -1.0, -0.5])
# Its distance, hypot(DISTANCE_PER_SEPARATION x separation, MIN_DISTANCE_M), frames both
# spacecraft whole while they are apart and, as they dock, keeps QuikSCAT's arrays, whose tips
# sweep 4.1 m from its tumble axis, inside the frame and clear of the telemetry panel and inset.
DISTANCE_PER_SEPARATION = 1.7
MIN_DISTANCE_M = 16.0
TICKS_PER_FRAME = 3
# Docking runs are filmed at these physics ticks per frame: 18x real time through the
# approach, 9x from the terminal segment, 3x from CONTACT_LEAD_S before the latch to the end,
# easing down one tick per frame at a time over the RAMP_S of the run before each change.
APPROACH_TICKS, TERMINAL_TICKS, CONTACT_TICKS = 6, 3, 1
CONTACT_LEAD_S = 12.0
RAMP_S = 20.0
FPS = 30
# Denoised, 8 samples look the same as 16 at half the GPU time.
SAMPLES = 8
EXPOSURE = 0.5
# A camera exposed for sunlit spacecraft would record neither stars nor city lights; these
# keep them faint.
STAR_STRENGTH = 0.15
CITY_LIGHTS = 0.05
PLUME_GLOW = 0.6
SCALE = VIEW.height / 720
INSET_PX = round(270 * SCALE)
# Lens: bloom around highlights, four-point streaks from the brightest, a vignette, grain.
BLOOM_THRESHOLD, BLOOM_SIGMA_PX, BLOOM_GAIN = 190.0, 6.0 * SCALE, 0.8
STREAK_THRESHOLD, STREAK_LENGTH_PX, STREAK_GAIN = 235.0, 40 * SCALE, 2.0
VIGNETTE, GRAIN_DN = 0.18, 1.5
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _look(position, aim, up):
    """Optical-camera-to-inertial rotation looking from `position` at `aim` with `up` up."""
    forward = (aim - position) / np.linalg.norm(aim - position)
    down = (up @ forward) * forward - up
    down /= np.linalg.norm(down)
    return np.column_stack((np.cross(down, forward), down, forward))


def _sun_side(frames_nl, sunlight_n):
    """+1 or -1: the normal side the Sun shines from (`sunlight_n` is the incident ray), on
    average over one frame-to-inertial rotation or a stack of them."""
    return 1.0 if np.sum(np.asarray(frames_nl)[..., :, 2] @ sunlight_n) >= 0 else -1.0


def chase_camera(target_n, chaser_n, basis_nl, side):
    """The movie's one camera: origin and optical-camera-to-inertial rotation framing both
    spacecraft, looking at the midpoint of their centers of mass with LVLH radial up. `side`
    is `_sun_side`'s sign for the run, which puts the camera on the normal side the Sun
    lights."""
    middle = (np.asarray(target_n) + np.asarray(chaser_n)) / 2
    distance = np.hypot(DISTANCE_PER_SEPARATION * np.linalg.norm(np.subtract(chaser_n, target_n)),
                        MIN_DISTANCE_M)
    position = middle + distance * (basis_nl @ (OFFSET_LVLH * [1, 1, side]))
    return position, _look(position, middle, basis_nl[:, 0])


def plume_throttles(force_body_n, limit_n):
    """Throttle in [0, 1] for exhaust along +x, -x, +y, -y, +z, -z of the chaser body.

    Exhaust along d pushes the chaser along -d.
    """
    force = np.clip(np.asarray(force_body_n, dtype=np.float64) / limit_n, -1, 1)
    return [float(value) for axis in force for value in (max(-axis, 0.0), max(axis, 0.0))]


def nozzle_throttles(axis_throttles):
    """Per-nozzle throttles, in `thrusters.layout` order, from `plume_throttles`' six."""
    exhausts = -layout()[1]
    axes = np.argmax(np.abs(exhausts), axis=1)
    return [axis_throttles[2 * axis + int(exhaust[axis] < 0)]
            for axis, exhaust in zip(axes, exhausts)]


def frame_ticks(count, dt_s, terminal_s=None, latch_tick=None):
    """Indices of the recorded ticks to film: every `TICKS_PER_FRAME`, or for a docking run
    `APPROACH_TICKS` before the terminal segment, `TERMINAL_TICKS` after it and
    `CONTACT_TICKS` from `CONTACT_LEAD_S` before the latch on, each change eased in over the
    `RAMP_S` before it; always ending on the last tick."""
    if latch_tick is None:
        ticks = list(range(0, count, TICKS_PER_FRAME))
    else:
        changes = [(latch_tick * dt_s - CONTACT_LEAD_S, CONTACT_TICKS)]
        if terminal_s is not None:
            changes.insert(0, (terminal_s, TERMINAL_TICKS))
        ticks, index = [], 0
        while index < count:
            ticks.append(index)
            step = APPROACH_TICKS
            for start_s, level in changes:
                step = round(np.interp(index * dt_s, [start_s - RAMP_S, start_s], [step, level]))
            index += step
    return ticks if ticks[-1] == count - 1 else ticks + [count - 1]


class _ShowcaseRenderer(BlenderRenderer):
    """BlenderRenderer process handling, driving the showcase worker."""

    def __init__(self, output_dir, worker="showcase_worker.py"):
        scene = {**realistic_scene(samples=SAMPLES), "exposure": EXPOSURE,
                 "star_strength": STAR_STRENGTH, "city_lights": CITY_LIGHTS,
                 "plume_glow": PLUME_GLOW}
        super().__init__(output_dir, camera=VIEW, scene=scene)
        self._worker = Path(__file__).with_name(worker).resolve()

    def render_view(self, output_path, target, chaser, earth, light_direction_c, rotation_nc,
                    throttles):
        """Render one frame; each pose is a (rotation, translation) pair in camera axes."""
        request = {
            "op": "render", "rotation_ct": target[0], "translation_ct": target[1],
            "chaser_rotation_cb": chaser[0], "chaser_translation_cb": chaser[1],
            "earth_rotation_ce": earth[0], "earth_translation_ce": earth[1],
            "light_direction_c": light_direction_c, "stars_rotation_nc": rotation_nc,
            "plume_throttles": throttles, "output_path": str(Path(output_path).resolve()),
        }
        return self._request(json_value(request))


def _line_kernel(angle_deg):
    """A normalized streak through the center at `angle_deg`, fading exponentially."""
    size = int(6 * STREAK_LENGTH_PX) | 1
    kernel = np.zeros((size, size), np.float32)
    middle, direction = size // 2, np.radians(angle_deg)
    for step in range(-middle, middle + 1):
        x = int(round(middle + step * np.cos(direction)))
        y = int(round(middle + step * np.sin(direction)))
        kernel[y, x] = np.exp(-abs(step) / STREAK_LENGTH_PX)
    return kernel / kernel.sum()


_STREAKS = [_line_kernel(angle) for angle in (0, 90)]
_rows, _columns = np.ogrid[-1:1:VIEW.height * 1j, -1:1:VIEW.width * 1j]
_VIGNETTE = (1 - VIGNETTE * np.clip((_rows ** 2 + (_columns * VIEW.width / VIEW.height) ** 2)
                                    / (1 + (VIEW.width / VIEW.height) ** 2), 0, 1) ** 1.5
             ).astype(np.float32)[..., None]


def _lens(image, index):
    """Bloom around highlights, horizontal and vertical streaks from the brightest, a
    vignette, and grain seeded by the frame index."""
    image = image.astype(np.float32)
    glow = cv2.GaussianBlur(np.clip(image - BLOOM_THRESHOLD, 0, None), (0, 0), BLOOM_SIGMA_PX)
    bright = np.clip(image - STREAK_THRESHOLD, 0, None)
    streaks = sum(cv2.filter2D(bright, -1, kernel) for kernel in _STREAKS)
    image = (image + BLOOM_GAIN * glow + STREAK_GAIN * streaks) * _VIGNETTE
    image += np.random.default_rng(index).normal(0, GRAIN_DN, image.shape[:2])[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


@lru_cache
def _font(name, size):
    return ImageFont.truetype(str(FONTS / name), round(size * SCALE))


def _lidar_inset(path):
    """A lidar scan as a depth image: nearer returns brighter, no return black."""
    with np.load(path) as data:
        ranges = data["ranges_m"].astype(np.float64)
    returns = np.isfinite(ranges)
    if not returns.any():
        return None
    near, far = np.percentile(ranges[returns], [2, 98])
    depth = np.clip((far - np.where(returns, ranges, far)) / max(far - near, 1e-6), 0, 1)
    colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored[~returns] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _overlay(frame, inset, title, rows, sensor):
    """A telemetry panel at the top left (label, value rows) and the docking-sensor inset at
    the top right, captioned under it."""
    height, width = frame.shape[:2]
    margin, inner = round(16 * SCALE), round(12 * SCALE)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    heading, label, value = _font("DejaVuSans-Bold.ttf", 15), _font("DejaVuSans.ttf", 12), \
        _font("DejaVuSansMono.ttf", 14)
    line = round(22 * SCALE)
    panel = (margin, margin, margin + round(300 * SCALE),
             margin + 2 * inner + round(26 * SCALE) + line * len(rows))
    draw.rounded_rectangle(panel, radius=round(8 * SCALE), fill=(8, 10, 14, 150))
    draw.text((panel[0] + inner, panel[1] + inner), title, font=heading, fill=(255, 255, 255))
    for row, (name, text) in enumerate(rows):
        y = panel[1] + inner + round(28 * SCALE) + row * line
        draw.text((panel[0] + inner, y + round(2 * SCALE)), name.upper(), font=label,
                  fill=(150, 160, 175))
        draw.text((panel[0] + round(118 * SCALE), y), text, font=value, fill=(240, 244, 250))
    x, y = width - INSET_PX - margin, margin
    tile = (np.full((INSET_PX, INSET_PX, 3), 20, np.uint8) if inset is None else
            cv2.resize(inset, (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA))
    image.paste(Image.fromarray(tile), (x, y))
    draw.rectangle((x - 1, y - 1, x + INSET_PX, y + INSET_PX), outline=(200, 200, 200))
    draw.text((x + INSET_PX, y + INSET_PX + round(6 * SCALE)),
              sensor + ("" if inset is not None else ": no image"), font=label,
              fill=(230, 230, 230), anchor="ra")
    return np.asarray(image)


class _Video:
    """H.264 (libx264, CRF 18, yuv420p) through ffmpeg when it is on PATH, else OpenCV's
    mp4v. Frames are RGB."""

    def __init__(self, path, fps, size):
        ffmpeg = shutil.which("ffmpeg")
        self.process = self.writer = None
        if ffmpeg:
            self.codec = "h264 (libx264, crf 18, preset slow, yuv420p)"
            self.process = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        else:
            self.codec = "mp4v"
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.writer.isOpened():
                raise OSError(f"cannot open video writer: {path}")

    def write(self, rgb):
        if self.process is not None:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else:
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise OSError("ffmpeg failed to encode the movie")
        else:
            self.writer.release()


def _encode(video, path, index, inset, title, rows, sensor, stills):
    """Lens effects and overlay on the clean render at `path`, then append it to the video;
    the clean render is deleted unless `stills` keeps the finished frame beside it."""
    frame = _overlay(_lens(read_rgb(path), index), inset, title, rows, sensor)
    video.write(frame)
    if stills:
        cv2.imwrite(str(path.with_suffix(".jpg")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        path.unlink()


def _phase_label(phase, seconds, gap, terminal_s, latch_s):
    """What the chaser is doing, in words, for a docking run's phase and timeline."""
    if latch_s is not None and seconds >= latch_s:
        return "docked" if phase == 3 else "captured, detumbling"
    if phase == 1 and terminal_s is not None:
        if seconds < terminal_s:
            return "holding at 5 m" if abs(gap - 5.0) < 0.3 else "approach to 5 m hold"
        return "final coast" if gap <= 1.3 else "terminal approach"
    return PHASE_NAMES[phase].replace("_", " ")


def render_showcase(run_dir, output, ticks=None, stills=False):
    """Render `run_dir` to `output`/showcase.mp4. `ticks` defaults to `frame_ticks`; `stills`
    keeps each finished frame as a JPEG instead of deleting the renders."""
    run_dir = Path(run_dir)
    manifest, trajectory = load_visual_run(run_dir)
    summary_in = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    output = new_output(output)
    dt_s = manifest["dt_ns"] / 1e9
    positions, quaternions = trajectory["positions_m"], trajectory["quaternions_xyzw"]
    rotations = quaternion_matrices(quaternions)
    basis = relative_state(positions, trajectory["velocities_mps"])[2]
    states = (positions, trajectory["velocities_mps"], quaternions,
              trajectory["angular_velocities_body_radps"], manifest["docking_ports_body_m"])
    # M11 runs: an unmarked target tumbling freely, scored in its port frame.
    tumbling = manifest["scenario"].startswith("visual-m11-")
    gaps = tumble_metrics(*states)["gap"] if tumbling else -port_relative_state(*states)[0][:, 1]
    closing = -np.gradient(gaps, dt_s)
    target_rate_deg_s = np.degrees(np.linalg.norm(
        trajectory["angular_velocities_body_radps"][:, 0], axis=-1))
    forces = trajectory["applied_forces_body_n"][:, 1]
    force_limit = float(manifest["force_limits_n"][1])
    sunlight_n = np.asarray(manifest["sunlight_direction_inertial"], dtype=np.float64)
    latch_tick = summary_in["contact"]["tick"] if summary_in.get("latched") else None
    latch_s = latch_tick * dt_s if latch_tick is not None else None
    terminal_s = summary_in.get("terminal_start_s")
    # M14 runs fire each nozzle from its recorded on-time and report propellant; others drive
    # the plumes from the applied force and report delta-v.
    thrusters = None
    if (run_dir / "thrusters.npz").is_file():
        with np.load(run_dir / "thrusters.npz") as data:
            thrusters = {key: data[key] for key in ("actual_on_s", "hardware_matrix")}
        thrust = np.linalg.norm(thrusters["hardware_matrix"][:3], axis=0)
        used = np.r_[0.0, np.cumsum(thrusters["actual_on_s"] @ thrust)] / (
            SPECIFIC_IMPULSE_S * STANDARD_GRAVITY_MPS2)
    else:
        used = np.r_[0.0, np.cumsum(np.linalg.norm(forces, axis=-1))] * dt_s / manifest[
            "masses_kg"][1]
    # Without camera images (lidar-only runs) the inset shows the latest lidar scan.
    captures = read_jsonl(run_dir / "observations.jsonl")
    sensor = "Docking camera"
    if not any(row.get("image_path") for row in captures) and (run_dir / "lidar.jsonl").is_file():
        captures, sensor = read_jsonl(run_dir / "lidar.jsonl"), "Docking lidar (depth)"
    capture_times = np.array([row["capture_time_ns"] for row in captures])
    count = len(trajectory["tick"])
    schedule = frame_ticks(count, dt_s, terminal_s, latch_tick)
    ticks = schedule if ticks is None else list(ticks)
    # The camera holds the normal side the Sun lights over the whole run, so it never jumps.
    # The side comes from the full schedule, so a partial render (stills, a preview) films the
    # same side as the full movie.
    side = _sun_side(basis[schedule], sunlight_n)

    frames = output / "frames"
    movie = output / "showcase.mp4"
    video = _Video(movie, FPS, (VIEW.width, VIEW.height))
    started, render_seconds, shown, inset, encoded = time.perf_counter(), [], None, None, None
    try:
        # One thread finishes and encodes each frame while Blender renders the next.
        worker = "m11_showcase_worker.py" if tumbling else "showcase_worker.py"
        with _ShowcaseRenderer(output, worker) as renderer, ThreadPoolExecutor(1) as encoder:
            for count_index, index in enumerate(ticks):
                seconds = int(trajectory["tick"][index]) * dt_s
                camera_n, camera_nc = chase_camera(positions[index, 0], positions[index, 1],
                                                   basis[index], side)
                path = frames / f"{count_index:06d}.png"
                step = min(index, len(forces) - 1)
                throttles = (list(thrusters["actual_on_s"][step] / dt_s) if thrusters is not None
                             else nozzle_throttles(plume_throttles(forces[step], force_limit)))
                poses = [(camera_nc.T @ rotations[index, body],
                          camera_nc.T @ (positions[index, body] - camera_n)) for body in (0, 1)]
                earth = (camera_nc.T @ earth_rotation_ne(seconds), camera_nc.T @ -camera_n)
                render_seconds.append(renderer.render_view(
                    path, *poses, earth, camera_nc.T @ sunlight_n, camera_nc,
                    throttles)["render_seconds"])
                latest = int(np.searchsorted(capture_times, seconds * 1e9, side="right")) - 1
                if latest != shown:
                    row = captures[latest] if latest >= 0 else {}
                    if sensor.startswith("Docking lidar"):
                        inset = _lidar_inset(run_dir / row["scan_path"]) if row else None
                    else:
                        image_path = row.get("image_path")
                        inset = None if image_path is None else read_rgb(run_dir / image_path)
                    shown = latest
                following = ticks[count_index + 1] if count_index + 1 < len(ticks) else index + 1
                speed = (following - index) * dt_s * FPS
                minutes, rest = divmod(seconds, 60)
                rows = [("Mission time", f"T+{int(minutes):02d}:{rest:04.1f}  ({speed:g}x)"),
                        ("Port gap", f"{gaps[index]:6.2f} m"),
                        ("Closing", f"{100 * closing[index]:6.2f} cm/s"),
                        ("Phase", _phase_label(int(trajectory["phase"][index]), seconds,
                                               gaps[index], terminal_s, latch_s))]
                if tumbling:
                    rows.append(("Stack rate" if latch_s is not None and seconds >= latch_s
                                 else "Target tumble", f"{target_rate_deg_s[index]:5.2f} deg/s"))
                rows.append(("Propellant", f"{1000 * used[min(index, len(used) - 1)]:5.1f} g")
                            if thrusters is not None else
                            ("Delta-v", f"{used[min(index, len(used) - 1)]:5.2f} m/s"))
                if encoded is not None:
                    encoded.result()
                lidar = sensor.startswith("Docking lidar")
                label = (sensor + ", inside its minimum range" if latch_tick is not None
                         and gaps[index] < 1.3 and lidar else sensor)
                # Latched, the lidar sits at the target's face and returns only noise.
                latched = latch_tick is not None and index >= latch_tick
                encoded = encoder.submit(_encode, video, path, count_index,
                                         None if latched and lidar else inset,
                                         run_dir.name, rows, label, stills)
                if count_index % 30 == 0:
                    print(f"frame {count_index + 1}/{len(ticks)}", flush=True)
            if encoded is not None:
                encoded.result()
            metadata = renderer.metadata
    finally:
        video.close()
    if not stills and frames.is_dir() and not any(frames.iterdir()):
        frames.rmdir()
    summary = {
        "source_run": str(run_dir.resolve()),
        "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "purpose": "presentation only; not used by physics, autonomy, or evaluation",
        "camera": {key: VIEW.to_dict()[key] for key in ("width", "height", "hfov_deg", "K")},
        "shot": {
            "look_at": "midpoint of target and chaser centers of mass",
            "offset_direction_lvlh": OFFSET_LVLH,
            "normal_side": side,
            "up": "target LVLH +radial",
            "distance_m": f"hypot({DISTANCE_PER_SEPARATION} * separation, {MIN_DISTANCE_M})",
        },
        "target": "unmarked, tumbling (M11 worker)" if tumbling else "M8 realistic target",
        "assets": {name: {"url": url, "sha256": digest} for name, (url, digest) in ASSETS.items()},
        "credits": ASSET_CREDITS, "exposure": EXPOSURE, "star_strength": STAR_STRENGTH,
        "city_lights": CITY_LIGHTS, "plume_glow": PLUME_GLOW,
        "plumes": "recorded on-times per nozzle" if thrusters is not None else
                  "applied force per body axis",
        "ticks_per_frame": ({"approach": APPROACH_TICKS, "terminal": TERMINAL_TICKS,
                             "contact_to_end": CONTACT_TICKS} if latch_tick is not None
                            else TICKS_PER_FRAME),
        "fps": FPS, "frames": len(render_seconds), "video_codec": video.codec,
        "denoiser": "OpenImageDenoise (CPU, fast quality and prefilter)",
        "lens": {"bloom": {"threshold": BLOOM_THRESHOLD, "sigma_px": BLOOM_SIGMA_PX,
                           "gain": BLOOM_GAIN},
                 "streaks": {"threshold": STREAK_THRESHOLD, "length_px": STREAK_LENGTH_PX,
                             "gain": STREAK_GAIN, "angles_deg": [0, 90]},
                 "vignette": VIGNETTE, "grain_dn": GRAIN_DN},
        "renderer": {key: metadata[key] for key in (
            "blender_version", "samples", "device", "devices", "worker_sha256")},
        "render_seconds_median": float(np.median(render_seconds)),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "showcase.json").write_text(json.dumps(json_value(summary), indent=2), encoding="utf-8")
    return movie
