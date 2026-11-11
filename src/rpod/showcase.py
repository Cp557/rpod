"""Presentation-only movie of a saved visual run, filmed by outside cameras.

Reads a validated schema-3 record and renders it in the realistic scene, with a chaser,
city lights, stars and thruster plumes added, under the run's own Sun. An M11 run is filmed
around the unmarked target, with the close shot riding in its tumbling body axes. A docking
run (M13, M14) gets a contact shot beside the docking interface and a detumble shot circling
the joined pair, and is filmed faster through the approach than around contact. A run with
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
from .visual_recording import load_visual_events, load_visual_run
from .visualization import PHASE_NAMES


VIEW = CameraModel(width=1920, height=1080, hfov_deg=50.0, camera_id="showcase")
# Wide shot: camera direction from the spacecraft midpoint in target LVLH (radial,
# along-track, normal), slightly above and behind the chaser, looking across the approach.
# The normal component's sign follows the Sun, so the camera sees the lit sides.
OFFSET_LVLH = np.array([0.25, -0.2, -0.95]) / np.linalg.norm([0.25, -0.2, -0.95])
DISTANCE_PER_SEPARATION = 1.5
MIN_DISTANCE_M = 12.0
# Close shot, after the wide shot's first ESTABLISHING_S of the run: over the chaser's
# shoulder at this offset from its center of mass, aimed midway between the chaser and the
# target's docking port. The offset is in target LVLH, or in the target's body axes when it
# tumbles (the two coincide for a target holding LVLH), with the normal sign toward the Sun.
CLOSE_OFFSET_M = np.array([1.2, -5.0, -2.4])
ESTABLISHING_S = 30.0
# Docking runs: from CONTACT_WINDOW_S[0] around the latch, a camera beside the docking
# interface at this offset (target body axes) from the midpoint of the two ports; from
# CONTACT_WINDOW_S[1], the same view pulled back to ORBIT_RADIUS_M from the pair's midpoint,
# in the target's body axes frozen at the latch and carried with LVLH, circling slowly about
# their +x, so the stack's tumble is seen to stop.
CONTACT_OFFSET_M = np.array([2.4, -3.6, 2.0])
CONTACT_WINDOW_S = (-12.0, 6.0)
ORBIT_RADIUS_M, ORBIT_RATE_RADPS = 9.0, np.radians(2.0)
TICKS_PER_FRAME = 3
# Docking runs are filmed at these physics ticks per frame: 18x real time through the
# approach, 9x from the terminal segment, 3x from the contact window to the end.
APPROACH_TICKS, TERMINAL_TICKS, CONTACT_TICKS = 6, 3, 1
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


def chase_camera(target_n, chaser_n, basis_nl, sunlight_n):
    """Wide shot: camera origin and optical-camera-to-inertial rotation framing both spacecraft.

    The camera looks at the midpoint of the two centers of mass, with LVLH radial up, from
    the normal side the Sun shines from.
    """
    middle = (np.asarray(target_n) + np.asarray(chaser_n)) / 2
    distance = max(DISTANCE_PER_SEPARATION * np.linalg.norm(np.subtract(chaser_n, target_n)),
                   MIN_DISTANCE_M)
    position = middle + distance * (basis_nl @ (OFFSET_LVLH * [1, 1, _sun_side(basis_nl, sunlight_n)]))
    return position, _look(position, middle, basis_nl[:, 0])


def close_camera(chaser_n, port_n, frame_nl, side):
    """Close shot: over the chaser's shoulder toward the target's docking port, frame +x up,
    aimed midway between their directions so both stay in frame at any range. `side` (+1 or
    -1) is the frame's normal side the camera takes."""
    position = np.asarray(chaser_n) + frame_nl @ (CLOSE_OFFSET_M * [1, 1, side])
    rays = [point - position for point in (np.asarray(chaser_n), np.asarray(port_n))]
    aim = position + sum(ray / np.linalg.norm(ray) for ray in rays)
    return position, _look(position, aim, frame_nl[:, 0])


def contact_camera(port_n, chaser_port_n, frame_nb, side):
    """Contact shot: beside the docking interface, riding in the target's body axes, looking
    across the gap at the midpoint of the two ports, body +x up."""
    middle = (np.asarray(port_n) + np.asarray(chaser_port_n)) / 2
    position = middle + frame_nb @ (CONTACT_OFFSET_M * [1, 1, side])
    return position, _look(position, middle, frame_nb[:, 0])


def orbit_camera(center_n, frame_nb, angle, side):
    """Detumble shot: the contact shot's direction pulled back to `ORBIT_RADIUS_M` from
    `center_n`, turned by `angle` about the frame's +x, which is up."""
    cosine, sine = np.cos(angle), np.sin(angle)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    offset = turn @ (CONTACT_OFFSET_M * [1, 1, side])
    position = np.asarray(center_n) + ORBIT_RADIUS_M * frame_nb @ (offset / np.linalg.norm(offset))
    return position, _look(position, np.asarray(center_n), frame_nb[:, 0])


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
    `CONTACT_TICKS` from the contact window on; always ending on the last tick."""
    if latch_tick is None:
        ticks = list(range(0, count, TICKS_PER_FRAME))
    else:
        contact_s = latch_tick * dt_s + CONTACT_WINDOW_S[0]
        ticks, index = [], 0
        while index < count:
            ticks.append(index)
            seconds = index * dt_s
            index += (CONTACT_TICKS if seconds >= contact_s else TERMINAL_TICKS
                      if terminal_s is not None and seconds >= terminal_s else APPROACH_TICKS)
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


def _overlay(frame, inset, title, rows, sensor, note=None):
    """A telemetry panel at the top left (label, value rows), an optional alert line under
    it, and the docking-sensor inset at the bottom right."""
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
    if note:
        box = draw.textbbox((0, 0), note, font=label)
        y = panel[3] + round(8 * SCALE)
        draw.rounded_rectangle((margin, y, margin + box[2] + 2 * inner, y + box[3] + inner),
                               radius=round(6 * SCALE), fill=(150, 70, 0, 190))
        draw.text((margin + inner, y + inner // 2), note, font=label, fill=(255, 255, 255))
    x, y = width - INSET_PX - margin, height - INSET_PX - margin
    tile = (np.full((INSET_PX, INSET_PX, 3), 20, np.uint8) if inset is None else
            cv2.resize(inset, (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA))
    image.paste(Image.fromarray(tile), (x, y))
    draw.rectangle((x - 1, y - 1, x + INSET_PX, y + INSET_PX), outline=(200, 200, 200))
    draw.text((x, y - round(20 * SCALE)), sensor + ("" if inset is not None else ": no image"),
              font=label, fill=(230, 230, 230))
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


def _encode(video, path, index, inset, title, rows, sensor, note, stills):
    """Lens effects and overlay on the clean render at `path`, then append it to the video;
    the clean render is deleted unless `stills` keeps the finished frame beside it."""
    frame = _overlay(_lens(read_rgb(path), index), inset, title, rows, sensor, note)
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
    events = load_visual_events(run_dir, manifest)
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
    ports = np.asarray(manifest["docking_ports_body_m"], dtype=np.float64)
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
    isolations = [event for event in events if event["kind"] == "safety_detection"]
    # Without camera images (lidar-only runs) the inset shows the latest lidar scan.
    captures = read_jsonl(run_dir / "observations.jsonl")
    sensor = "Docking camera"
    if not any(row.get("image_path") for row in captures) and (run_dir / "lidar.jsonl").is_file():
        captures, sensor = read_jsonl(run_dir / "lidar.jsonl"), "Docking lidar (depth)"
    capture_times = np.array([row["capture_time_ns"] for row in captures])
    count = len(trajectory["tick"])
    ticks = frame_ticks(count, dt_s, terminal_s, latch_tick) if ticks is None else list(ticks)
    close_ticks = [index for index in ticks if index * dt_s >= ESTABLISHING_S]
    # The close shot rides in the tumbling target's body axes, on the side the Sun lights for
    # most of the shot, so the camera never jumps.
    shot_frames = rotations[:, 0] if tumbling else basis
    side = _sun_side(shot_frames[close_ticks], sunlight_n) if close_ticks else 1.0

    frames = output / "frames"
    movie = output / "showcase.mp4"
    video = _Video(movie, FPS, (VIEW.width, VIEW.height))
    started, render_seconds, shown, inset, encoded = time.perf_counter(), [], None, None, None
    shots = []
    try:
        # One thread finishes and encodes each frame while Blender renders the next.
        worker = "m11_showcase_worker.py" if tumbling else "showcase_worker.py"
        with _ShowcaseRenderer(output, worker) as renderer, ThreadPoolExecutor(1) as encoder:
            for count_index, index in enumerate(ticks):
                seconds = int(trajectory["tick"][index]) * dt_s
                port_n = positions[index, 0] + rotations[index, 0] @ ports[0]
                if latch_s is not None and seconds >= latch_s + CONTACT_WINDOW_S[1]:
                    shot = "detumble"
                    frozen_nb = basis[index] @ basis[latch_tick].T @ rotations[latch_tick, 0]
                    position_nc, rotation_nc = orbit_camera(
                        positions[index].mean(axis=0), frozen_nb,
                        ORBIT_RATE_RADPS * (seconds - latch_s - CONTACT_WINDOW_S[1]), side)
                elif latch_s is not None and seconds >= latch_s + CONTACT_WINDOW_S[0]:
                    shot = "contact"
                    chaser_port_n = positions[index, 1] + rotations[index, 1] @ ports[1]
                    position_nc, rotation_nc = contact_camera(port_n, chaser_port_n,
                                                              rotations[index, 0], side)
                elif seconds >= ESTABLISHING_S:
                    shot = "close"
                    position_nc, rotation_nc = close_camera(positions[index, 1], port_n,
                                                            shot_frames[index], side)
                else:
                    shot = "wide"
                    position_nc, rotation_nc = chase_camera(
                        positions[index, 0], positions[index, 1], basis[index], sunlight_n)
                if not shots or shots[-1][0] != shot:
                    shots.append((shot, seconds))

                def pose(body):
                    return (rotation_nc.T @ rotations[index, body],
                            rotation_nc.T @ (positions[index, body] - position_nc))

                path = frames / f"{count_index:06d}.png"
                earth = (rotation_nc.T @ earth_rotation_ne(seconds),
                         rotation_nc.T @ -position_nc)
                step = min(index, len(forces) - 1)
                throttles = (list(thrusters["actual_on_s"][step] / dt_s) if thrusters is not None
                             else nozzle_throttles(plume_throttles(forces[step], force_limit)))
                result = renderer.render_view(path, pose(0), pose(1), earth,
                                              rotation_nc.T @ sunlight_n, rotation_nc, throttles)
                render_seconds.append(result["render_seconds"])
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
                isolated = [event for event in isolations if event["tick"] <= index]
                note = None
                if isolated:
                    name = isolated[-1]["reason"].removesuffix("_isolated").split("_", 2)
                    note = f"Thruster {int(name[1])} {name[2].replace('_', ' ')}: isolated"
                if encoded is not None:
                    encoded.result()
                label = (sensor + ", inside its minimum range" if latch_tick is not None
                         and gaps[index] < 1.3 and sensor.startswith("Docking lidar") else sensor)
                encoded = encoder.submit(_encode, video, path, count_index, inset,
                                         run_dir.name, rows, label, note, stills)
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
        "shots": [{"shot": shot, "from_s": seconds} for shot, seconds in shots],
        "wide_shot": {
            "until_s": ESTABLISHING_S,
            "look_at": "midpoint of target and chaser centers of mass",
            "offset_direction_lvlh": OFFSET_LVLH,
            "normal_side": "sign of the toward-Sun LVLH normal component",
            "up": "target LVLH +radial",
            "distance_m": f"max({DISTANCE_PER_SEPARATION} * separation, {MIN_DISTANCE_M})",
        },
        "close_shot": {
            "from_s": ESTABLISHING_S, "offset_from_chaser_m": CLOSE_OFFSET_M,
            "frame": "target body axes" if tumbling else "target LVLH", "normal_side": side,
            "look_at": "midway between the directions to the chaser and the target docking port",
        },
        "contact_shot": {"window_s_about_latch": CONTACT_WINDOW_S,
                         "offset_from_port_midpoint_m": CONTACT_OFFSET_M,
                         "frame": "target body axes"},
        "detumble_shot": {"radius_m": ORBIT_RADIUS_M, "rate_deg_s": np.degrees(ORBIT_RATE_RADPS),
                          "frame": "target body axes at the latch, carried with LVLH"},
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
