"""Presentation-only movie of a saved visual run, filmed by outside cameras.

Reads a validated schema-3 record and renders it in the realistic scene, with a chaser,
city lights, stars and thruster plumes added, under the run's own Sun. Nothing here feeds
physics, autonomy, or evaluation.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera import CameraModel, earth_rotation_ne
from .frames import port_relative_state, quaternion_matrices, relative_state
from .rendering import ASSET_CREDITS, ASSETS, BlenderRenderer, realistic_scene
from .vision_records import json_value, new_output, read_jsonl, read_rgb, sha256_file
from .visual_recording import load_visual_run
from .visualization import PHASE_NAMES


VIEW = CameraModel(width=1920, height=1080, hfov_deg=50.0, camera_id="showcase")
# Wide shot: camera direction from the spacecraft midpoint in target LVLH (radial,
# along-track, normal), slightly above and behind the chaser, looking across the approach.
# The normal component's sign follows the Sun, so the camera sees the lit sides.
OFFSET_LVLH = np.array([0.25, -0.2, -0.95]) / np.linalg.norm([0.25, -0.2, -0.95])
DISTANCE_PER_SEPARATION = 1.5
MIN_DISTANCE_M = 12.0
# Close shot, after the wide shot's first ESTABLISHING_S of the run: over the chaser's
# shoulder at this LVLH offset from its center of mass (normal sign as above), aimed midway
# between the chaser and the target's docking port.
CLOSE_OFFSET_LVLH_M = np.array([1.2, -5.0, -2.4])
ESTABLISHING_S = 30.0
TICKS_PER_FRAME = 3
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
BLOOM_THRESHOLD, BLOOM_SIGMA_PX, BLOOM_GAIN = 190.0, 6.0 * SCALE, 0.8


def _look(position, aim, up):
    """Optical-camera-to-inertial rotation looking from `position` at `aim` with `up` up."""
    forward = (aim - position) / np.linalg.norm(aim - position)
    down = (up @ forward) * forward - up
    down /= np.linalg.norm(down)
    return np.column_stack((np.cross(down, forward), down, forward))


def _sun_side(basis_nl, sunlight_n):
    """+1 or -1: the LVLH normal side the Sun shines from (`sunlight_n` is the incident ray)."""
    return 1.0 if (basis_nl.T @ sunlight_n)[2] >= 0 else -1.0


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


def close_camera(chaser_n, port_n, basis_nl, sunlight_n):
    """Close shot: over the chaser's shoulder toward the target's docking port, LVLH radial
    up, aimed midway between their directions so both stay in frame at any range."""
    offset = CLOSE_OFFSET_LVLH_M * [1, 1, _sun_side(basis_nl, sunlight_n)]
    position = np.asarray(chaser_n) + basis_nl @ offset
    rays = [point - position for point in (np.asarray(chaser_n), np.asarray(port_n))]
    aim = position + sum(ray / np.linalg.norm(ray) for ray in rays)
    return position, _look(position, aim, basis_nl[:, 0])


def plume_throttles(force_body_n, limit_n):
    """Throttle in [0, 1] for exhaust along +x, -x, +y, -y, +z, -z of the chaser body.

    Exhaust along d pushes the chaser along -d.
    """
    force = np.clip(np.asarray(force_body_n, dtype=np.float64) / limit_n, -1, 1)
    return [float(value) for axis in force for value in (max(-axis, 0.0), max(axis, 0.0))]


class _ShowcaseRenderer(BlenderRenderer):
    """BlenderRenderer process handling, driving the showcase worker."""

    def __init__(self, output_dir):
        scene = {**realistic_scene(samples=SAMPLES), "exposure": EXPOSURE,
                 "star_strength": STAR_STRENGTH, "city_lights": CITY_LIGHTS,
                 "plume_glow": PLUME_GLOW}
        super().__init__(output_dir, camera=VIEW, scene=scene)
        self._worker = Path(__file__).with_name("showcase_worker.py").resolve()

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


def _text(image, text, origin, scale):
    for color, thickness in (((0, 0, 0), round(4 * SCALE)), ((255, 255, 255), round(SCALE))):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale * SCALE, color,
                    thickness, cv2.LINE_AA)


def _bloom(image):
    """Soft glow around highlights, as a camera lens adds."""
    glow = cv2.GaussianBlur(np.clip(image.astype(np.float32) - BLOOM_THRESHOLD, 0, None),
                            (0, 0), BLOOM_SIGMA_PX)
    return np.clip(image + BLOOM_GAIN * glow, 0, 255).astype(np.uint8)


def _overlay(frame, inset, lines, close):
    """Docking-camera inset at the bottom, clear of the spacecraft (centre in the wide shot,
    right in the close shot), and labels at the top left."""
    height, width = frame.shape[:2]
    margin = round(16 * SCALE)
    x = width - INSET_PX - margin if close else (width - INSET_PX) // 2
    y = height - INSET_PX - margin
    if inset is None:
        frame[y:y + INSET_PX, x:x + INSET_PX] = 20
    else:
        frame[y:y + INSET_PX, x:x + INSET_PX] = cv2.resize(
            inset, (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (x - 1, y - 1), (x + INSET_PX, y + INSET_PX), (200, 200, 200), 1)
    _text(frame, "Docking camera" + ("" if inset is not None else ": no image"),
          (x, y - round(8 * SCALE)), 0.5)
    for row, line in enumerate(lines):
        _text(frame, line, (margin, round((32 + 28 * row) * SCALE)), 0.7)
    return frame


def _encode(writer, path, inset, lines, close):
    """Bloom and overlay the clean render at `path`, then append it to the video."""
    frame = _overlay(_bloom(read_rgb(path)), inset, lines, close)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def render_showcase(run_dir, output, ticks=None):
    """Render `run_dir` to `output`/showcase.mp4; `ticks` defaults to every third tick."""
    run_dir = Path(run_dir)
    manifest, trajectory = load_visual_run(run_dir)
    output = new_output(output)
    positions, quaternions = trajectory["positions_m"], trajectory["quaternions_xyzw"]
    rotations = quaternion_matrices(quaternions)
    basis = relative_state(positions, trajectory["velocities_mps"])[2]
    gaps = -port_relative_state(
        positions, trajectory["velocities_mps"], quaternions,
        trajectory["angular_velocities_body_radps"], manifest["docking_ports_body_m"],
    )[0][:, 1]
    target_port_body = np.asarray(manifest["docking_ports_body_m"][0], dtype=np.float64)
    forces = trajectory["applied_forces_body_n"][:, 1]
    force_limit = float(manifest["force_limits_n"][1])
    sunlight_n = np.asarray(manifest["sunlight_direction_inertial"], dtype=np.float64)
    captures = read_jsonl(run_dir / "observations.jsonl")
    capture_times = np.array([row["capture_time_ns"] for row in captures])
    ticks = range(0, len(trajectory["tick"]), TICKS_PER_FRAME) if ticks is None else ticks

    frames = output / "frames"
    movie = output / "showcase.mp4"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (VIEW.width, VIEW.height))
    if not writer.isOpened():
        raise OSError(f"cannot open video writer: {movie}")
    started, render_seconds, shown, inset, encoded = time.perf_counter(), [], None, None, None
    try:
        # One thread overlays and encodes each frame while Blender renders the next.
        with _ShowcaseRenderer(output) as renderer, ThreadPoolExecutor(1) as encoder:
            for count, index in enumerate(ticks):
                time_ns = int(trajectory["tick"][index]) * manifest["dt_ns"]
                close = time_ns / 1e9 >= ESTABLISHING_S
                if close:
                    port_n = positions[index, 0] + rotations[index, 0] @ target_port_body
                    position_nc, rotation_nc = close_camera(positions[index, 1], port_n,
                                                            basis[index], sunlight_n)
                else:
                    position_nc, rotation_nc = chase_camera(
                        positions[index, 0], positions[index, 1], basis[index], sunlight_n)

                def pose(body):
                    return (rotation_nc.T @ rotations[index, body],
                            rotation_nc.T @ (positions[index, body] - position_nc))

                path = frames / f"{count:06d}.png"
                earth = (rotation_nc.T @ earth_rotation_ne(time_ns / 1e9),
                         rotation_nc.T @ -position_nc)
                throttles = plume_throttles(forces[min(index, len(forces) - 1)], force_limit)
                result = renderer.render_view(path, pose(0), pose(1), earth,
                                              rotation_nc.T @ sunlight_n, rotation_nc, throttles)
                render_seconds.append(result["render_seconds"])
                latest = int(np.searchsorted(capture_times, time_ns, side="right")) - 1
                if latest != shown:
                    image_path = captures[latest]["image_path"] if latest >= 0 else None
                    inset = None if image_path is None else read_rgb(run_dir / image_path)
                    shown = latest
                lines = (
                    run_dir.name,
                    f"T+ {time_ns / 1e9:6.1f} s",
                    f"Port gap {gaps[index]:5.1f} m",
                    f"Phase: {PHASE_NAMES[int(trajectory['phase'][index])]}",
                )
                if encoded is not None:
                    encoded.result()
                encoded = encoder.submit(_encode, writer, path, inset, lines, close)
                if count % 30 == 0:
                    print(f"frame {count + 1}/{len(ticks)}", flush=True)
            if encoded is not None:
                encoded.result()
            metadata = renderer.metadata
    finally:
        writer.release()
    summary = {
        "source_run": str(run_dir.resolve()),
        "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "purpose": "presentation only; not used by physics, autonomy, or evaluation",
        "camera": {key: VIEW.to_dict()[key] for key in ("width", "height", "hfov_deg", "K")},
        "wide_shot": {
            "until_s": ESTABLISHING_S,
            "look_at": "midpoint of target and chaser centers of mass",
            "offset_direction_lvlh": OFFSET_LVLH,
            "normal_side": "sign of the toward-Sun LVLH normal component",
            "up": "target LVLH +radial",
            "distance_m": f"max({DISTANCE_PER_SEPARATION} * separation, {MIN_DISTANCE_M})",
        },
        "close_shot": {
            "from_s": ESTABLISHING_S, "offset_from_chaser_lvlh_m": CLOSE_OFFSET_LVLH_M,
            "look_at": "midway between the directions to the chaser and the target docking port",
        },
        "assets": {name: {"url": url, "sha256": digest} for name, (url, digest) in ASSETS.items()},
        "credits": ASSET_CREDITS, "exposure": EXPOSURE, "star_strength": STAR_STRENGTH,
        "city_lights": CITY_LIGHTS, "plume_glow": PLUME_GLOW,
        "ticks_per_frame": TICKS_PER_FRAME, "fps": FPS, "frames": len(render_seconds),
        "video_codec": "mp4v", "denoiser": "OpenImageDenoise (CPU, fast quality and prefilter)",
        "bloom": {"threshold": BLOOM_THRESHOLD, "sigma_px": BLOOM_SIGMA_PX, "gain": BLOOM_GAIN},
        "renderer": {key: metadata[key] for key in (
            "blender_version", "samples", "device", "devices", "worker_sha256")},
        "render_seconds_median": float(np.median(render_seconds)),
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "showcase.json").write_text(json.dumps(json_value(summary), indent=2), encoding="utf-8")
    return movie
