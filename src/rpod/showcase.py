"""Presentation-only movie of a saved visual run, filmed by an outside chase camera.

Reads a validated schema-3 record and renders it with detailed spacecraft, NASA Earth and
star imagery, and the run's own Sun. Nothing here feeds physics, autonomy, or evaluation.
"""

import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

import cv2
import numpy as np

from .camera import CameraModel
from .frames import port_relative_state, quaternion_matrices, relative_state
from .rendering import BlenderRenderer, realistic_scene
from .vision_records import json_value, new_output, read_jsonl, read_rgb, sha256_file
from .visual_recording import load_visual_run
from .visualization import PHASE_NAMES


VIEW = CameraModel(width=1280, height=720, hfov_deg=50.0, camera_id="showcase")
# Camera direction from the spacecraft midpoint in target LVLH (radial, along-track, normal):
# slightly above and behind the chaser, looking across the approach. The normal component's
# sign follows the Sun, so the camera sees the lit sides.
OFFSET_LVLH = np.array([0.25, -0.2, -0.95]) / np.linalg.norm([0.25, -0.2, -0.95])
DISTANCE_PER_SEPARATION = 1.5
MIN_DISTANCE_M = 12.0
TICKS_PER_FRAME = 3
FPS = 30
SAMPLES = 16
EXPOSURE = 0.5
# A camera exposed for sunlit spacecraft would not record stars; this keeps the brightest faint.
STAR_STRENGTH = 0.15
INSET_PX = 270
BLOOM_THRESHOLD, BLOOM_SIGMA_PX, BLOOM_GAIN = 190.0, 6.0, 0.8
# The simulation has no epoch, so Earth's rotation angle is chosen: at t = 0 the target's
# inertial longitude 32 deg sits over 70 deg E, so the view looks over southern India.
EARTH_ROTATION_DEG = -37.8
EARTH_RATE_RADPS = 7.2921159e-5
ASSET_DIR = Path(__file__).resolve().parents[2] / ".tools/showcase-assets"
EARTH_OBSERVATORY = "https://eoimages.gsfc.nasa.gov/images/imagerecords"
ASSETS = {
    "day": (f"{EARTH_OBSERVATORY}/73000/73751/world.topo.bathy.200407.3x21600x10800.jpg",
            "d225f1f35a6448a4d1d8f6de6e48f3433e470085b70a35800e64f384f269a7b0"),
    "clouds": (f"{EARTH_OBSERVATORY}/57000/57747/cloud_combined_8192.tif",
               "d137775d8966ab8d443fd5126dc6e7ad72072bc1ed50555c5818d221735daf0f"),
    "night": (f"{EARTH_OBSERVATORY}/144000/144898/BlackMarble_2016_3km.jpg",
              "230aac448ae68c358be433dd518888cccb3a85ccf66f7b44326441c324ad6725"),
    "stars": ("https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr",
              "dc6c4f413e85707a29a25a9451148154554ecca2c996f84fa8f47b65ef9ff7c4"),
}
CREDITS = ("Earth: NASA Earth Observatory (Blue Marble Next Generation, Blue Marble clouds, "
           "Black Marble 2016). Stars: NASA/Goddard Space Flight Center Scientific "
           "Visualization Studio, Deep Star Maps 2020; Gaia DR2: ESA/Gaia/DPAC.")


def showcase_assets():
    """Paths of the pinned NASA images, downloading any that are missing."""
    paths = {}
    for name, (url, digest) in ASSETS.items():
        path = ASSET_DIR / url.rsplit("/", 1)[1]
        if not path.is_file():
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as file:
                shutil.copyfileobj(response, file)
            partial.replace(path)
        if sha256_file(path) != digest:
            raise ValueError(f"showcase asset does not match its pinned sha256: {path}")
        paths[name] = path
    return paths


def earth_rotation_ne(time_s):
    """Earth-body-to-inertial rotation about inertial +Z."""
    angle = np.radians(EARTH_ROTATION_DEG) + EARTH_RATE_RADPS * time_s
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def chase_camera(target_n, chaser_n, basis_nl, sunlight_n):
    """Camera origin and optical-camera-to-inertial rotation framing both spacecraft.

    The camera looks at the midpoint of the two centers of mass, with LVLH radial up, from
    the normal side the Sun shines from (`sunlight_n` is the incident-ray direction).
    """
    middle = (np.asarray(target_n) + np.asarray(chaser_n)) / 2
    distance = max(DISTANCE_PER_SEPARATION * np.linalg.norm(np.subtract(chaser_n, target_n)),
                   MIN_DISTANCE_M)
    side = 1.0 if (basis_nl.T @ sunlight_n)[2] >= 0 else -1.0
    forward = -(basis_nl @ (OFFSET_LVLH * [1, 1, side]))
    position = middle - distance * forward
    up = basis_nl[:, 0]
    down = (up @ forward) * forward - up
    down /= np.linalg.norm(down)
    return position, np.column_stack((np.cross(down, forward), down, forward))


class _ShowcaseRenderer(BlenderRenderer):
    """BlenderRenderer process handling, driving the showcase worker."""

    def __init__(self, output_dir, assets):
        scene = {**realistic_scene(samples=SAMPLES), "exposure": EXPOSURE,
                 "star_strength": STAR_STRENGTH,
                 "showcase_assets": {name: str(path) for name, path in assets.items()}}
        super().__init__(output_dir, camera=VIEW, scene=scene)
        self._worker = Path(__file__).with_name("showcase_worker.py").resolve()

    def render_view(self, output_path, target, chaser, earth, light_direction_c, rotation_nc):
        """Render one frame; each pose is a (rotation, translation) pair in camera axes."""
        request = {
            "op": "render", "rotation_ct": target[0], "translation_ct": target[1],
            "chaser_rotation_cb": chaser[0], "chaser_translation_cb": chaser[1],
            "earth_rotation_ce": earth[0], "earth_translation_ce": earth[1],
            "light_direction_c": light_direction_c, "stars_rotation_nc": rotation_nc,
            "output_path": str(Path(output_path).resolve()),
        }
        return self._request(json_value(request))


def _text(image, text, origin, scale):
    for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)


def _bloom(image):
    """Soft glow around highlights, as a camera lens adds."""
    glow = cv2.GaussianBlur(np.clip(image.astype(np.float32) - BLOOM_THRESHOLD, 0, None),
                            (0, 0), BLOOM_SIGMA_PX)
    return np.clip(image + BLOOM_GAIN * glow, 0, 255).astype(np.uint8)


def _overlay(frame, inset, lines):
    """Docking-camera inset at the bottom centre, clear of both spacecraft, and labels at the
    top left."""
    height, width = frame.shape[:2]
    x, y = (width - INSET_PX) // 2, height - INSET_PX - 16
    if inset is None:
        frame[y:y + INSET_PX, x:x + INSET_PX] = 20
    else:
        frame[y:y + INSET_PX, x:x + INSET_PX] = cv2.resize(
            inset, (INSET_PX, INSET_PX), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (x - 1, y - 1), (x + INSET_PX, y + INSET_PX), (200, 200, 200), 1)
    _text(frame, "Docking camera" + ("" if inset is not None else ": no image"), (x, y - 8), 0.5)
    for row, line in enumerate(lines):
        _text(frame, line, (16, 32 + 28 * row), 0.7)
    return frame


def render_showcase(run_dir, output, ticks=None):
    """Render `run_dir` to `output`/showcase.mp4; `ticks` defaults to every third tick."""
    run_dir = Path(run_dir)
    manifest, trajectory = load_visual_run(run_dir)
    assets = showcase_assets()
    output = new_output(output)
    positions, quaternions = trajectory["positions_m"], trajectory["quaternions_xyzw"]
    rotations = quaternion_matrices(quaternions)
    basis = relative_state(positions, trajectory["velocities_mps"])[2]
    gaps = -port_relative_state(
        positions, trajectory["velocities_mps"], quaternions,
        trajectory["angular_velocities_body_radps"], manifest["docking_ports_body_m"],
    )[0][:, 1]
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
    started, render_seconds, shown, inset = time.perf_counter(), [], None, None
    try:
        with _ShowcaseRenderer(output, assets) as renderer:
            for count, index in enumerate(ticks):
                position_nc, rotation_nc = chase_camera(positions[index, 0], positions[index, 1],
                                                        basis[index], sunlight_n)
                time_ns = int(trajectory["tick"][index]) * manifest["dt_ns"]

                def pose(body):
                    return (rotation_nc.T @ rotations[index, body],
                            rotation_nc.T @ (positions[index, body] - position_nc))

                path = frames / f"{count:06d}.png"
                earth = (rotation_nc.T @ earth_rotation_ne(time_ns / 1e9),
                         rotation_nc.T @ -position_nc)
                result = renderer.render_view(path, pose(0), pose(1), earth,
                                              rotation_nc.T @ sunlight_n, rotation_nc)
                render_seconds.append(result["render_seconds"])
                latest = int(np.searchsorted(capture_times, time_ns, side="right")) - 1
                if latest != shown:
                    image_path = captures[latest]["image_path"] if latest >= 0 else None
                    inset = None if image_path is None else read_rgb(run_dir / image_path)
                    shown = latest
                frame = _overlay(_bloom(read_rgb(path)), inset, (
                    run_dir.name,
                    f"T+ {time_ns / 1e9:6.1f} s",
                    f"Port gap {gaps[index]:5.1f} m",
                    f"Phase: {PHASE_NAMES[int(trajectory['phase'][index])]}",
                ))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if count % 30 == 0:
                    print(f"frame {count + 1}/{len(ticks)}", flush=True)
            metadata = renderer.metadata
    finally:
        writer.release()
    summary = {
        "source_run": str(run_dir.resolve()),
        "source_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "purpose": "presentation only; not used by physics, autonomy, or evaluation",
        "camera": {key: VIEW.to_dict()[key] for key in ("width", "height", "hfov_deg", "K")},
        "camera_rule": {
            "look_at": "midpoint of target and chaser centers of mass",
            "offset_direction_lvlh": OFFSET_LVLH,
            "normal_side": "sign of the toward-Sun LVLH normal component",
            "up": "target LVLH +radial",
            "distance_m": f"max({DISTANCE_PER_SEPARATION} * separation, {MIN_DISTANCE_M})",
        },
        "earth_rotation_deg_at_t0": EARTH_ROTATION_DEG, "earth_rate_radps": EARTH_RATE_RADPS,
        "assets": {name: {"url": url, "sha256": digest} for name, (url, digest) in ASSETS.items()},
        "credits": CREDITS, "exposure": EXPOSURE, "star_strength": STAR_STRENGTH,
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
