"""Small file and observation helpers shared by perception and visual runs."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .observations import OwnNavigation


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write image: {path}")


def degrade_image(image, seed, noise_dn=2.0, blur_px=0.5):
    """Gaussian blur, then seeded additive Gaussian noise, rounded back to 8-bit RGB."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_px)
    noise = np.random.default_rng(seed).normal(0.0, noise_dn, image.shape)
    return np.clip(np.rint(blurred.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(json_value(record), allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def own_navigation(snapshot):
    return OwnNavigation(
        time_ns=snapshot.time_ns,
        position_m=np.array(snapshot.positions_m[1], dtype=np.float64),
        velocity_mps=np.array(snapshot.velocities_mps[1], dtype=np.float64),
        quaternion_xyzw=np.array(snapshot.quaternions_xyzw[1], dtype=np.float64),
        angular_velocity_body_radps=np.array(
            snapshot.angular_velocities_body_radps[1], dtype=np.float64
        ),
    )


def new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
