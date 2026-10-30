"""Persistent Blender subprocess adapter; rendering never owns simulation time."""

from collections import deque
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from .camera import (CameraModel, MARKER_CENTERS, MARKER_DICTIONARY,
                     MARKER_QUIET_ZONE_M, MARKER_SIDE_M)
from .lidar import REFLECTOR_RADIUS_M, REFLECTOR_STANDOFF_M, REFLECTORS_BODY_M
from .scenarios import EARTH_RADIUS_M


def realistic_scene(device="CUDA", samples=16):
    """Space lighting, a detailed satellite around the marked face, and a procedural Earth.

    The lidar reflectors are drawn in the camera image too. Earth is placed per render.
    """
    return {"name": "realistic", "device": device, "samples": samples,
            "earth_radius_m": EARTH_RADIUS_M, "reflectors_body_m": REFLECTORS_BODY_M.tolist(),
            "reflector_radius_m": REFLECTOR_RADIUS_M,
            "reflector_standoff_m": REFLECTOR_STANDOFF_M}


class BlenderRenderer:
    def __init__(self, output_dir, blender_path=None, camera=None, scene=None):
        self.output_dir = Path(output_dir).resolve()
        self.camera = camera or CameraModel()
        self.scene = scene
        portable = Path(__file__).resolve().parents[2] / ".tools/blender-5.2.3-windows-x64/blender.exe"
        self.blender_path = Path(blender_path or (portable if portable.is_file() else shutil.which("blender") or portable))
        self._worker = Path(__file__).with_name("blender_worker.py").resolve()
        self._process = None
        self._responses = queue.Queue()
        self._logs = deque(maxlen=30)
        self._memory_stop = threading.Event()
        self._memory_thread = None
        self._reader_thread = None
        self._ps_process = None
        self.metadata = {}
        self.provenance = self.metadata
        self.performance = {
            "startup_seconds": None, "render_seconds": [], "request_seconds": [],
            "peak_rss_bytes": None, "rss_scope": "unavailable",
        }

    @property
    def startup_seconds(self):
        return self.performance["startup_seconds"]

    @property
    def peak_rss_bytes(self):
        return self.performance["peak_rss_bytes"]

    def _assets(self):
        import cv2

        directory = self.output_dir / "renderer_assets"
        directory.mkdir(parents=True, exist_ok=True)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        markers = []
        for marker_id, center in MARKER_CENTERS.items():
            paths = []
            hashes = []
            for rendered_id in (marker_id, marker_id + 20):
                pixels = cv2.aruco.generateImageMarker(dictionary, rendered_id, 600)
                encoded, data = cv2.imencode(".png", pixels)
                if not encoded:
                    raise RuntimeError("failed to encode marker texture")
                content = data.tobytes()
                path = directory / f"marker_{rendered_id}.png"
                path.write_bytes(content)
                paths.append(str(path))
                hashes.append(hashlib.sha256(content).hexdigest())
            markers.append({
                "id": marker_id, "center": list(center), "side_m": MARKER_SIDE_M,
                "quiet_zone_m": MARKER_QUIET_ZONE_M, "image_path": paths[0],
                "unknown_image_path": paths[1], "sha256": hashes[0],
                "unknown_sha256": hashes[1],
            })
        return markers

    def _read_output(self):
        for line in self._process.stdout:
            marker = line.find("RPOD_JSON ")
            if marker >= 0:
                try:
                    self._responses.put(json.loads(line[marker + len("RPOD_JSON "):]))
                except json.JSONDecodeError:
                    self._logs.append(line.rstrip())
            else:
                self._logs.append(line.rstrip())
        self._responses.put({"ok": False, "error": "Blender worker exited before replying"})

    def _sample_rss(self):
        if self._ps_process is None:
            return None
        try:
            value = self._ps_process.memory_info().rss
        except (OSError, self._psutil.Error):
            return None
        previous = self.performance["peak_rss_bytes"] or 0
        self.performance["peak_rss_bytes"] = max(previous, value)
        return value

    def _monitor_memory(self):
        while not self._memory_stop.wait(0.1):
            self._sample_rss()

    def _request(self, request):
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Blender renderer is not running")
        self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self._process.stdin.flush()
        try:
            reply = self._responses.get(timeout=180)
        except queue.Empty as error:
            raise RuntimeError("Blender renderer timed out; " + "\n".join(self._logs)) from error
        if not reply.get("ok"):
            raise RuntimeError(f"Blender renderer failed: {reply.get('error')}; " + "\n".join(self._logs))
        return {key: value for key, value in reply.items() if key != "ok"}

    def __enter__(self):
        if self._process is not None:
            raise RuntimeError("BlenderRenderer cannot be entered twice")
        if not self.blender_path.is_file():
            raise FileNotFoundError(f"Blender executable not found: {self.blender_path}")
        markers = self._assets()
        started = time.perf_counter()
        self._process = subprocess.Popen(
            [str(self.blender_path), "--background", "--factory-startup", "--disable-autoexec", "--python", str(self._worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            self._psutil = psutil
            try:
                self._ps_process = psutil.Process(self._process.pid)
                self.performance["rss_scope"] = "Blender process RSS sampled every 0.1 s; not OS high-water memory"
                self._sample_rss()
                self._memory_thread = threading.Thread(target=self._monitor_memory, daemon=True)
                self._memory_thread.start()
            except psutil.Error:
                self._ps_process = None
        try:
            settings = self._request({"op": "init", "camera": self.camera.to_dict(),
                                      "markers": markers, "scene": self.scene})
            self.performance["startup_seconds"] = time.perf_counter() - started
            self.metadata.update({
                **settings, "camera": self.camera.to_dict(),
                "dictionary": MARKER_DICTIONARY, "markers": markers,
                "worker_sha256": hashlib.sha256(self._worker.read_bytes()).hexdigest(),
            })
        except BaseException:
            self.close(graceful=False)
            raise
        return self

    def render(
        self, rotation_ct, translation_ct, output_path,
        light_direction_c=None, variant="target", calibration_points=None, earth=None,
    ) -> dict:
        """Render one frame; `earth` is an optional (rotation_ce, translation_ce) pose."""
        rotation = np.asarray(rotation_ct, dtype=np.float64)
        translation = np.asarray(translation_ct, dtype=np.float64)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
            raise ValueError("render requires a finite proper rotation and translation")
        light = np.asarray([0, 0, 1] if light_direction_c is None else light_direction_c, dtype=np.float64)
        if light.shape != (3,) or not np.isfinite(light).all() or np.linalg.norm(light) == 0:
            raise ValueError("incident light direction must be a finite nonzero three-vector")
        points = None if calibration_points is None else np.asarray(calibration_points, dtype=np.float64)
        if points is not None and (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
            raise ValueError("calibration points must have shape (N, 3) and be finite")
        request = {
            "op": "render", "rotation_ct": rotation.tolist(), "translation_ct": translation.tolist(),
            "light_direction_c": (light / np.linalg.norm(light)).tolist(),
            "variant": variant, "calibration_points": None if points is None else points.tolist(),
        }
        if earth is not None:
            rotation_ce, translation_ce = (np.asarray(value, dtype=np.float64) for value in earth)
            if (rotation_ce.shape != (3, 3) or translation_ce.shape != (3,)
                    or not np.isfinite(rotation_ce).all() or not np.isfinite(translation_ce).all()):
                raise ValueError("Earth pose requires a finite 3x3 rotation and 3-vector")
            request.update(earth_rotation_ce=rotation_ce.tolist(),
                           earth_translation_ce=translation_ce.tolist())
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        request["output_path"] = str(output)
        started = time.perf_counter()
        result = self._request(request)
        self.performance["request_seconds"].append(time.perf_counter() - started)
        self.performance["render_seconds"].append(result["render_seconds"])
        result["process_rss_bytes"] = self._sample_rss()
        return result

    def close(self, *, graceful=True):
        if self._process is None:
            return
        if self._process.poll() is None:
            if graceful:
                try:
                    self._request({"op": "quit"})
                except (OSError, RuntimeError):
                    pass
            if self._process.poll() is None:
                if not graceful:
                    self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
        self._memory_stop.set()
        if self._memory_thread is not None:
            self._memory_thread.join(timeout=1)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1)
        self._process.stdin.close()
        self._process.stdout.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(graceful=exc_type is None)
