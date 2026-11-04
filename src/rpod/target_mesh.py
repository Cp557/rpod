"""The unmarked M11 target: its renderer and its rendered geometry, exported once from Blender.

`UnmarkedRenderer` drives `m11_worker.py`, the realistic scene without markers, reflectors or
Sun-tracking wings. `export_target_mesh` loads that scene once and writes every rendered
target triangle in body axes, so lidar ray casts, ICP and keypoint labels share the camera's
geometry.
"""

from pathlib import Path
import tempfile

import numpy as np

from ._core import TargetMesh
from .rendering import BlenderRenderer, realistic_scene

MESH_PATH = Path("datasets/m11/target_mesh.npz")


class UnmarkedRenderer(BlenderRenderer):
    """BlenderRenderer process handling, driving the unmarked-target worker."""

    def __init__(self, output_dir, camera=None, scene=None, blender_path=None):
        super().__init__(output_dir, blender_path=blender_path, camera=camera,
                         scene=scene or realistic_scene())
        self._worker = Path(__file__).with_name("m11_worker.py").resolve()


def export_target_mesh(path=MESH_PATH, blender_path=None):
    """Write the unmarked target's triangles in body axes; return the removed aids' names."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        scene = {**realistic_scene(), "export_mesh_path": str(path)}
        with UnmarkedRenderer(directory, scene=scene, blender_path=blender_path) as renderer:
            return renderer.metadata["cooperative_aids_removed"]


def load_target_mesh(path=MESH_PATH):
    with np.load(path) as data:
        return TargetMesh(np.ascontiguousarray(data["vertices_body_m"], dtype=np.float64),
                          np.ascontiguousarray(data["triangles"], dtype=np.uint32))
