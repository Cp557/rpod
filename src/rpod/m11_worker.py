"""Blender process for the unmarked, tumbling M11 target.

Reuses the realistic `SceneRenderer` and its JSON-lines protocol, and changes the target:
- the cooperative aids are removed: marker pads and tags, lidar reflectors, and the posts of
  the raised pads;
- the solar wings stay at their stowed angle instead of tracking the Sun, as on a dead
  satellite;
- with `export_mesh_path` in the scene, init writes every remaining target triangle in body
  axes to that `.npz`, so lidar ray casts and keypoint labels use the rendered geometry.
Run with Blender's bundled Python.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_worker  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
import numpy as np  # noqa: E402

# Lidar reflector discs and the 12 mm posts that hold the raised marker pads.
REFLECTOR_MATERIAL = "reflector_glass"
POST_RADIUS_M = 0.012


def _cooperative(obj, markers):
    if obj in markers or obj.name.startswith(("quiet_", "marker_")):
        return True
    if obj.type != "MESH":
        return False
    names = {material.name for material in obj.data.materials if material}
    if REFLECTOR_MATERIAL in names:
        return True
    width = obj.dimensions.x
    return ("handrail" not in names and abs(width - 2 * POST_RADIUS_M) < 1e-4
            and obj.location.y < -1.05)


class UnmarkedRenderer(blender_worker.SceneRenderer):
    def __init__(self, request):
        super().__init__(request)
        markers = set(self.marker_objects)
        removed = [obj for obj in self.target_objects if _cooperative(obj, markers)]
        for obj in removed:
            obj.hide_render = True
        self.target_objects = [obj for obj in self.target_objects if obj not in removed]
        self.removed_names = sorted(obj.name for obj in removed)
        path = (request.get("scene") or {}).get("export_mesh_path")
        if path:
            self._export(Path(path))

    def _update_realistic(self, request, rotation_ct, ray):
        stowed = self.wings.rotation_euler.copy()
        super()._update_realistic(request, rotation_ct, ray)
        self.wings.rotation_euler = stowed

    def _export(self, path):
        """Triangles of every rendered target part, in target body axes."""
        self.target.matrix_world = Matrix.Identity(4)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        vertices, triangles, parts = [], [], []
        for obj in self.target_objects:
            if obj.type != "MESH":
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            base = len(vertices)
            vertices.extend(tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices)
            triangles.extend([base + index for index in triangle.vertices]
                             for triangle in mesh.loop_triangles)
            parts.append(obj.name)
            evaluated.to_mesh_clear()
        np.savez(path, vertices_body_m=np.asarray(vertices, dtype=np.float64),
                 triangles=np.asarray(triangles, dtype=np.int64), parts=np.asarray(parts),
                 removed=np.asarray(self.removed_names))

    def metadata(self):
        return {**super().metadata(), "cooperative_aids_removed": self.removed_names,
                "wings": "stowed"}


if __name__ == "__main__":
    # blender_worker.main constructs its module-level SceneRenderer.
    blender_worker.SceneRenderer = UnmarkedRenderer
    blender_worker.main()
