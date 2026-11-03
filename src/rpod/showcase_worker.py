"""Blender process for the presentation-only showcase movie.

Reuses the calibrated worker's realistic scene and JSON-lines protocol unchanged, and adds
NASA's IBEX model as the chaser, thruster plumes, and filmic render settings. Run with
Blender's bundled Python.
"""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_worker  # noqa: E402
from blender_worker import (  # noqa: E402
    ALONG_Y, _aim, _camera_only, _hardware_materials, _import_model, _math, _part, _strut,
    _thruster, box, cone, cylinder,
)
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402


PLUME_LENGTH_M = 0.5
# NASA's 1.5 m IBEX model scaled to the 1 m chaser, turned so its solar deck faces -Y and
# its base sits at y = 0.45, just behind the docking ring.
CHASER_MODEL_SCALE = 0.72
CHASER_MODEL_OFFSET = (0.0, 0.234, 0.0)


def _plume_material(glow):
    """Faint transparent exhaust, brightest at the nozzle; the object's colour (red) is the
    throttle."""
    material = bpy.data.materials.new("thruster_plume")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    along = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(tree.nodes.new("ShaderNodeTexCoord").outputs["Object"], along.inputs[0])
    nozzle = _math(tree, "ADD", _math(tree, "DIVIDE", along.outputs["Z"], PLUME_LENGTH_M), 0.5)
    facing = tree.nodes.new("ShaderNodeLayerWeight")
    core = _math(tree, "SUBTRACT", 1.0, facing.outputs["Facing"])
    throttle = tree.nodes.new("ShaderNodeObjectInfo").outputs["Color"]
    red = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(throttle, red.inputs[0])
    glow_node = tree.nodes.new("ShaderNodeEmission")
    glow_node.inputs["Color"].default_value = (0.75, 0.82, 1.0, 1)
    tree.links.new(_math(tree, "MULTIPLY", _math(tree, "MULTIPLY", _math(
        tree, "MULTIPLY", nozzle, core), red.outputs[0]), glow), glow_node.inputs["Strength"])
    add = tree.nodes.new("ShaderNodeAddShader")
    tree.links.new(tree.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[0])
    tree.links.new(glow_node.outputs[0], add.inputs[1])
    tree.links.new(add.outputs[0], tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    material.cycles.emission_sampling = "NONE"
    return material


class ShowcaseRenderer(blender_worker.SceneRenderer):
    def __init__(self, request):
        super().__init__(request)
        scene = self.scene
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_use_gpu = False
        scene.cycles.denoising_quality = "FAST"
        scene.cycles.denoising_prefilter = "FAST"
        # The CPU denoiser takes about half of each frame; give it every core.
        scene.render.threads_mode = "AUTO"
        scene.cycles.max_bounces = 6
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 3
        scene.cycles.transparent_max_bounces = 8
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = float(request["scene"]["exposure"])
        self.sun.data.energy = 4.0
        self.chaser = bpy.data.objects.new("chaser", None)
        scene.collection.objects.link(self.chaser)
        self.plumes = []
        self._build_chaser(request["scene"]["assets"]["chaser_model"],
                           _plume_material(float(request["scene"]["plume_glow"])))

    def _plume(self, material, pod, direction):
        """A cone of exhaust along `direction` from the nozzle at `pod`, hidden until fired."""
        direction = Vector(direction)
        obj = _part(cone, material, self.chaser, vertices=24, radius1=0.12, radius2=0.012,
                    depth=PLUME_LENGTH_M, end_fill_type="NOTHING",
                    location=Vector(pod) + direction * (0.04 + PLUME_LENGTH_M / 2))
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = _aim(direction, "-Z")
        _camera_only(obj)
        axis = next(index for index in range(3) if direction[index])
        self.plumes.append((2 * axis + (direction[axis] < 0), obj))

    def _build_chaser(self, path, plume):
        """NASA's IBEX model as the servicer, its solar deck aft and its base forward, with a
        docking ring, camera and lidar heads at their mounts, and corner thruster pods firing
        along each body axis."""
        chaser, m = self.chaser, _hardware_materials()
        _import_model(path, chaser, CHASER_MODEL_SCALE, (math.pi / 2, 0, 0), CHASER_MODEL_OFFSET)
        for x in (-1, 1):
            for z in (-1, 1):
                pod = (0.36 * x, 0.1, 0.36 * z)
                for direction in ((x, 0, 0), (0, 0, z), (0, -1, 0), (0, 1, 0)):
                    _thruster((m["nozzle"], m["aluminum"]), chaser, pod, direction, 0.6)
                    self._plume(plume, pod, direction)
        _part(cylinder, m["aluminum"], chaser, vertices=48, radius=0.18, depth=0.06,
              location=(0, 0.47, 0), rotation=ALONG_Y)
        _part(cylinder, m["aluminum"], chaser, vertices=64, radius=0.22, depth=0.05,
              location=(0, 0.525, 0), rotation=ALONG_Y)
        _part(cylinder, m["dark"], chaser, vertices=64, radius=0.15, depth=0.004,
              location=(0, 0.551, 0), rotation=ALONG_Y)
        for angle in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3):
            _strut(m["aluminum"], chaser, (0.2 * math.cos(angle), 0.55, 0.2 * math.sin(angle)),
                   (0.13 * math.cos(angle), 0.66, 0.13 * math.sin(angle)), 0.012)
        for x, radius in ((0.2, 0.03), (-0.2, 0.035)):
            _part(box, m["dark"], chaser, apply_scale=True, size=1,
                  location=(x, 0.5, 0.2), scale=(0.1, 0.12, 0.1))
            _part(cylinder, m["dark"], chaser, vertices=24, radius=0.04, depth=0.05,
                  end_fill_type="NOTHING", location=(x, 0.575, 0.2), rotation=ALONG_Y)
            _part(cylinder, m["glass"], chaser, vertices=24, radius=radius, depth=0.01,
                  location=(x, 0.565, 0.2), rotation=ALONG_Y)

    def render(self, request):
        transform = Matrix(request["chaser_rotation_cb"]).to_4x4()
        transform.translation = Vector(request["chaser_translation_cb"])
        if not all(math.isfinite(value) for row in transform for value in row):
            raise ValueError("chaser pose requires a finite 3x3 rotation and 3-vector")
        self.chaser.matrix_world = transform
        throttles = request["plume_throttles"]
        for index, obj in self.plumes:
            obj.color = (throttles[index], 0, 0, 1)
            obj.hide_render = throttles[index] <= 0
        return super().render(request)


if __name__ == "__main__":
    # blender_worker.main constructs its module-level SceneRenderer.
    blender_worker.SceneRenderer = ShowcaseRenderer
    blender_worker.main()
