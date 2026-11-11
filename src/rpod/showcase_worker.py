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
    ALONG_Y, _aim, _bump, _camera_only, _equirectangular, _hardware_materials, _image,
    _import_model, _math, _mix, _noise, _object_coordinates, _part, _principled, _ramp, _sphere,
    _strut, _thruster, box, cone, cylinder,
)
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402


PLUME_LENGTH_M = 0.5
# Clouds on their own shell this far above the surface, so they cast shadows on the ground.
CLOUD_ALTITUDE_M = 8_000.0
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


def _crinkle(tree, scale, strength):
    """Normal of crinkled foil: Voronoi facets with fine wrinkles, in object coordinates."""
    coordinates = _object_coordinates(tree)
    facets = tree.nodes.new("ShaderNodeTexVoronoi")
    facets.inputs["Scale"].default_value = scale
    tree.links.new(coordinates, facets.inputs["Vector"])
    height = _math(tree, "ADD", facets.outputs["Distance"],
                   _math(tree, "MULTIPLY", _noise(tree, coordinates, 6 * scale, 6), 0.35))
    return _bump(tree, height, strength, 0.02)


# Presentation looks for the NASA models' materials, by name: (metallic, roughness, crinkle
# scale per model unit or None, coat weight).
LOOKS = {
    "Shiny_Gold": (0.5, 0.4, 22.0, 0.0), "foil_gold": (0.5, 0.4, 30.0, 0.0),
    "Shiny": (0.5, 0.4, 22.0, 0.0), "Boxes": (0.5, 0.4, 18.0, 0.0),
    "foil_silver": (0.5, 0.4, 30.0, 0.0), "black_krinkle": (0.0, 0.35, 30.0, 0.0),
    "QS_aluminum": (1.0, 0.32, None, 0.0), "QS_aluminum.001": (1.0, 0.32, None, 0.0),
    "QS_aluminum1": (1.0, 0.38, None, 0.0), "QS_steel1": (1.0, 0.4, None, 0.0),
    "White": (0.0, 0.45, None, 0.0), "QS_gray": (0.0, 0.5, None, 0.0),
    "QS_gray.001": (0.0, 0.5, None, 0.0), "Default": (0.0, 0.5, None, 0.0),
    "Default.001": (0.0, 0.5, None, 0.0), "Default.002": (0.0, 0.5, None, 0.0),
    "QS_solar": (0.2, 0.3, None, 1.0), "shiny_panel": (0.9, 0.18, None, 0.0),
}


def _docking_face(material):
    """The target's docking face as white thermal paint over 0.5 m panels: seams as shallow
    grooves, and roughness varying across each panel."""
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    coordinates = _object_coordinates(tree)
    panels = tree.nodes.new("ShaderNodeTexBrick")
    panels.offset, panels.squash = 0.0, 1.0
    for name, value in (("Scale", 2.0), ("Mortar Size", 0.004), ("Mortar Smooth", 0.3),
                        ("Brick Width", 1.0), ("Row Height", 1.0)):
        panels.inputs[name].default_value = value
    tree.links.new(coordinates, panels.inputs["Vector"])
    tree.links.new(_bump(tree, panels.outputs["Fac"], 0.5, 0.003), shader.inputs["Normal"])
    blotches = _noise(tree, coordinates, 3.0, 4)
    tree.links.new(_math(tree, "MULTIPLY_ADD", blotches, 0.25, 0.4), shader.inputs["Roughness"])
    shader.inputs["Base Color"].default_value = (0.55, 0.545, 0.52, 1)


def _restyle_models():
    """Edit the imported models' materials in place: foils metallic and crinkled, paint and
    metal with plausible roughness, solar cells under glossy cover glass. Names stay, so the
    wing drive's `QS_solar` lookup still finds its meshes."""
    _docking_face(bpy.data.materials["docking_face"])
    for material in bpy.data.materials:
        look = LOOKS.get(material.name)
        if look is None or not material.use_nodes:
            continue
        tree = material.node_tree
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        metallic, roughness, crinkle, coat = look
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Coat Weight"].default_value = coat
        shader.inputs["Coat Roughness"].default_value = 0.03
        if crinkle is not None:
            tree.links.new(_crinkle(tree, crinkle, 0.3), shader.inputs["Normal"])


def _earth_without_clouds(assets, city_lights, sun_frame):
    """`blender_worker._earth_surface` with its clouds moved to their own layer and a sunglint
    ocean: water darker and smoother (roughness 0.15) than the land."""
    material, tree, shader = _principled("earth_showcase", (0, 0, 0))
    uv = _equirectangular(tree, _object_coordinates(tree), 1.0)
    day = _image(tree, assets["day"], uv)["Color"]
    rgb = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(day, rgb.inputs[0])
    water = _ramp(tree, _math(tree, "SUBTRACT", rgb.outputs[2], rgb.outputs[0]), 0.03, 0.12)
    ocean = _mix(tree, "RGBA", _ramp(tree, rgb.outputs[2], 0.1, 0.6),
                 (0.003, 0.01, 0.03), (0.008, 0.025, 0.06))
    tree.links.new(_mix(tree, "RGBA", water, day, ocean), shader.inputs["Base Color"])
    tree.links.new(_mix(tree, "VALUE", water, 0.85, 0.15), shader.inputs["Roughness"])
    if city_lights:
        night = _image(tree, assets["night"], uv)["Color"]
        sun_space = tree.nodes.new("ShaderNodeTexCoord")
        sun_space.object = sun_frame
        toward_sun = tree.nodes.new("ShaderNodeSeparateXYZ")
        normalize = tree.nodes.new("ShaderNodeVectorMath")
        normalize.operation = "NORMALIZE"
        tree.links.new(sun_space.outputs["Object"], normalize.inputs[0])
        tree.links.new(normalize.outputs["Vector"], toward_sun.inputs[0])
        dark = _ramp(tree, toward_sun.outputs[2], 0.05, -0.12)
        tree.links.new(night, shader.inputs["Emission Color"])
        tree.links.new(_math(tree, "MULTIPLY", dark, city_lights),
                       shader.inputs["Emission Strength"])
    material.cycles.emission_sampling = "NONE"
    return material


def _cloud_layer(assets, radius, earth):
    """Clouds `CLOUD_ALTITUDE_M` up, riding with Earth: white where the cloud map is dense,
    transparent elsewhere, so they shade the ground beneath and hide its city lights."""
    shell = _sphere("clouds", radius + CLOUD_ALTITUDE_M, 512, 256)
    shell.parent = earth
    material, tree, shader = _principled("cloud_layer", (0.88, 0.89, 0.92), roughness=1.0,
                                         specular=0.2)
    cloud = _image(tree, assets["clouds"], _equirectangular(tree, _object_coordinates(tree), 1.0),
                   "Non-Color")["Color"]
    tree.links.new(_ramp(tree, cloud, 0.15, 0.85), shader.inputs["Alpha"])
    shell.data.materials.append(material)
    return shell


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
        assets = request["scene"]["assets"]
        self._build_chaser(assets["chaser_model"],
                           _plume_material(float(request["scene"]["plume_glow"])))
        _restyle_models()
        self.earth.data.materials[0] = _earth_without_clouds(
            assets, float(request["scene"]["city_lights"]), self.sun_frame)
        self.clouds = _cloud_layer(assets, float(request["scene"]["earth_radius_m"]),
                                   self.earth)

    def _plume(self, material, pod, direction):
        """A cone of exhaust along `direction` from the nozzle at `pod`, hidden until fired."""
        direction = Vector(direction)
        obj = _part(cone, material, self.chaser, vertices=24, radius1=0.12, radius2=0.012,
                    depth=PLUME_LENGTH_M, end_fill_type="NOTHING",
                    location=Vector(pod) + direction * (0.04 + PLUME_LENGTH_M / 2))
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = _aim(direction, "-Z")
        _camera_only(obj)
        self.plumes.append(obj)

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

    def _update_realistic(self, request, rotation_ct, ray):
        super()._update_realistic(request, rotation_ct, ray)
        self.clouds.hide_render = self.earth.hide_render

    def render(self, request):
        transform = Matrix(request["chaser_rotation_cb"]).to_4x4()
        transform.translation = Vector(request["chaser_translation_cb"])
        if not all(math.isfinite(value) for row in transform for value in row):
            raise ValueError("chaser pose requires a finite 3x3 rotation and 3-vector")
        self.chaser.matrix_world = transform
        # One throttle per nozzle, in `thrusters.layout` order.
        for throttle, obj in zip(request["plume_throttles"], self.plumes, strict=True):
            obj.color = (throttle, 0, 0, 1)
            obj.hide_render = throttle <= 0
        return super().render(request)


if __name__ == "__main__":
    # blender_worker.main constructs its module-level SceneRenderer.
    blender_worker.SceneRenderer = ShowcaseRenderer
    blender_worker.main()
