"""Standalone Blender process: calibrated camera-frame scenes over JSON lines.

Run with Blender's bundled Python, never with the RPOD interpreter. The worker
consumes supplied poses and does not advance a simulation or animation clock.
"""

import json
import math
from pathlib import Path
import sys
import time
import traceback

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


PREFIX = "RPOD_JSON "
SURFACE_EPSILON_M = 0.00001
CALIBRATION_COLORS = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),
    (1, 0, 1), (1, 1, 0), (1, 0.4, 0), (0.4, 0, 1),
)


def _material(name, color=None, image_path=None, emission=False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    if emission:
        shader = nodes.new("ShaderNodeEmission")
        shader.inputs["Color"].default_value = (*color, 1)
        shader.inputs["Strength"].default_value = 1
    else:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = 1
        shader.inputs["Metallic"].default_value = 0
        shader.inputs["Specular IOR Level"].default_value = 0
        if image_path is None:
            shader.inputs["Base Color"].default_value = (*color, 1)
        else:
            texture = nodes.new("ShaderNodeTexImage")
            texture.image = bpy.data.images.load(str(Path(image_path).resolve()), check_existing=True)
            texture.image.colorspace_settings.name = "sRGB"
            texture.interpolation = "Closest"
            material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    material.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return material


def _principled(name, color, metallic=0.0, roughness=0.5, specular=0.5):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Specular IOR Level"].default_value = specular
    return material, tree, shader


def _socket(sockets, name, kind):
    return next(socket for socket in sockets
                if socket.name == name and socket.type == kind and socket.enabled)


def _noise(tree, vector, scale, detail):
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    tree.links.new(vector, noise.inputs["Vector"])
    return noise.outputs["Fac"]


def _ramp(tree, value, low, high):
    """0 below `low`, 1 above `high`, linear between."""
    ramp = tree.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = low
    ramp.inputs["From Max"].default_value = high
    tree.links.new(value, ramp.inputs["Value"])
    return ramp.outputs["Result"]


def _mix(tree, kind, factor, a, b):
    """Blend constants or sockets `a` and `b` by `factor`; kind is RGBA or VALUE."""
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA" if kind == "RGBA" else "FLOAT"
    tree.links.new(factor, _socket(mix.inputs, "Factor", "VALUE"))
    for name, value in (("A", a), ("B", b)):
        socket = _socket(mix.inputs, name, kind)
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = (*value, 1) if kind == "RGBA" else value
    return _socket(mix.outputs, "Result", kind)


def _foil(name, color):
    """Crinkled metallic multilayer-insulation blanket."""
    material, tree, shader = _principled(name, color, metallic=1, roughness=0.3)
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    coordinates = tree.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    tree.links.new(_noise(tree, coordinates, 7, 6), bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def _solar_cells():
    """Glass-covered dark cells on a metallic grid, in the panel's body x-z plane."""
    material, tree, shader = _principled("solar_cells", (0.01, 0.015, 0.05),
                                         roughness=0.12, specular=0.6)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (math.pi / 2, 0, 0)
    tree.links.new(tree.nodes.new("ShaderNodeTexCoord").outputs["Object"], mapping.inputs["Vector"])
    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0
    for name, value in (("Color1", (0.01, 0.015, 0.05, 1)), ("Color2", (0.012, 0.02, 0.06, 1)),
                        ("Mortar", (0.6, 0.6, 0.6, 1)), ("Scale", 1.0),
                        ("Mortar Size", 0.012), ("Brick Width", 0.4), ("Row Height", 0.2)):
        brick.inputs[name].default_value = value
    tree.links.new(mapping.outputs["Vector"], brick.inputs["Vector"])
    tree.links.new(brick.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(brick.outputs["Fac"], shader.inputs["Metallic"])
    return material


def _earth_surface():
    """Procedural oceans, land, and clouds evaluated on the unit normal.

    Only camera rays see the textured surface; light reflected onto the spacecraft uses a
    uniform 0.3 albedo, which Cycles evaluates without the texture nodes.
    """
    material, tree, shader = _principled("earth_surface", (0, 0, 0))
    normalize = tree.nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    tree.links.new(tree.nodes.new("ShaderNodeTexCoord").outputs["Object"], normalize.inputs[0])
    direction = normalize.outputs["Vector"]
    land = _ramp(tree, _noise(tree, direction, 1.6, 6), 0.5, 0.56)
    cloud = _ramp(tree, _noise(tree, direction, 60, 8), 0.5, 0.62)
    ground = _mix(tree, "RGBA", land, (0.006, 0.02, 0.05), (0.09, 0.08, 0.045))
    tree.links.new(_mix(tree, "RGBA", cloud, ground, (0.75, 0.75, 0.75)), shader.inputs["Base Color"])
    rough = _mix(tree, "VALUE", land, 0.12, 0.95)
    tree.links.new(_mix(tree, "VALUE", cloud, rough, 1.0), shader.inputs["Roughness"])
    albedo = tree.nodes.new("ShaderNodeBsdfDiffuse")
    albedo.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1)
    choose = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], choose.inputs[0])
    tree.links.new(albedo.outputs[0], choose.inputs[1])
    tree.links.new(shader.outputs[0], choose.inputs[2])
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    tree.links.new(choose.outputs[0], output.inputs["Surface"])
    return material


def _sphere(name, radius, segments, rings):
    """A smooth UV sphere built directly; Blender's primitive operator is slow at this size."""
    vertices = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for segment in range(segments):
            azimuth = 2 * math.pi * segment / segments
            vertices.append((radius * math.sin(polar) * math.cos(azimuth),
                             radius * math.sin(polar) * math.sin(azimuth), radius * math.cos(polar)))

    def index(ring, segment):
        return 2 + (ring - 1) * segments + segment % segments

    faces = [(0, index(1, segment), index(1, segment + 1)) for segment in range(segments)]
    faces += [(index(ring, segment), index(ring + 1, segment), index(ring + 1, segment + 1),
               index(ring, segment + 1)) for ring in range(1, rings - 1) for segment in range(segments)]
    faces += [(1, index(rings - 1, segment + 1), index(rings - 1, segment))
              for segment in range(segments)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _part(add, material, parent, apply_scale=False, **placement):
    """Add a primitive, place it in target-body coordinates, and parent it."""
    add(**placement)
    obj = bpy.context.object
    if apply_scale:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent = parent
    obj.data.materials.append(material)
    return obj


def _use_device(scene, backend):
    """Render on every device of `backend`: CPU, a GPU API such as CUDA, or e.g. CUDA+CPU.

    Returns the names of the devices in use.
    """
    if backend == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    kinds = backend.split("+")
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = kinds[0]
    preferences.refresh_devices()
    names = []
    for device in preferences.devices:
        device.use = device.type in kinds
        if device.use:
            names.append(device.name)
    if not any(device.use and device.type == kinds[0] for device in preferences.devices):
        raise RuntimeError(f"no {kinds[0]} render device is available")
    scene.cycles.device = "GPU"
    return names


def _front_plane(name, center, side, material, parent):
    """A -Y normal, preserving texture top-left at body (-X,+Z)."""
    x, y, z = center
    half = side / 2
    vertices = [(x-half, y, z+half), (x-half, y, z-half),
                (x+half, y, z-half), (x+half, y, z+half)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="marker_uv")
    for loop, coordinate in zip(mesh.polygons[0].loop_indices,
                                ((0, 1), (0, 0), (1, 0), (1, 1))):
        uv.data[loop].uv = coordinate
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.data.materials.append(material)
    return obj


class SceneRenderer:
    def __init__(self, request):
        camera = request["camera"]
        self.width, self.height = int(camera["width"]), int(camera["height"])
        hfov = float(camera["hfov_deg"])
        if self.width <= 0 or self.height <= 0 or not 0 < hfov < 180:
            raise ValueError("invalid camera dimensions or field of view")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        self.scene = scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 16
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 6
        scene.render.use_persistent_data = True
        scene.render.resolution_x = self.width
        scene.render.resolution_y = self.height
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = 1
        scene.render.pixel_aspect_y = 1
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.use_file_extension = False
        scene.render.film_transparent = False
        if hasattr(scene.render, "use_motion_blur"):
            scene.render.use_motion_blur = False
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1

        world = bpy.data.worlds.new("fixed_ambient")
        world.use_nodes = True
        self.background = world.node_tree.nodes["Background"]
        self.background.inputs["Color"].default_value = (1, 1, 1, 1)
        self.background.inputs["Strength"].default_value = 0.02
        scene.world = world

        camera_data = bpy.data.cameras.new("calibrated_camera")
        camera_data.type = "PERSP"
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.sensor_width = 36
        camera_data.lens = 36 / (2 * math.tan(math.radians(hfov) / 2))
        camera_data.clip_start = 0.01
        camera_data.clip_end = 1000
        camera_data.dof.use_dof = False
        intrinsic = camera.get("K")
        if intrinsic is not None:
            camera_data.shift_x = ((self.width-1)/2 - float(intrinsic[0][2])) / self.width
            camera_data.shift_y = (float(intrinsic[1][2]) - (self.height-1)/2) / self.width
        self.camera = bpy.data.objects.new("calibrated_camera", camera_data)
        scene.collection.objects.link(self.camera)
        self.camera.matrix_world = Matrix(((1, 0, 0, 0), (0, -1, 0, 0),
                                          (0, 0, -1, 0), (0, 0, 0, 1)))
        scene.camera = self.camera

        sun_data = bpy.data.lights.new("fixed_sun", "SUN")
        sun_data.energy = 3
        sun_data.angle = 0
        self.sun = bpy.data.objects.new("fixed_sun", sun_data)
        scene.collection.objects.link(self.sun)

        self.target = bpy.data.objects.new("target", None)
        scene.collection.objects.link(self.target)
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.object
        cube.name = "target_body"
        cube.parent = self.target
        cube.data.materials.append(_material("target_matte", color=(0.35, 0.35, 0.35)))
        self.target_objects = [cube]
        self.marker_objects = []
        self.marker_materials = []
        white = _material("quiet_zone", color=(1, 1, 1))
        for marker in request["markers"]:
            center = [float(value) for value in marker["center"]]
            side = float(marker["side_m"])
            quiet = float(marker["quiet_zone_m"])
            pad_center = [center[0], center[1] - SURFACE_EPSILON_M, center[2]]
            image_center = [center[0], center[1] - 2 * SURFACE_EPSILON_M, center[2]]
            pad = _front_plane(f"quiet_{marker['id']}", pad_center, side+2*quiet, white, self.target)
            known = _material(f"marker_{marker['id']}", image_path=marker["image_path"])
            unknown = _material(f"unknown_{marker['id']}", image_path=marker["unknown_image_path"])
            tag = _front_plane(f"marker_{marker['id']}", image_center, side, known, self.target)
            self.target_objects.extend((pad, tag))
            self.marker_objects.append(tag)
            self.marker_materials.append((known, unknown))

        mesh = bpy.data.meshes.new("occluder")
        mesh.from_pydata([(-10, -10, 1), (-10, 10, 1), (10, 10, 1), (10, -10, 1)],
                         [], [(0, 1, 2, 3)])
        mesh.update()
        self.occluder = bpy.data.objects.new("occluder", mesh)
        scene.collection.objects.link(self.occluder)
        self.occluder.data.materials.append(_material("occluder_matte", color=(0.3, 0.3, 0.3)))
        self.occluder.hide_render = True
        self.calibration_objects = []
        self.calibration_materials = [
            _material(f"calibration_{index}", color=color, emission=True)
            for index, color in enumerate(CALIBRATION_COLORS)
        ]
        self.ambient, self.earth, self.realistic = 0.02, None, None
        if request.get("scene") is not None:
            self._build_realistic(request["scene"], cube)

    def _build_realistic(self, spec, cube):
        """Space lighting, a detailed satellite around the marked face, and Earth."""
        if spec.get("name") != "realistic":
            raise ValueError(f"unknown render scene: {spec.get('name')}")
        scene, target = self.scene, self.target
        devices = _use_device(scene, spec["device"])
        scene.cycles.samples = int(spec["samples"])
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.sample_clamp_indirect = 10.0
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        self.camera.data.clip_end = 1e8
        self.sun.data.angle = math.radians(0.53)
        self.ambient = 0.0
        self.realistic = {"scene": "realistic", "device": spec["device"], "devices": devices,
                          "max_bounces": scene.cycles.max_bounces,
                          "sun_angular_diameter_deg": 0.53}

        # The 2 m bus: foil blankets, with white paint on the -Y docking face.
        cube.data.materials.clear()
        cube.data.materials.append(_foil("gold_foil", (1.0, 0.72, 0.3)))
        cube.data.materials.append(_principled("docking_face", (0.75, 0.75, 0.72), roughness=0.6)[0])
        for polygon in cube.data.polygons:
            polygon.material_index = int(polygon.normal.y < -0.9)
        aluminum = _principled("aluminum", (0.9, 0.9, 0.92), metallic=1, roughness=0.35)[0]
        handrail = _principled("handrail", (0.9, 0.65, 0.25), metallic=1, roughness=0.35)[0]
        dark = _principled("hatch", (0.04, 0.04, 0.045), roughness=0.5)[0]
        glass = _principled("reflector_glass", (0.02, 0.02, 0.025), roughness=0.04, specular=1)[0]
        white = _principled("white_paint", (0.8, 0.8, 0.78), roughness=0.5)[0]
        cylinder, box = bpy.ops.mesh.primitive_cylinder_add, bpy.ops.mesh.primitive_cube_add
        along_y = (math.pi / 2, 0, 0)
        parts = [
            _part(cylinder, aluminum, target, vertices=64, radius=0.17, depth=0.06,
                  location=(0, -1.03, 0), rotation=along_y),
            _part(cylinder, dark, target, vertices=64, radius=0.12, depth=0.004,
                  location=(0, -1.061, 0), rotation=along_y),
        ]
        for marker in self.marker_objects:
            pad_y = sum(vertex.co.y for vertex in marker.data.vertices) / 4
            if pad_y < -1.01:
                center = [sum(vertex.co[axis] for vertex in marker.data.vertices) / 4
                          for axis in (0, 2)]
                depth = -1 - pad_y
                for dx in (-0.3, 0.3):
                    for dz in (-0.3, 0.3):
                        parts.append(_part(cylinder, aluminum, target, vertices=12, radius=0.012,
                                           depth=depth, rotation=along_y, location=(
                                               center[0] + dx, -1 - depth / 2, center[1] + dz)))
        standoff, radius = spec["reflector_standoff_m"], spec["reflector_radius_m"]
        for x, y, z in spec["reflectors_body_m"]:
            parts.append(_part(cylinder, glass, target, vertices=32, radius=radius,
                               depth=2 * standoff, location=(x, y - standoff, z),
                               rotation=along_y))
        for x in (-0.95, 0.95):
            parts.append(_part(cylinder, handrail, target, vertices=12, radius=0.017, depth=1.6,
                               location=(x, -1.07, 0)))
            for z in (-0.75, 0.75):
                parts.append(_part(cylinder, handrail, target, vertices=12, radius=0.012,
                                   depth=0.07, location=(x, -1.035, z), rotation=along_y))
        cells = _solar_cells()
        for side in (-1, 1):
            parts.append(_part(cylinder, aluminum, target, vertices=16, radius=0.04, depth=0.5,
                               location=(side * 1.25, 0, 0), rotation=(0, math.pi / 2, 0)))
            parts.append(_part(box, cells, target, apply_scale=True, size=1,
                               location=(side * 3.5, 0, 0), scale=(4.0, 0.03, 1.8)))
        parts.append(_part(cylinder, aluminum, target, vertices=16, radius=0.03, depth=0.5,
                           location=(0.4, 0.4, 1.25)))
        dish = _part(bpy.ops.mesh.primitive_uv_sphere_add, white, target, segments=48,
                     ring_count=24, radius=0.4, location=(0.4, 0.4, 1.55), scale=(1, 1, 0.2))
        self.target_objects.extend((*parts, dish))

        self.earth = _sphere("earth", float(spec["earth_radius_m"]), 512, 256)
        self.earth.data.materials.append(_earth_surface())
        self.earth.hide_render = True

    def metadata(self):
        return {
            "blender_version": bpy.app.version_string,
            "engine": self.scene.render.engine,
            "samples": self.scene.cycles.samples,
            "threads": self.scene.render.threads,
            "seed": self.scene.cycles.seed,
            "view_transform": self.scene.view_settings.view_transform,
            "look": self.scene.view_settings.look,
            "exposure": self.scene.view_settings.exposure,
            "surface_epsilon_m": SURFACE_EPSILON_M,
            **(self.realistic or {}),
        }

    def render(self, request):
        variant = request.get("variant", "target")
        if variant not in ("target", "empty", "occluded", "out_of_view", "unknown_ids"):
            raise ValueError(f"unknown render variant: {variant}")
        rotation = Matrix(request["rotation_ct"])
        translation = Vector(request["translation_ct"])
        if len(rotation) != 3 or len(rotation[0]) != 3 or len(translation) != 3:
            raise ValueError("target pose requires a 3x3 rotation and 3-vector translation")
        if not all(math.isfinite(value) for row in rotation for value in row):
            raise ValueError("target rotation must be finite")
        if not all(math.isfinite(value) for value in translation):
            raise ValueError("target translation must be finite")
        if variant == "out_of_view":
            translation.x += 4 * abs(translation.z) + 10
        transform = rotation.to_4x4()
        transform.translation = translation
        self.target.matrix_world = transform
        ray = Vector(request.get("light_direction_c", (0, 0, 1)))
        if len(ray) != 3 or not all(math.isfinite(value) for value in ray) or ray.length == 0:
            raise ValueError("Sun incident-ray direction must be a finite nonzero 3-vector")
        self.sun.rotation_mode = "QUATERNION"
        self.sun.rotation_quaternion = ray.normalized().to_track_quat("-Z", "Y")
        for tag, materials in zip(self.marker_objects, self.marker_materials):
            tag.data.materials[0] = materials[variant == "unknown_ids"]

        for obj in self.calibration_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self.calibration_objects = []
        points = request.get("calibration_points")
        calibration = points is not None
        for obj in self.target_objects:
            obj.hide_render = calibration or variant == "empty"
        self.occluder.hide_render = calibration or variant != "occluded"
        self.sun.hide_render = calibration
        self.background.inputs["Strength"].default_value = 0 if calibration else self.ambient
        if self.earth is not None:
            earth = request.get("earth_translation_ce")
            if earth is not None:
                transform = Matrix(request["earth_rotation_ce"]).to_4x4()
                transform.translation = Vector(earth)
                if not all(math.isfinite(value) for row in transform for value in row):
                    raise ValueError("Earth pose must be finite")
                self.earth.matrix_world = transform
            self.earth.hide_render = calibration or earth is None
        if calibration:
            if not 1 <= len(points) <= len(CALIBRATION_COLORS):
                raise ValueError("calibration fixture requires one to eight points")
            for index, point in enumerate(points):
                position = Vector(point)
                if len(position) != 3 or not all(math.isfinite(value) for value in position):
                    raise ValueError("calibration points must be finite 3-vectors")
                bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.01)
                obj = bpy.context.object
                obj.name = f"calibration_point_{index}"
                obj.parent = self.target
                obj.location = position
                obj.data.materials.append(self.calibration_materials[index])
                self.calibration_objects.append(obj)

        bpy.context.view_layer.update()
        output = Path(request["output_path"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.scene.render.filepath = str(output)
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        result = {**self.metadata(), "render_seconds": time.perf_counter()-started,
                  "output_path": str(output), "variant": variant,
                  "rendered_translation_ct": list(translation)}
        if calibration:
            projections = []
            for point in points:
                world_point = self.target.matrix_world @ Vector(point)
                ndc = world_to_camera_view(self.scene, self.camera, world_point)
                projections.append([self.width*ndc.x-0.5, self.height*(1-ndc.y)-0.5])
            result["projected_pixels"] = projections
            result["blender_projections_px"] = projections
            result["calibration_colors"] = [list(color) for color in CALIBRATION_COLORS[:len(points)]]
        return result


def main():
    renderer = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "init":
                if renderer is not None:
                    raise ValueError("worker is already initialized")
                renderer = SceneRenderer(request)
                result = renderer.metadata()
            elif operation == "render":
                if renderer is None:
                    raise ValueError("worker must be initialized before rendering")
                result = renderer.render(request)
            elif operation == "quit":
                print(PREFIX + json.dumps({"ok": True}), flush=True)
                break
            else:
                raise ValueError(f"unknown worker operation: {operation}")
            print(PREFIX + json.dumps({"ok": True, **result}, allow_nan=False), flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(PREFIX + json.dumps({"ok": False, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
