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
ATMOSPHERE_M = 60_000.0
AIR_GLOW = 0.6
STAR_SPHERE_M = 5e7  # beyond Earth's far side, inside the camera's 1e8 m clip
# The target's 2 m body is drawn as a docking plate from y = -1 to PLATE_BACK_Y, in front of
# NASA's QuikSCAT model: its 1.8 m bus scaled to 2 m and moved just behind the plate, with
# its wing hinge axis (model z = 0.7 m) on the drive.
PLATE_BACK_Y = -0.95
TARGET_MODEL_SCALE = 1.1
TARGET_MODEL_OFFSET = (0.0, 0.08, -0.05)
TARGET_WING_AXIS_Z = 0.7
cylinder, box, cone = (bpy.ops.mesh.primitive_cylinder_add, bpy.ops.mesh.primitive_cube_add,
                       bpy.ops.mesh.primitive_cone_add)
ALONG_Y = (math.pi / 2, 0, 0)


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


def _math(tree, operation, *values):
    """Scalar math on sockets or constants."""
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, values):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs[0]


def _xyz(tree, vector):
    node = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(vector, node.inputs[0])
    return node.outputs


def _unit(tree, vector):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = "NORMALIZE"
    tree.links.new(vector, node.inputs[0])
    return node.outputs["Vector"]


def _dot(tree, a, b):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = "DOT_PRODUCT"
    tree.links.new(a, node.inputs[0])
    tree.links.new(b, node.inputs[1])
    return node.outputs["Value"]


def _object_coordinates(tree):
    return tree.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def _bump(tree, height, strength, distance=1.0):
    node = tree.nodes.new("ShaderNodeBump")
    node.inputs["Strength"].default_value = strength
    node.inputs["Distance"].default_value = distance
    tree.links.new(height, node.inputs["Height"])
    return node.outputs["Normal"]


def _image(tree, path, vector, colorspace="sRGB"):
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.colorspace_settings.name = colorspace
    node.interpolation = "Linear"
    node.extension = "REPEAT"
    tree.links.new(vector, node.inputs["Vector"])
    return node.outputs


def _equirectangular(tree, direction, east_sign):
    """Plate-carree UV of a direction: longitude atan2(y, x) on u, latitude on v."""
    x, y, z = _xyz(tree, _unit(tree, direction))
    u = _math(tree, "ADD", _math(tree, "MULTIPLY", _math(tree, "ARCTAN2", y, x),
                                 east_sign / (2 * math.pi)), 0.5)
    v = _math(tree, "ADD", _math(tree, "MULTIPLY", _math(tree, "ARCSINE", z), 1 / math.pi), 0.5)
    uv = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(u, uv.inputs[0])
    tree.links.new(v, uv.inputs[1])
    return uv.outputs[0]


def _metal(name, color, roughness):
    material, tree, shader = _principled(name, color, metallic=1, roughness=roughness)
    tree.links.new(_bump(tree, _noise(tree, _object_coordinates(tree), 300, 2), 0.04, 0.001),
                   shader.inputs["Normal"])
    return material


def _hardware_materials():
    """Materials shared by both spacecraft models."""
    return {
        "aluminum": _metal("bare_aluminum", (0.88, 0.88, 0.9), 0.3),
        "dark": _principled("black_anodize", (0.02, 0.02, 0.022), roughness=0.45)[0],
        "nozzle": _metal("nozzle_niobium", (0.35, 0.33, 0.32), 0.45),
        "glass": _principled("lens_glass", (0.01, 0.01, 0.015), roughness=0.02, specular=1)[0],
    }


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


def _aim(direction, track):
    """Rotation turning local `track` ('Z' or '-Z') onto `direction`."""
    direction = Vector(direction).normalized()
    return direction.to_track_quat(track, "Y" if abs(direction.y) < 0.9 else "X")


def _import_model(path, parent, scale, rotation, location):
    """Import a glTF model under an empty placed in `parent`'s body axes.

    Returns the imported meshes and the empty.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    holder = bpy.data.objects.new(f"model_{Path(path).stem}", None)
    bpy.context.scene.collection.objects.link(holder)
    holder.parent = parent
    holder.scale = (scale, scale, scale)
    holder.rotation_euler = rotation
    holder.location = location
    for obj in imported:
        if obj.parent is None:
            obj.parent = holder
    return [obj for obj in imported if obj.type == "MESH"], holder


def _faces(obj, materials, choose):
    """Assign `materials` to `obj` by face normal with `choose(normal) -> index`."""
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = choose(polygon.normal)
    return obj


def _strut(material, parent, start, end, radius):
    """A cylinder between two body points."""
    start, end = Vector(start), Vector(end)
    axis = end - start
    obj = _part(cylinder, material, parent, vertices=12, radius=radius, depth=axis.length,
                location=(start + end) / 2)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = _aim(axis, "Z")
    return obj


def _thruster(materials, parent, location, direction, size=1.0):
    """A small bell nozzle pointing along `direction` from a valve block."""
    nozzle, valve = materials
    direction = Vector(direction).normalized()
    turn = _aim(direction, "-Z")
    block = _part(box, valve, parent, apply_scale=True, size=1, location=location,
                  scale=(0.05 * size, 0.05 * size, 0.04 * size))
    bell = _part(cone, nozzle, parent, vertices=24, radius1=0.028 * size, radius2=0.01 * size,
                 depth=0.06 * size, end_fill_type="NOTHING",
                 location=Vector(location) + direction * 0.045 * size)
    for obj in (block, bell):
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = turn
    return [block, bell]


def _wing_drive(parent):
    """An empty at the body origin; wings parented to it turn about body X with it."""
    obj = bpy.data.objects.new("wing_drive", None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    return obj


def _track_sun(drive, rotation_cb, toward_sun_c):
    """Turn the wings about body X so the cell (-Y) side faces the Sun."""
    sun = rotation_cb.transposed() @ toward_sun_c
    drive.rotation_euler = (math.atan2(-sun.z, -sun.y), 0, 0)


def _earth_surface(assets, city_lights, sun_frame):
    """NASA Blue Marble surface with water darkened, clouds, and optional city lights on the
    night side, judged in `sun_frame`, an Earth-centred frame whose +Z points at the Sun."""
    material, tree, shader = _principled("earth_nasa", (0, 0, 0))
    uv = _equirectangular(tree, _object_coordinates(tree), 1.0)
    day = _image(tree, assets["day"], uv)["Color"]
    cloud = _image(tree, assets["clouds"], uv, "Non-Color")["Color"]
    rgb = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(day, rgb.inputs[0])
    water = _ramp(tree, _math(tree, "SUBTRACT", rgb.outputs[2], rgb.outputs[0]), 0.03, 0.12)
    ocean = _mix(tree, "RGBA", _ramp(tree, rgb.outputs[2], 0.1, 0.6),
                 (0.004, 0.012, 0.035), (0.01, 0.03, 0.07))
    cover = _ramp(tree, cloud, 0.15, 0.85)
    tree.links.new(_mix(tree, "RGBA", cover, _mix(tree, "RGBA", water, day, ocean),
                        (0.85, 0.86, 0.88)), shader.inputs["Base Color"])
    tree.links.new(_mix(tree, "VALUE", cover, _mix(tree, "VALUE", water, 0.85, 0.22), 0.9),
                   shader.inputs["Roughness"])
    if city_lights:
        night = _image(tree, assets["night"], uv)["Color"]
        sun_space = tree.nodes.new("ShaderNodeTexCoord")
        sun_space.object = sun_frame
        dark = _ramp(tree, _xyz(tree, _unit(tree, sun_space.outputs["Object"]))[2], 0.05, -0.12)
        lights = _math(tree, "MULTIPLY", dark, _math(tree, "SUBTRACT", 1.0,
                                                     _math(tree, "MULTIPLY", cover, 0.85)))
        tree.links.new(night, shader.inputs["Emission Color"])
        tree.links.new(_math(tree, "MULTIPLY", lights, city_lights),
                       shader.inputs["Emission Strength"])
    return material


def _air(radius):
    """Transparent shell whose front face emits the single-scattered skylight of its ray.

    In Earth radii, a ray entering at P along d crosses the air for
    L = s_out - s_in if it reaches the ground, else 2 s_out, where s_out and s_in are the
    half-chords of the shell and Earth at the ray's closest approach b. Glow grows as
    (L / L_max)^2, so the horizon is brightest and the view straight down stays clear. The
    colour follows the Sun's elevation at the path midpoint: blue by day, red at twilight.
    The shell's object frame is Earth-centred with +Z toward the Sun.
    """
    air = bpy.data.materials.new("atmosphere_glow")
    air.use_nodes = True
    tree = air.node_tree
    tree.nodes.clear()
    top = (1 + ATMOSPHERE_M / radius) ** 2

    def vector(operation, a, scale):
        node = tree.nodes.new("ShaderNodeVectorMath")
        node.operation = operation
        tree.links.new(a, node.inputs[0])
        node.inputs["Scale"].default_value = scale
        return node.outputs["Vector"]

    def root(value):
        return _math(tree, "SQRT", _math(tree, "MAXIMUM", value, 0.0))

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    point = vector("SCALE", _object_coordinates(tree), 1 / radius)
    incoming = tree.nodes.new("ShaderNodeVectorTransform")
    incoming.vector_type, incoming.convert_from, incoming.convert_to = "VECTOR", "WORLD", "OBJECT"
    tree.links.new(geometry.outputs["Incoming"], incoming.inputs["Vector"])
    ray = _unit(tree, vector("SCALE", incoming.outputs["Vector"], -1.0))
    along = _dot(tree, point, ray)
    closest = _math(tree, "SUBTRACT", _dot(tree, point, point), _math(tree, "MULTIPLY", along, along))
    shell = root(_math(tree, "SUBTRACT", top, closest))
    ground = root(_math(tree, "SUBTRACT", 1.0, closest))
    hits = _math(tree, "LESS_THAN", closest, 1.0)
    length = _math(tree, "SUBTRACT", _math(tree, "MULTIPLY", shell, 2.0), _math(
        tree, "MULTIPLY", hits, _math(tree, "ADD", shell, ground)))
    middle = tree.nodes.new("ShaderNodeVectorMath")
    middle.operation = "MULTIPLY_ADD"
    tree.links.new(ray, middle.inputs[0])
    half = tree.nodes.new("ShaderNodeCombineXYZ")
    for axis in range(3):
        tree.links.new(_math(tree, "MULTIPLY", length, 0.5), half.inputs[axis])
    tree.links.new(half.outputs[0], middle.inputs[1])
    tree.links.new(point, middle.inputs[2])
    light = _xyz(tree, _unit(tree, middle.outputs["Vector"]))[2]
    sky = tree.nodes.new("ShaderNodeValToRGB")
    ramp = sky.color_ramp
    ramp.elements[0].position, ramp.elements[0].color = 0.42, (0, 0, 0, 1)
    ramp.elements[1].position, ramp.elements[1].color = 0.56, (0.3, 0.55, 1.0, 1)
    for position, color in ((0.46, (0.4, 0.06, 0.01, 1)), (0.49, (1.0, 0.42, 0.1, 1)),
                            (0.515, (0.75, 0.7, 0.75, 1))):
        ramp.elements.new(position).color = color
    tree.links.new(_math(tree, "MULTIPLY_ADD", light, 0.5, 0.5), sky.inputs["Fac"])
    fraction = _math(tree, "DIVIDE", length, 2 * math.sqrt(top - 1))
    front = _math(tree, "SUBTRACT", 1.0, geometry.outputs["Backfacing"])
    glow = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(sky.outputs["Color"], glow.inputs["Color"])
    tree.links.new(_math(tree, "MULTIPLY", _math(tree, "MULTIPLY", _math(
        tree, "MULTIPLY", fraction, fraction), front), AIR_GLOW), glow.inputs["Strength"])
    add = tree.nodes.new("ShaderNodeAddShader")
    tree.links.new(tree.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[0])
    tree.links.new(glow.outputs[0], add.inputs[1])
    tree.links.new(add.outputs[0], tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return air


def _camera_only(obj):
    for key in ("visible_diffuse", "visible_glossy", "visible_transmission",
                "visible_volume_scatter", "visible_shadow"):
        setattr(obj, key, False)


def _star_sphere(path, strength):
    """Deep Star Maps 2020 on a sphere around the camera, seen only by camera rays. Each
    render sets its axes to inertial, so its object coordinates are celestial directions."""
    obj = _sphere("stars", STAR_SPHERE_M, 64, 32)
    _camera_only(obj)
    material = bpy.data.materials.new("star_map")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    glow = tree.nodes.new("ShaderNodeEmission")
    glow.inputs["Strength"].default_value = strength
    tree.links.new(_image(tree, path, _equirectangular(tree, _object_coordinates(tree), -1.0),
                          "Linear Rec.709")["Color"], glow.inputs["Color"])
    tree.links.new(glow.outputs[0], tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    material.cycles.emission_sampling = "NONE"
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
        """Space lighting, a detailed satellite around the marked face, NASA Earth and air."""
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

        # The 2 m body becomes a 5 cm docking plate, white on -Y, mounted on a NASA model bus.
        hardware = _hardware_materials()
        aluminum, dark = hardware["aluminum"], hardware["dark"]
        for vertex in cube.data.vertices:
            vertex.co.y = min(vertex.co.y, PLATE_BACK_Y)
        _faces(cube, (_principled("docking_face", (0.75, 0.75, 0.72), roughness=0.6)[0], aluminum),
               lambda normal: int(normal.y > -0.9))
        handrail = _principled("handrail", (0.9, 0.65, 0.25), metallic=1, roughness=0.35)[0]
        glass = _principled("reflector_glass", (0.02, 0.02, 0.025), roughness=0.04, specular=1)[0]
        parts = [
            _part(cylinder, aluminum, target, vertices=64, radius=0.17, depth=0.06,
                  location=(0, -1.03, 0), rotation=ALONG_Y),
            _part(cylinder, dark, target, vertices=64, radius=0.12, depth=0.004,
                  location=(0, -1.061, 0), rotation=ALONG_Y),
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
                                           depth=depth, rotation=ALONG_Y, location=(
                                               center[0] + dx, -1 - depth / 2, center[1] + dz)))
        standoff, radius = spec["reflector_standoff_m"], spec["reflector_radius_m"]
        for x, y, z in spec["reflectors_body_m"]:
            parts.append(_part(cylinder, glass, target, vertices=32, radius=radius,
                               depth=2 * standoff, location=(x, y - standoff, z),
                               rotation=ALONG_Y))
        for x in (-0.95, 0.95):
            parts.append(_part(cylinder, handrail, target, vertices=12, radius=0.017, depth=1.6,
                               location=(x, -1.07, 0)))
            for z in (-0.75, 0.75):
                parts.append(_part(cylinder, handrail, target, vertices=12, radius=0.012,
                                   depth=0.07, location=(x, -1.035, z), rotation=ALONG_Y))
        assets = {key: Path(value) for key, value in spec["assets"].items()}
        self.target_objects.extend((*parts, *self._build_bus(assets["target_model"])))

        # Shaders read the Sun from this Earth-centred frame's +Z, which each render turns by
        # transform: no material changes between renders, which corrupted materials mid-run.
        self.sun_frame = bpy.data.objects.new("sun_frame", None)
        scene.collection.objects.link(self.sun_frame)
        earth_radius = float(spec["earth_radius_m"])
        self.earth = _sphere("earth", earth_radius, 512, 256)
        surface = _earth_surface(assets, float(spec.get("city_lights", 0.0)), self.sun_frame)
        self.earth.data.materials.append(surface)
        self.earth.hide_render = True
        self.atmosphere = _sphere("atmosphere", earth_radius + ATMOSPHERE_M, 512, 256)
        self.atmosphere.parent = self.sun_frame
        _camera_only(self.atmosphere)
        air = _air(earth_radius)
        self.atmosphere.data.materials.append(air)
        for material in (surface, air):
            material.cycles.emission_sampling = "NONE"
        self.stars = None
        if spec.get("star_strength"):
            self.stars = _star_sphere(assets["stars"], float(spec["star_strength"]))

    def _build_bus(self, path):
        """NASA's QuikSCAT model behind the docking plate: its bus scaled to the 2 m body, and
        its solar wings on a drive that tracks the Sun about body X."""
        meshes, holder = _import_model(path, self.target, TARGET_MODEL_SCALE, (0, 0, 0),
                                       TARGET_MODEL_OFFSET)
        self.wings = _wing_drive(holder)
        self.wings.location = (0, 0, TARGET_WING_AXIS_Z)
        bpy.context.view_layer.update()
        for obj in meshes:
            if any(material.name.startswith("QS_solar") for material in obj.data.materials):
                world = obj.matrix_world.copy()
                obj.parent = self.wings
                obj.matrix_world = world
        return meshes

    def _update_realistic(self, request, rotation_ct, ray):
        """Sun-tracking wings, the Sun frame at Earth's centre, and the star sphere's
        inertial axes."""
        toward_sun = -ray.normalized()
        _track_sun(self.wings, rotation_ct, toward_sun)
        frame = toward_sun.to_track_quat("Z", "Y").to_matrix().to_4x4()
        frame.translation = self.earth.matrix_world.translation
        self.sun_frame.matrix_world = frame
        self.atmosphere.hide_render = self.earth.hide_render
        if self.stars is not None:
            self.stars.matrix_world = Matrix(request["stars_rotation_nc"]).transposed().to_4x4()

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
        if self.realistic is not None:
            self._update_realistic(request, rotation, ray)
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
