"""Blender process for the presentation-only showcase movie.

Reuses the calibrated worker's realistic scene and JSON-lines protocol unchanged, keeps the
target's marked docking face, and replaces everything else that is seen from outside: the
target's bus and wings, a chaser, a NASA-textured Earth with atmosphere and city lights, and
a star field. Run with Blender's bundled Python.
"""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blender_worker  # noqa: E402
from blender_worker import _mix, _noise, _part, _principled, _ramp, _sphere  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402


ATMOSPHERE_M = 60_000.0
AIR_GLOW = 0.6
# A camera exposed for sunlit spacecraft would not record city lights; this keeps them faint.
CITY_LIGHTS = 0.05
cylinder, box, cone = (bpy.ops.mesh.primitive_cylinder_add, bpy.ops.mesh.primitive_cube_add,
                       bpy.ops.mesh.primitive_cone_add)
ALONG_X, ALONG_Y = (0, math.pi / 2, 0), (math.pi / 2, 0, 0)


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


def _aim(direction, track):
    """Rotation turning local `track` ('Z' or '-Z') onto `direction`."""
    direction = Vector(direction).normalized()
    return direction.to_track_quat(track, "Y" if abs(direction.y) < 0.9 else "X")


def _xyz(tree, vector):
    node = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(vector, node.inputs[0])
    return node.outputs


def _unit(tree, vector):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = "NORMALIZE"
    tree.links.new(vector, node.inputs[0])
    return node.outputs["Vector"]


def _object_coordinates(tree):
    return tree.nodes.new("ShaderNodeTexCoord").outputs["Object"]


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


def _vector_input(tree, name):
    """A per-render vector set through the node's X, Y and Z inputs."""
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    node.name = name
    return node


def _dot(tree, a, b):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = "DOT_PRODUCT"
    tree.links.new(a, node.inputs[0])
    tree.links.new(b, node.inputs[1])
    return node.outputs["Value"]


def _bump(tree, height, strength, distance=1.0):
    node = tree.nodes.new("ShaderNodeBump")
    node.inputs["Strength"].default_value = strength
    node.inputs["Distance"].default_value = distance
    tree.links.new(height, node.inputs["Height"])
    return node.outputs["Normal"]


def _mli(name, color, half_size, crinkle=12):
    """Crinkled multilayer insulation with taped edges and a grid of tack points.

    Expects a blanket plane whose object x and y span +/- `half_size`.
    """
    material, tree, shader = _principled(name, color, metallic=0.9, roughness=0.3)
    coordinates = _object_coordinates(tree)
    x, y, _ = _xyz(tree, coordinates)
    edge = _math(tree, "MAXIMUM", _math(tree, "ABSOLUTE", x), _math(tree, "ABSOLUTE", y))
    tape = _math(tree, "GREATER_THAN", edge, half_size - 0.035)
    tacks = tree.nodes.new("ShaderNodeTexVoronoi")
    tacks.inputs["Scale"].default_value = 6
    tacks.inputs["Randomness"].default_value = 0
    tree.links.new(coordinates, tacks.inputs["Vector"])
    tack = _math(tree, "LESS_THAN", tacks.outputs["Distance"], 0.04)
    # Crumpled foil: flat facets meeting at creases, on irregular (noise-warped) cells.
    warp = tree.nodes.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 8
    tree.links.new(coordinates, warp.inputs["Vector"])
    warped = tree.nodes.new("ShaderNodeVectorMath")
    warped.operation = "MULTIPLY_ADD"
    tree.links.new(warp.outputs["Color"], warped.inputs[0])
    warped.inputs[1].default_value = (0.06, 0.06, 0.06)
    tree.links.new(coordinates, warped.inputs[2])
    facets = tree.nodes.new("ShaderNodeTexVoronoi")
    facets.feature = "DISTANCE_TO_EDGE"
    facets.inputs["Scale"].default_value = crinkle
    tree.links.new(warped.outputs["Vector"], facets.inputs["Vector"])
    folds = _noise(tree, coordinates, 4, 3)
    wrinkle = _mix(tree, "VALUE", _math(tree, "ADD", tape, tack), _math(
        tree, "ADD", facets.outputs["Distance"], _math(tree, "MULTIPLY", folds, 2.0)), 0.0)
    tree.links.new(_bump(tree, wrinkle, 0.55, 0.02), shader.inputs["Normal"])
    shade = _ramp(tree, _noise(tree, coordinates, 5, 3), 0.3, 0.75)
    film = _mix(tree, "RGBA", shade, tuple(0.6 * value for value in color), color)
    plain = _mix(tree, "RGBA", tape, film, (0.75, 0.75, 0.76))
    tree.links.new(_mix(tree, "RGBA", tack, plain, (0.05, 0.05, 0.05)), shader.inputs["Base Color"])
    rough = _mix(tree, "VALUE", shade, 0.22, 0.45)
    tree.links.new(_mix(tree, "VALUE", tape, rough, 0.3), shader.inputs["Roughness"])
    return material


def _white_paint(name, grid=None):
    """Matte white paint, optionally with panel seams every `grid` metres."""
    material, tree, shader = _principled(name, (0.82, 0.82, 0.8), roughness=0.55)
    coordinates = _object_coordinates(tree)
    tree.links.new(_ramp(tree, _noise(tree, coordinates, 4, 3), 0.2, 0.8),
                   shader.inputs["Coat Weight"])
    shader.inputs["Coat Roughness"].default_value = 0.35
    if grid is not None:
        seams = tree.nodes.new("ShaderNodeTexBrick")
        seams.offset = 0
        for key, value in (("Mortar Size", 0.004), ("Scale", 1 / grid), ("Brick Width", 1.0),
                           ("Row Height", 1.0), ("Color1", (0.83, 0.83, 0.81, 1)),
                           ("Color2", (0.78, 0.78, 0.76, 1)), ("Mortar", (0.3, 0.3, 0.3, 1))):
            seams.inputs[key].default_value = value
        tree.links.new(coordinates, seams.inputs["Vector"])
        tree.links.new(seams.outputs["Color"], shader.inputs["Base Color"])
    return material


def _radiator():
    """Optical solar reflector tiles: second-surface mirrors on a dark bond line."""
    material, tree, shader = _principled("osr_radiator", (0.9, 0.9, 0.92), metallic=1,
                                         roughness=0.06)
    tiles = tree.nodes.new("ShaderNodeTexBrick")
    tiles.offset = 0
    for key, value in (("Mortar Size", 0.03), ("Scale", 25.0), ("Brick Width", 1.0),
                       ("Row Height", 1.0), ("Color1", (0.92, 0.92, 0.94, 1)),
                       ("Color2", (0.8, 0.82, 0.86, 1)), ("Mortar", (0.05, 0.05, 0.05, 1))):
        tiles.inputs[key].default_value = value
    tree.links.new(_object_coordinates(tree), tiles.inputs["Vector"])
    tree.links.new(tiles.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(_mix(tree, "VALUE", tiles.outputs["Fac"], 0.06, 0.7), shader.inputs["Roughness"])
    return material


def _solar_cells(name, half_x, half_z):
    """Coverglass triple-junction cells on a panel whose object x-z face spans +/- half sizes."""
    material, tree, shader = _principled(name, (0.02, 0.025, 0.07), roughness=0.08, specular=0.7)
    shader.inputs["Thin Film Thickness"].default_value = 420
    shader.inputs["Thin Film IOR"].default_value = 1.9
    coordinates = _object_coordinates(tree)
    x, _, z = _xyz(tree, coordinates)
    cells = tree.nodes.new("ShaderNodeTexBrick")
    cells.offset = 0
    for key, value in (("Mortar Size", 0.0015), ("Scale", 1.0), ("Brick Width", 0.08),
                       ("Row Height", 0.04), ("Color1", (0.01, 0.012, 0.04, 1)),
                       ("Color2", (0.015, 0.016, 0.05, 1)), ("Mortar", (0.25, 0.25, 0.25, 1)),
                       ("Bias", 0.0)):
        cells.inputs[key].default_value = value
    plane = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(x, plane.inputs[0])
    tree.links.new(z, plane.inputs[1])
    tree.links.new(plane.outputs[0], cells.inputs["Vector"])
    border = _math(tree, "MAXIMUM",
                   _math(tree, "GREATER_THAN", _math(tree, "ABSOLUTE", x), half_x - 0.025),
                   _math(tree, "GREATER_THAN", _math(tree, "ABSOLUTE", z), half_z - 0.025))
    tree.links.new(_mix(tree, "RGBA", border, cells.outputs["Color"], (0.3, 0.3, 0.3)),
                   shader.inputs["Base Color"])
    grid = _math(tree, "MAXIMUM", cells.outputs["Fac"], border)
    tree.links.new(_mix(tree, "VALUE", grid, 0.0, 0.7), shader.inputs["Metallic"])
    tree.links.new(_mix(tree, "VALUE", grid, 0.06, 0.45), shader.inputs["Roughness"])
    return material


def _panel_back():
    material, tree, shader = _principled("panel_back", (0.04, 0.04, 0.045), roughness=0.5)
    weave = tree.nodes.new("ShaderNodeTexWave")
    weave.inputs["Scale"].default_value = 80
    tree.links.new(_object_coordinates(tree), weave.inputs["Vector"])
    tree.links.new(_bump(tree, weave.outputs["Fac"], 0.08, 0.002), shader.inputs["Normal"])
    return material


def _metal(name, color, roughness):
    material, tree, shader = _principled(name, color, metallic=1, roughness=roughness)
    tree.links.new(_bump(tree, _noise(tree, _object_coordinates(tree), 300, 2), 0.04, 0.001),
                   shader.inputs["Normal"])
    return material


def _blanket(material, parent, normal_axis, sign, half_size, offset, subdivisions):
    """A wrinkled MLI blanket over one face of a cube of half-width `half_size`."""
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdivisions, y_subdivisions=subdivisions,
                                    size=2 * half_size)
    obj = bpy.context.object
    rotation = {("x", 1): (0, math.pi / 2, 0), ("x", -1): (0, -math.pi / 2, 0),
                ("y", 1): (-math.pi / 2, 0, 0), ("y", -1): (math.pi / 2, 0, 0),
                ("z", 1): (0, 0, 0), ("z", -1): (math.pi, 0, 0)}[normal_axis, sign]
    obj.rotation_euler = rotation
    location = [0.0, 0.0, 0.0]
    location["xyz".index(normal_axis)] = sign * (half_size + offset)
    obj.location = location
    texture = bpy.data.textures.new(f"wrinkles_{material.name}", "CLOUDS")
    texture.noise_scale = 0.15 * half_size
    texture.noise_depth = 3
    displace = obj.modifiers.new("wrinkles", "DISPLACE")
    displace.texture = texture
    displace.texture_coords = "LOCAL"
    displace.mid_level = 0
    displace.strength = 0.02 * half_size
    obj.data.shade_smooth()
    obj.parent = parent
    obj.data.materials.append(material)
    return obj


def _faces(obj, materials, choose):
    """Assign `materials` to `obj` by face normal with `choose(normal) -> index`."""
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = choose(polygon.normal)
    return obj


def _paraboloid(name, radius, focal, material, parent, location, segments=48, rings=16):
    """A dish facing +Z with its vertex at `location`."""
    vertices = [(0.0, 0.0, 0.0)]
    for ring in range(1, rings + 1):
        r = radius * ring / rings
        for segment in range(segments):
            angle = 2 * math.pi * segment / segments
            vertices.append((r * math.cos(angle), r * math.sin(angle), r * r / (4 * focal)))

    def index(ring, segment):
        return 1 + (ring - 1) * segments + segment % segments

    faces = [(0, index(1, s), index(1, s + 1)) for s in range(segments)]
    faces += [(index(r, s), index(r + 1, s), index(r + 1, s + 1), index(r, s + 1))
              for r in range(1, rings) for s in range(segments)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.modifiers.new("shell", "SOLIDIFY").thickness = 0.012
    obj.parent = parent
    obj.data.materials.append(material)
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


def _star_tracker(materials, parent, location, direction):
    housing, baffle = materials
    direction = Vector(direction).normalized()
    turn = _aim(direction, "Z")
    body = _part(box, housing, parent, apply_scale=True, size=1, location=location,
                 scale=(0.11, 0.11, 0.09))
    tube = _part(cylinder, baffle, parent, vertices=32, radius=0.055, depth=0.14,
                 end_fill_type="NOTHING", location=Vector(location) + direction * 0.11)
    for obj in (body, tube):
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = turn
    return [body, tube]


def _pivot(parent):
    """An empty at the body origin; wings parented to it turn about body X with it."""
    obj = bpy.data.objects.new("wing_drive", None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    return obj


def _track_sun(pivot, rotation_cb, toward_sun_c):
    """Turn the wings about body X so the cell (-Y) side faces the Sun."""
    sun = rotation_cb.transposed() @ toward_sun_c
    pivot.rotation_euler = (math.atan2(-sun.z, -sun.y), 0, 0)


def _wing(side, materials, parent, root_x, panel_size, panels, yoke_half_width):
    """A yoke and `panels` hinged panels along side*X, cells on the -Y side at zero drive angle."""
    cells, back, frame, strut = materials
    width, height = panel_size
    parts = [_strut(strut, parent, (side * root_x, 0, z), (side * (root_x + 0.45), 0, 0), 0.018)
             for z in (-yoke_half_width, yoke_half_width)]
    parts.append(_part(box, frame, parent, apply_scale=True, size=1,
                       location=(side * (root_x + 0.46), 0, 0), scale=(0.05, 0.06, 0.12)))
    start = root_x + 0.5
    for index in range(panels):
        center = start + (index + 0.5) * width + index * 0.04
        panel = _part(box, cells, parent, apply_scale=True, size=1,
                      location=(side * center, 0, 0), scale=(width, 0.025, height))
        _faces(panel, (cells, back, frame),
               lambda normal: 0 if normal.y < -0.9 else 1 if normal.y > 0.9 else 2)
        parts.append(panel)
        if index:
            hinge_x = side * (center - width / 2 - 0.02)
            parts += [_part(cylinder, frame, parent, vertices=12, radius=0.015, depth=0.12,
                            location=(hinge_x, 0, z)) for z in (-height / 3, height / 3)]
    return parts


class ShowcaseRenderer(blender_worker.SceneRenderer):
    def __init__(self, request):
        super().__init__(request)
        spec = request["scene"]
        assets = {key: Path(value) for key, value in spec["showcase_assets"].items()}
        scene = self.scene
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_use_gpu = False
        scene.cycles.denoising_quality = "FAST"
        scene.cycles.denoising_prefilter = "FAST"
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.02
        scene.cycles.max_bounces = 6
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 3
        scene.cycles.transparent_max_bounces = 8
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = float(spec["exposure"])
        self.sun.data.energy = 4.0

        keep = set(self.marker_objects) | {
            obj for obj in self.target_objects
            if obj.name.startswith("quiet_") or obj.location.y < -0.99}
        for obj in self.target_objects:
            if obj not in keep:
                bpy.data.objects.remove(obj, do_unlink=True)
        self.target_objects = [obj for obj in self.target_objects if obj in keep]
        self.target_objects += self._build_target()
        self.chaser = bpy.data.objects.new("chaser", None)
        scene.collection.objects.link(self.chaser)
        self._build_chaser()
        self._build_earth(assets, float(spec["earth_radius_m"]))
        self._build_stars(assets, float(spec["star_strength"]))

    def _common(self):
        aluminum = _metal("bare_aluminum", (0.88, 0.88, 0.9), 0.3)
        return {
            "aluminum": aluminum,
            "dark": _principled("black_anodize", (0.02, 0.02, 0.022), roughness=0.45)[0],
            "nozzle": _metal("nozzle_niobium", (0.35, 0.33, 0.32), 0.45),
            "glass": _principled("lens_glass", (0.01, 0.01, 0.015), roughness=0.02, specular=1)[0],
            "back": _panel_back(),
            "white": _white_paint("white_paint_seams", grid=0.5),
            "radiator": _radiator(),
        }

    def _build_target(self):
        """The 2 m bus behind the kept docking face: blankets, radiators, avionics, wings, dish."""
        target, m = self.target, self._common()
        gold = _mli("gold_kapton_mli", (1.0, 0.62, 0.16), 1.0)
        structure = _part(box, m["white"], target, size=1.99)
        _faces(structure, (m["white"], m["aluminum"]), lambda normal: int(normal.y > -0.9))
        bevel = structure.modifiers.new("edges", "BEVEL")
        bevel.width, bevel.segments = 0.015, 2
        parts = [structure]
        for axis, sign in (("x", 1), ("x", -1), ("y", 1), ("z", 1), ("z", -1)):
            parts.append(_blanket(gold, target, axis, sign, 1.0, 0.004, 120))
        parts.append(_part(box, m["white"], target, apply_scale=True, size=1,
                           location=(-0.35, 0.15, 1.035), scale=(1.0, 1.4, 0.012)))
        for x in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                parts.append(_part(box, m["aluminum"], target, apply_scale=True, size=1,
                                   location=(x, 0, z), scale=(0.05, 2.04, 0.05)))
        nozzles = (m["nozzle"], m["aluminum"])
        for x in (-1, 1):
            for z in (-1, 1):
                parts += _thruster(nozzles, target, (0.85 * x, 1.03, 0.85 * z), (0, 1, 0))
                parts += _thruster(nozzles, target, (1.03 * x, 0.85, 0.85 * z), (x, 0, 0), 0.8)
        trackers = (m["aluminum"], m["dark"])
        parts += _star_tracker(trackers, target, (-0.6, 0.75, 1.06), (-0.3, 0.4, 1))
        parts += _star_tracker(trackers, target, (-0.75, 0.55, 1.06), (-0.6, 0.2, 1))
        for location, scale in (((0.6, 1.03, -0.5), (0.18, 0.02, 0.18)),
                                ((0.5, -0.4, -1.035), (0.2, 0.2, 0.02))):
            parts.append(_part(box, m["white"], target, apply_scale=True, size=1,
                               location=location, scale=scale))
        for location in ((0.9, 0.9, 1.04), (-0.9, -0.9, 1.04), (0.9, -0.9, -1.04)):
            parts.append(_part(box, m["glass"], target, apply_scale=True, size=1,
                               location=location, scale=(0.05, 0.05, 0.02)))
        panel = (2.0, 1.8)
        cells = _solar_cells("target_cells", panel[0] / 2, panel[1] / 2)
        self.target_wings = _pivot(target)
        for side in (-1, 1):
            parts += _wing(side, (cells, m["back"], m["aluminum"], m["aluminum"]),
                           self.target_wings, 1.0, panel, 2, 0.3)
        parts.append(_part(cylinder, m["aluminum"], target, vertices=24, radius=0.035, depth=0.42,
                           location=(0.4, 0.4, 1.25)))
        parts.append(_part(box, m["dark"], target, apply_scale=True, size=1,
                           location=(0.4, 0.4, 1.48), scale=(0.14, 0.14, 0.1)))
        dish = _principled("dish_paint", (0.85, 0.85, 0.83), roughness=0.5)[0]
        parts.append(_paraboloid("dish", 0.4, 0.3, dish, target, (0.4, 0.4, 1.55)))
        for angle in (0, 2 * math.pi / 3, 4 * math.pi / 3):
            rim = (0.4 + 0.38 * math.cos(angle), 0.4 + 0.38 * math.sin(angle), 1.55 + 0.12)
            parts.append(_strut(m["aluminum"], target, rim, (0.4, 0.4, 1.83), 0.006))
        parts.append(_part(cylinder, m["dark"], target, vertices=24, radius=0.03, depth=0.06,
                           location=(0.4, 0.4, 1.84)))
        return parts

    def _build_chaser(self):
        """The 1 m servicer: silver blankets, white docking face with ring, camera and lidar."""
        chaser, m = self.chaser, self._common()
        silver = _mli("silver_mli", (0.82, 0.82, 0.85), 0.5, crinkle=18)
        structure = _part(box, m["white"], chaser, size=0.99)
        _faces(structure, (m["white"], m["aluminum"]), lambda normal: int(normal.y < 0.9))
        bevel = structure.modifiers.new("edges", "BEVEL")
        bevel.width, bevel.segments = 0.01, 2
        for axis, sign in (("x", 1), ("x", -1), ("y", -1), ("z", -1)):
            _blanket(silver, chaser, axis, sign, 0.5, 0.003, 70)
        _part(box, m["radiator"], chaser, apply_scale=True, size=1,
              location=(0, 0, 0.505), scale=(0.9, 0.9, 0.008))
        for x in (-1, 1):
            for z in (-1, 1):
                _part(box, m["aluminum"], chaser, apply_scale=True, size=1,
                      location=(0.5 * x, 0, 0.5 * z), scale=(0.035, 1.02, 0.035))
                for direction in ((x, 0, 0), (0, 0, z), (0, -1, 0)):
                    _thruster((m["nozzle"], m["aluminum"]), chaser,
                              (0.52 * x, -0.38, 0.52 * z), direction, 0.6)
        _part(cylinder, m["aluminum"], chaser, vertices=64, radius=0.22, depth=0.05,
              location=(0, 0.525, 0), rotation=ALONG_Y)
        _part(cylinder, m["dark"], chaser, vertices=64, radius=0.15, depth=0.004,
              location=(0, 0.551, 0), rotation=ALONG_Y)
        for angle in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3):
            _strut(m["aluminum"], chaser, (0.2 * math.cos(angle), 0.55, 0.2 * math.sin(angle)),
                   (0.13 * math.cos(angle), 0.66, 0.13 * math.sin(angle)), 0.012)
        for x, radius in ((0.2, 0.03), (-0.2, 0.035)):
            _part(box, m["dark"], chaser, apply_scale=True, size=1,
                  location=(x, 0.53, 0.2), scale=(0.1, 0.06, 0.1))
            _part(cylinder, m["dark"], chaser, vertices=24, radius=0.04, depth=0.05,
                  end_fill_type="NOTHING", location=(x, 0.575, 0.2), rotation=ALONG_Y)
            _part(cylinder, m["glass"], chaser, vertices=24, radius=radius, depth=0.01,
                  location=(x, 0.565, 0.2), rotation=ALONG_Y)
        _star_tracker((m["aluminum"], m["dark"]), chaser, (-0.3, -0.25, 0.53), (-0.3, -0.3, 1))
        _part(cylinder, m["white"], chaser, vertices=24, radius=0.05, depth=0.02,
              location=(0.3, -0.3, 0.52))
        panel = (1.6, 0.8)
        cells = _solar_cells("chaser_cells", panel[0] / 2, panel[1] / 2)
        self.chaser_wings = _pivot(chaser)
        for side in (-1, 1):
            _wing(side, (cells, m["back"], m["aluminum"], m["aluminum"]), self.chaser_wings, 0.5,
                  panel, 1, 0.15)

    def _build_earth(self, assets, radius):
        """NASA Blue Marble day side, Black Marble city lights, clouds, and an atmosphere shell.

        The sun-facing nodes are set per render.
        """
        material, tree, shader = _principled("earth_nasa", (0, 0, 0))
        uv = _equirectangular(tree, _object_coordinates(tree), 1.0)
        day = _image(tree, assets["day"], uv)["Color"]
        cloud = _image(tree, assets["clouds"], uv, "Non-Color")["Color"]
        night = _image(tree, assets["night"], uv)["Color"]
        rgb = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(day, rgb.inputs[0])
        water = _ramp(tree, _math(tree, "SUBTRACT", rgb.outputs[2], rgb.outputs[0]), 0.03, 0.12)
        ocean = _mix(tree, "RGBA", _ramp(tree, rgb.outputs[2], 0.1, 0.6),
                     (0.004, 0.012, 0.035), (0.01, 0.03, 0.07))
        ground = _mix(tree, "RGBA", water, day, ocean)
        cover = _ramp(tree, cloud, 0.15, 0.85)
        tree.links.new(_mix(tree, "RGBA", cover, ground, (0.85, 0.86, 0.88)), shader.inputs["Base Color"])
        tree.links.new(_mix(tree, "VALUE", cover, _mix(tree, "VALUE", water, 0.85, 0.22), 0.9),
                       shader.inputs["Roughness"])
        self.earth_sun = _vector_input(tree, "toward_sun")
        dark = _ramp(tree, _dot(tree, _unit(tree, _object_coordinates(tree)),
                                self.earth_sun.outputs[0]), 0.05, -0.12)
        lights = _math(tree, "MULTIPLY", dark, _math(tree, "SUBTRACT", 1.0,
                                                     _math(tree, "MULTIPLY", cover, 0.85)))
        tree.links.new(night, shader.inputs["Emission Color"])
        tree.links.new(_math(tree, "MULTIPLY", lights, CITY_LIGHTS), shader.inputs["Emission Strength"])
        self.earth.data.materials.clear()
        self.earth.data.materials.append(material)

        self.atmosphere = _sphere("atmosphere", radius + ATMOSPHERE_M, 512, 256)
        self.atmosphere.parent = self.earth
        for key in ("visible_diffuse", "visible_glossy", "visible_transmission",
                    "visible_volume_scatter", "visible_shadow"):
            setattr(self.atmosphere, key, False)
        air = self._air(radius)
        self.atmosphere.data.materials.append(air)
        for material in (material, air):
            material.cycles.emission_sampling = "NONE"

    def _air(self, radius):
        """Transparent shell whose front face emits the single-scattered skylight of its ray.

        In Earth radii, a ray entering at P along d crosses the air for
        L = s_out - s_in if it reaches the ground, else 2 s_out, where s_out and s_in are the
        half-chords of the shell and Earth at the ray's closest approach b. Glow grows as
        (L / L_max)^2, so the horizon is brightest and the view straight down stays clear. The
        colour follows the Sun's elevation at the path midpoint: blue by day, red at twilight.
        """
        air = bpy.data.materials.new("atmosphere_glow")
        air.use_nodes = True
        tree = air.node_tree
        tree.nodes.clear()
        top = (1 + ATMOSPHERE_M / radius) ** 2

        def vector(operation, a, b=None, scale=None):
            node = tree.nodes.new("ShaderNodeVectorMath")
            node.operation = operation
            tree.links.new(a, node.inputs[0])
            if b is not None:
                tree.links.new(b, node.inputs[1])
            if scale is not None:
                node.inputs["Scale"].default_value = scale
            return node.outputs["Vector"]

        def root(value):
            return _math(tree, "SQRT", _math(tree, "MAXIMUM", value, 0.0))

        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        point = vector("SCALE", _object_coordinates(tree), scale=1 / radius)
        incoming = tree.nodes.new("ShaderNodeVectorTransform")
        incoming.vector_type, incoming.convert_from, incoming.convert_to = "VECTOR", "WORLD", "OBJECT"
        tree.links.new(geometry.outputs["Incoming"], incoming.inputs["Vector"])
        ray = _unit(tree, vector("SCALE", incoming.outputs["Vector"], scale=-1.0))
        along = _dot(tree, point, ray)
        closest = _math(tree, "SUBTRACT", _dot(tree, point, point),
                        _math(tree, "MULTIPLY", along, along))
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
        self.air_sun = _vector_input(tree, "toward_sun")
        light = _dot(tree, _unit(tree, middle.outputs["Vector"]), self.air_sun.outputs[0])
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

    def _build_stars(self, assets, strength):
        """Deep Star Maps 2020 in celestial coordinates, seen only by camera rays.

        The camera-to-inertial rotation rows are set per render.
        """
        tree = self.scene.world.node_tree
        direction = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
        self.star_rows = [_vector_input(tree, f"inertial_row_{axis}") for axis in range(3)]
        inertial = tree.nodes.new("ShaderNodeCombineXYZ")
        for axis, row in enumerate(self.star_rows):
            tree.links.new(_dot(tree, row.outputs[0], direction), inertial.inputs[axis])
        stars = _image(tree, assets["stars"], _equirectangular(tree, inertial.outputs[0], -1.0),
                       "Linear Rec.709")["Color"]
        self.scene.world.cycles.sampling_method = "NONE"
        camera_ray = tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
        tree.links.new(stars, self.background.inputs["Color"])
        tree.links.new(_math(tree, "MULTIPLY", camera_ray, strength),
                       self.background.inputs["Strength"])

    def render(self, request):
        transform = Matrix(request["chaser_rotation_cb"]).to_4x4()
        transform.translation = Vector(request["chaser_translation_cb"])
        if not all(math.isfinite(value) for row in transform for value in row):
            raise ValueError("chaser pose requires a finite 3x3 rotation and 3-vector")
        self.chaser.matrix_world = transform
        toward_sun = -Vector(request["light_direction_c"]).normalized()
        _track_sun(self.target_wings, Matrix(request["rotation_ct"]), toward_sun)
        _track_sun(self.chaser_wings, Matrix(request["chaser_rotation_cb"]), toward_sun)
        # Earth and its atmosphere shade in Earth-body axes.
        sun = Matrix(request["earth_rotation_ce"]).transposed() @ toward_sun
        for node in (self.earth_sun, self.air_sun):
            for axis in range(3):
                node.inputs[axis].default_value = sun[axis]
        for row, values in zip(self.star_rows, request["stars_rotation_nc"]):
            for axis in range(3):
                row.inputs[axis].default_value = values[axis]
        return super().render(request)


if __name__ == "__main__":
    # blender_worker.main constructs its module-level SceneRenderer.
    blender_worker.SceneRenderer = ShowcaseRenderer
    blender_worker.main()
