# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene baking for the Cinema 4D Bridge (.c4dbridge, format version 2).

A .c4dbridge file is a zip with:
    manifest.json               hierarchy, per-frame transforms, cameras, lights, cuts
    mesh/<id>/points.f32        V x 3 float32, already in Cinema 4D axes (x, z, y), Blender units
    mesh/<id>/polys.i32         F x 4 int32 vertex indices, Cinema 4D winding (triangles repeat c)
    mesh/<id>/uv.f32            F x 4 x 2 float32 (u, 1 - v)                      [optional]
    mesh/<id>/normals.i16       F x 4 x 3 int16 (normal * 32000, C4D axes)         [optional]
    mesh/<id>/mats.u16          F uint16 material slot per polygon

Transforms are Blender-space 3x4 matrices relative to the exported parent (or
world), run-length encoded per frame; the importer converts axes.
"""

import hashlib
import json
import os
import zipfile

import bpy
import numpy as np

FORMAT_ID = "camera-bridge"
FORMAT_VERSION = 2
ADDON_VERSION = "2.3.0"
PRECISION = 6

BLOB_EXT = {"points": "f32", "polys": "i32", "uv": "f32", "normals": "i16", "mats": "u16"}
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
UNSUPPORTED_GEOMETRY = {'CURVES', 'POINTCLOUD', 'VOLUME', 'GREASEPENCIL', 'GPENCIL'}


# ---------------------------------------------------------------------------
# Scene helpers shared with the camera exporter
# ---------------------------------------------------------------------------

def frame_size(scene):
    r = scene.render
    return r.resolution_x * r.pixel_aspect_x, r.resolution_y * r.pixel_aspect_y


def camera_cuts(scene):
    """Camera switches exactly as Blender resolves them (BKE_scene_camera_switch_find)."""
    by_frame = {}
    for m in scene.timeline_markers:
        cam = m.camera
        if cam is None or cam.hide_render or m.frame in by_frame:
            continue
        by_frame[m.frame] = cam
    cuts = []
    for frame in sorted(by_frame):
        cam = by_frame[frame]
        if not cuts or cuts[-1][1] != cam:
            cuts.append((frame, cam))
    return cuts


def frame_range(scene, mode):
    if mode == 'PREVIEW' and scene.use_preview_range:
        return scene.frame_preview_start, scene.frame_preview_end
    return scene.frame_start, scene.frame_end


def _r(v):
    return round(float(v), PRECISION)


def pack_channel(values):
    first = values[0]
    if all(v == first for v in values):
        return first
    return values


def rle(values):
    """[[index, value], ...] entries only where the value changes."""
    out = []
    for i, v in enumerate(values):
        if not out or out[-1][1] != v:
            out.append([i, v])
    return out


def matrix_rows(m, precision=8):
    _r = lambda v: round(float(v), precision)
    return [_r(m[0][0]), _r(m[0][1]), _r(m[0][2]), _r(m[0][3]),
            _r(m[1][0]), _r(m[1][1]), _r(m[1][2]), _r(m[1][3]),
            _r(m[2][0]), _r(m[2][1]), _r(m[2][2]), _r(m[2][3])]


def orthonormal(m):
    """Copy of a 4x4 with its 3x3 part normalized (scale stripped)."""
    out = m.to_3x3().normalized().to_4x4()
    out.translation = m.translation
    return out


def is_invertible(m):
    return abs(m.to_3x3().determinant()) > 1e-12


# ---------------------------------------------------------------------------
# Selection of what to export
# ---------------------------------------------------------------------------

def layer_collections(view_layer):
    """(layer_collection, parent_collection_or_None) for every non-excluded collection."""
    out = []

    def walk(lc, parent):
        for child in lc.children:
            if child.exclude:
                continue
            out.append((child, parent))
            walk(child, child.collection)

    walk(view_layer.layer_collection, None)
    return out


def gather_objects(context, mode):
    scene, view_layer = context.scene, context.view_layer
    enabled = {lc.collection for lc, _parent in layer_collections(view_layer)} | {scene.collection}
    available = {ob for ob in view_layer.objects if any(c in enabled for c in ob.users_collection)}
    if mode == 'SELECTED':
        picked = [ob for ob in context.selected_objects if ob in available]
    elif mode == 'CAMERAS':
        picked = []
        for _f, cam in camera_cuts(scene):
            if cam not in picked:
                picked.append(cam)
        if not picked and scene.camera:
            picked = [scene.camera]
        picked = [ob for ob in picked if ob in available]
    else:
        picked = list(available)
    # Always bring ancestors along so the hierarchy (camera rigs, controllers) survives.
    result = set(picked)
    for ob in picked:
        parent = ob.parent
        while parent is not None and parent in available:
            result.add(parent)
            parent = parent.parent
    return sorted(result, key=lambda ob: ob.name)


def object_kind(ob):
    if ob.type == 'CAMERA':
        return "camera"
    if ob.type == 'LIGHT':
        return "light"
    if ob.type in GEOMETRY_TYPES:
        return "mesh"
    return "null"


# ---------------------------------------------------------------------------
# Per-frame sampling
# ---------------------------------------------------------------------------

def sample_camera(ob_eval, depsgraph, world, width, height):
    cam = ob_eval.data
    fit = cam.sensor_fit
    if fit == 'AUTO':
        horizontal, size = width >= height, cam.sensor_width
    elif fit == 'HORIZONTAL':
        horizontal, size = True, cam.sensor_width
    else:
        horizontal, size = False, cam.sensor_height
    aspect = width / height
    gate = size if horizontal else size * aspect
    ortho_width = cam.ortho_scale if horizontal else cam.ortho_scale * aspect
    if horizontal:
        off_x, off_y = cam.shift_x, cam.shift_y * aspect
    else:
        off_x, off_y = cam.shift_x / aspect, cam.shift_y

    dof = cam.dof
    focus = dof.focus_distance
    target = dof.focus_object
    if target is not None:
        target_eval = target.evaluated_get(depsgraph)
        point = target_eval.matrix_world.translation
        sub = getattr(dof, "focus_subtarget", "")
        if sub and target.type == 'ARMATURE' and sub in target_eval.pose.bones:
            point = target_eval.matrix_world @ target_eval.pose.bones[sub].head
        focus = max(-(world.inverted_safe() @ point).z, 0.0)

    return {
        "lens": _r(cam.lens), "gate": _r(gate), "offset_x": _r(off_x), "offset_y": _r(off_y),
        "clip_start": _r(cam.clip_start), "clip_end": _r(cam.clip_end),
        "focus_distance": _r(focus), "fstop": _r(dof.aperture_fstop), "ortho_width": _r(ortho_width),
    }


def sample_light(ob_eval, world_scale):
    light = ob_eval.data
    sx, sy = abs(world_scale[0]), abs(world_scale[1])
    out = {
        "color": [_r(c) for c in light.color],
        "energy": _r(light.energy),
    }
    if light.type == 'AREA':
        size_y = light.size_y if light.shape in {'RECTANGLE', 'ELLIPSE'} else light.size
        out["size_x"] = _r(light.size * sx)
        out["size_y"] = _r(size_y * sy)
    elif light.type == 'SPOT':
        out["spot_size"] = _r(light.spot_size)
        out["spot_blend"] = _r(light.spot_blend)
    return out


class ObjectTrack:
    """Everything recorded per frame for one object."""

    def __init__(self, ob, parent_exported):
        self.ob = ob
        self.kind = object_kind(ob)
        self.parent = ob.parent if parent_exported else None
        constraints = any(c.enabled for c in ob.constraints) if hasattr(ob, "constraints") else False
        self.use_basis = self.parent is not None and ob.parent_type == 'OBJECT' and not constraints
        self.strip_scale = self.kind in {"camera", "light"}
        self.matrices = []
        self.hide_render = []
        self.samples = []
        self.warned_basis = False

    def local_matrix(self, ob_eval, parent_eval):
        world = ob_eval.matrix_world
        if self.strip_scale:
            world = orthonormal(world)
        if self.parent is None:
            return world
        parent_world = parent_eval.matrix_world
        if self.parent.type in {'CAMERA', 'LIGHT'}:
            parent_world = orthonormal(parent_world)
        if self.use_basis and not self.strip_scale and self.parent.type not in {'CAMERA', 'LIGHT'}:
            local = ob_eval.matrix_parent_inverse @ ob_eval.matrix_basis
            # Trust the parent-relative basis unless a driver/constraint-like effect disagrees.
            if not is_invertible(parent_world):
                return local
            if all(abs(a - b) < 1e-4 for ra, rb in zip(parent_world @ local, world) for a, b in zip(ra, rb)):
                return local
        if is_invertible(parent_world):
            return parent_world.inverted() @ world
        return None  # parent collapsed to zero scale: filled from neighbouring frames


def bake(context, tracks, start, end, deform_candidates):
    scene = context.scene
    width, height = frame_size(scene)
    frames = list(range(start, end + 1))
    deform_base, deforming, topology_changes = {}, set(), set()

    wm = context.window_manager
    wm.progress_begin(0, len(frames))
    frame_current, subframe = scene.frame_current, scene.frame_subframe
    try:
        for i, frame in enumerate(frames):
            scene.frame_set(frame)
            depsgraph = context.evaluated_depsgraph_get()
            for tr in tracks:
                ob_eval = tr.ob.evaluated_get(depsgraph)
                parent_eval = tr.parent.evaluated_get(depsgraph) if tr.parent else None
                local = tr.local_matrix(ob_eval, parent_eval)
                tr.matrices.append(matrix_rows(local) if local is not None else None)
                tr.hide_render.append(bool(ob_eval.hide_render))
                world = ob_eval.matrix_world
                if tr.kind == "camera":
                    tr.samples.append(sample_camera(ob_eval, depsgraph, orthonormal(world), width, height))
                elif tr.kind == "light":
                    tr.samples.append(sample_light(ob_eval, world.to_scale()))
                if tr.ob in deform_candidates and tr.ob not in deforming and tr.ob.type == 'MESH':
                    mesh = ob_eval.data
                    co = np.empty(len(mesh.vertices) * 3, np.float32)
                    mesh.vertices.foreach_get("co", co)
                    base = deform_base.setdefault(tr.ob, co)
                    if base.shape != co.shape:
                        topology_changes.add(tr.ob)
                        deforming.add(tr.ob)
                    elif base is not co and not np.array_equal(base, co):
                        deforming.add(tr.ob)
            wm.progress_update(i)
    finally:
        scene.frame_set(frame_current, subframe=subframe)
        wm.progress_end()
    return frames, deforming, topology_changes


def fill_gaps(matrices):
    """Frames where the local transform was undefined borrow the nearest defined one."""
    valid = [m for m in matrices if m is not None]
    if not valid:
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        return [identity] * len(matrices)
    out, last = [], None
    first = valid[0]
    for m in matrices:
        if m is not None:
            last = m
        out.append(m if m is not None else (last or first))
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def mesh_arrays(ob_eval):
    """Evaluated geometry as Cinema 4D-ready numpy arrays, or None if empty."""
    try:
        me = ob_eval.to_mesh()
    except RuntimeError:
        return None
    try:
        if me is None or not len(me.polygons):
            return None
        n_verts, n_loops, n_polys = len(me.vertices), len(me.loops), len(me.polygons)
        co = np.empty(n_verts * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)[:, [0, 2, 1]]  # Blender (x, y, z) -> C4D (x, z, y)

        loop_vert = np.empty(n_loops, np.int32)
        me.loops.foreach_get("vertex_index", loop_vert)
        start = np.empty(n_polys, np.int32)
        total = np.empty(n_polys, np.int32)
        mat_index = np.empty(n_polys, np.int32)
        me.polygons.foreach_get("loop_start", start)
        me.polygons.foreach_get("loop_total", total)
        me.polygons.foreach_get("material_index", mat_index)

        # Corner loops per output polygon, reversed for Cinema 4D's winding:
        # quad (a, b, c, d) -> (a, d, c, b); triangle (a, b, c) -> (a, c, b, b).
        quads = np.nonzero(total == 4)[0]
        tris = np.nonzero(total == 3)[0]
        ngons = np.nonzero(total > 4)[0]
        parts, mats = [], []
        if len(quads):
            s = start[quads][:, None]
            parts.append(s + np.array([0, 3, 2, 1], np.int32))
            mats.append(mat_index[quads])
        if len(tris):
            s = start[tris][:, None]
            parts.append(s + np.array([0, 2, 1, 1], np.int32))
            mats.append(mat_index[tris])
        if len(ngons):
            n_tris = len(me.loop_triangles)
            tri_loops = np.empty(n_tris * 3, np.int32)
            tri_poly = np.empty(n_tris, np.int32)
            me.loop_triangles.foreach_get("loops", tri_loops)
            me.loop_triangles.foreach_get("polygon_index", tri_poly)
            tri_loops = tri_loops.reshape(-1, 3)
            keep = np.isin(tri_poly, ngons)
            t = tri_loops[keep]
            parts.append(np.stack([t[:, 0], t[:, 2], t[:, 1], t[:, 1]], axis=1))
            mats.append(mat_index[tri_poly[keep]])
        corners = np.concatenate(parts).astype(np.int64)
        polys = loop_vert[corners].astype(np.int32)
        mats = np.concatenate(mats).clip(0, 65535).astype(np.uint16)

        uv = None
        uv_layer = next((l for l in me.uv_layers if l.active_render), None) or me.uv_layers.active
        if uv_layer is not None:
            uv_loops = np.empty(n_loops * 2, np.float32)
            uv_layer.uv.foreach_get("vector", uv_loops)
            uv_loops = uv_loops.reshape(-1, 2)
            uv_loops[:, 1] = 1.0 - uv_loops[:, 1]
            uv = uv_loops[corners]

        domain = getattr(me, "normals_domain", 'CORNER')
        normals = None
        shading = {"FACE": "flat", "POINT": "smooth"}.get(domain, "custom")
        if shading == "custom":
            n = np.empty(n_loops * 3, np.float32)
            me.corner_normals.foreach_get("vector", n)
            n = n.reshape(-1, 3)[:, [0, 2, 1]]
            normals = np.clip(np.round(n[corners] * 32000.0), -32767, 32767).astype(np.int16)

        return {"points": co, "polys": polys, "uv": uv, "normals": normals, "mats": mats, "shading": shading}
    finally:
        ob_eval.to_mesh_clear()


def geometry_key(ob):
    """Objects sharing unmodified mesh data share one geometry block."""
    if ob.type == 'MESH' and not ob.modifiers and not ob.data.shape_keys:
        return "data:" + ob.data.name_full
    return "obj:" + ob.name_full


def material_info(mat):
    color, metallic, roughness = list(mat.diffuse_color), mat.metallic, mat.roughness
    tree = getattr(mat, "node_tree", None)
    if tree is not None:
        bsdf = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is not None:
            color = list(bsdf.inputs["Base Color"].default_value)
            metallic = bsdf.inputs["Metallic"].default_value
            roughness = bsdf.inputs["Roughness"].default_value
    return {"name": mat.name, "color": [_r(c) for c in color[:3]],
            "metallic": _r(metallic), "roughness": _r(roughness)}


# ---------------------------------------------------------------------------
# Original keyframes (optional): Blender's own F-Curves instead of baked keys
# ---------------------------------------------------------------------------

EULER_ORDERS = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}
CURVE_PATHS = {"location": "loc", "rotation_euler": "rot", "scale": "scale"}
SUPPORTED_INTERPOLATION = {"BEZIER", "LINEAR", "CONSTANT"}
IDENTITY = None  # set lazily to avoid importing mathutils at module level


def _is_identity(m, tol=1e-6):
    return all(abs(m[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4))


def _axis_aligned(m, tol=1e-5):
    """A parent-inverse that is only (positive) scale + translation can be folded into
    the channel values; anything with rotation or mirroring cannot."""
    for i in range(3):
        for j in range(3):
            if i != j and abs(m[i][j]) > tol:
                return None
        if m[i][i] <= tol:
            return None
    return [m[i][i] for i in range(3)], [m[i][3] for i in range(3)]


def _fcurves_of(ob):
    action_data = ob.animation_data
    if action_data is None or action_data.action is None:
        return None
    try:
        from bpy_extras import anim_utils
        bag = anim_utils.action_get_channelbag_for_slot(action_data.action, action_data.action_slot)
        return list(bag.fcurves) if bag else []
    except (ImportError, AttributeError, TypeError):
        return list(getattr(action_data.action, "fcurves", []))


def _curve_keys(fc, factor=1.0, offset=0.0):
    """Blender F-Curve -> the same key/handle format used for transforms."""
    keys = []
    for kp in fc.keyframe_points:
        if kp.interpolation not in SUPPORTED_INTERPOLATION:
            return None
        frame, value = kp.co
        left, right = kp.handle_left, kp.handle_right
        keys.append({"f": _r(frame), "v": _r(factor * value + offset), "i": kp.interpolation[0],
                     "l": [_r(left.x - frame), _r(factor * (left.y - value))],
                     "r": [_r(right.x - frame), _r(factor * (right.y - value))]})
    keys.sort(key=lambda k: k["f"])
    return keys


def data_curves(ob, kind, width, height):
    """Keys on the camera or light data itself: lens, focus, clipping, colour, cone.

    Only channels that map to a single Cinema 4D parameter by a constant factor are
    returned; the rest keep their baked keys.
    """
    data = ob.data
    animation = getattr(data, "animation_data", None)
    if animation is None or animation.action is None:
        return {}
    aspect = width / height
    if kind == "camera":
        fit = data.sensor_fit
        horizontal = (width >= height) if fit == 'AUTO' else (fit == 'HORIZONTAL')
        targets = {
            "lens": ("lens", 1.0), "clip_start": ("clip_start", 1.0), "clip_end": ("clip_end", 1.0),
            "dof.aperture_fstop": ("fstop", 1.0),
            "shift_x": ("offset_x", 1.0 if horizontal else 1.0 / aspect),
            "shift_y": ("offset_y", aspect if horizontal else 1.0),
            ("sensor_height" if fit == 'VERTICAL' else "sensor_width"): ("gate", 1.0 if horizontal else aspect),
        }
        if data.dof.focus_object is None:
            targets["dof.focus_distance"] = ("focus_distance", 1.0)
    else:
        targets = {"color": ("color", 1.0), "energy": ("energy", 1.0)}
        if data.type == 'SPOT':
            targets["spot_size"] = ("spot_size", 1.0)
    out = {}
    for fc in _fcurves_of(data) or []:
        target = targets.get(fc.data_path)
        if target is None or fc.mute or fc.modifiers or not len(fc.keyframe_points):
            continue
        name, factor = target
        keys = _curve_keys(fc, factor)
        if keys is None:
            continue
        if name == "color":
            out.setdefault("color", {})[str(fc.array_index)] = keys
        else:
            out[name] = keys
    return out


def object_curves(ob, kind, parent_exported):
    """Blender's own keyframes for this object, ready for a 1:1 rebuild in Cinema 4D.

    Returns (curves, None) or (None, reason). Everything that can't be reproduced
    key-for-key - constraints, drivers, NLA, quaternion keys, exotic interpolation,
    a rotating parent-inverse - is reported so the caller can bake that object instead.
    """
    if kind == "light":
        return None, "lights keep baked keys"
    if ob.parent is not None and not parent_exported:
        return None, "parent not exported"
    animation = ob.animation_data
    if animation is None or animation.action is None:
        return None, "no action"
    if any(getattr(c, "enabled", True) for c in ob.constraints):
        return None, "constraints"
    if getattr(animation, "drivers", None) and len(animation.drivers):
        return None, "drivers"
    if any(track.strips for track in getattr(animation, "nla_tracks", [])):
        return None, "NLA strips"
    if any(abs(v) > 1e-9 for v in ob.delta_rotation_euler) or \
            any(abs(v) > 1e-9 for v in (ob.delta_rotation_quaternion[1:] if ob.rotation_mode == 'QUATERNION' else ())):
        return None, "delta rotation"

    fcurves = _fcurves_of(ob) or []
    channels = {"loc": {}, "rot": {}, "scale": {}}
    for fc in fcurves:
        if fc.data_path == "rotation_quaternion" and len(fc.keyframe_points):
            return None, "quaternion keys"
        if fc.data_path == "rotation_axis_angle" and len(fc.keyframe_points):
            return None, "axis-angle keys"
        group = CURVE_PATHS.get(fc.data_path)
        if group is None or fc.mute or not len(fc.keyframe_points):
            continue
        if fc.modifiers:
            return None, "F-Curve modifiers"
        if group == "rot" and ob.rotation_mode not in EULER_ORDERS:
            return None, f"rotation mode {ob.rotation_mode}"
        for kp in fc.keyframe_points:
            if kp.interpolation not in SUPPORTED_INTERPOLATION:
                return None, f"{kp.interpolation.lower()} interpolation"
        channels[group][fc.array_index] = fc
    if not any(channels.values()):
        return None, "no transform keys"

    # Parent inverse and delta transforms are folded into the values, so Cinema 4D
    # only has to permute axes: C4D (x, y, z) = Blender (x, z, y).
    scale_fold, offset_fold, offset_matrix = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], None
    if ob.parent is not None:
        if ob.parent_type != 'OBJECT':
            return None, f"{ob.parent_type.lower()} parenting"
        mpi = ob.matrix_parent_inverse
        if not _is_identity(mpi):
            folded = _axis_aligned(mpi)
            if folded is not None:
                scale_fold, offset_fold = folded
            else:
                # A rotating parent-inverse can't fold into the channels; the importer
                # puts it on a small static null between parent and object.
                offset_matrix = matrix_rows(mpi)

    loc_delta, scale_delta = list(ob.delta_location), list(ob.delta_scale)
    basis_rotation = (list(ob.rotation_euler) if ob.rotation_mode in EULER_ORDERS
                      else list(ob.matrix_basis.to_euler('XYZ')))
    order = ob.rotation_mode if ob.rotation_mode in EULER_ORDERS else 'XYZ'

    def fold(group, index):
        """(factor, offset) applied to every value and handle of one channel."""
        if group == "loc":
            return scale_fold[index], scale_fold[index] * loc_delta[index] + offset_fold[index]
        if group == "scale":
            return scale_fold[index] * scale_delta[index], 0.0
        return 1.0, 0.0

    static = {
        "loc": [_r(scale_fold[i] * (ob.location[i] + loc_delta[i]) + offset_fold[i]) for i in range(3)],
        "rot": [_r(v) for v in basis_rotation],
        "scale": [_r(scale_fold[i] * scale_delta[i] * ob.scale[i]) for i in range(3)],
    }
    out = {}
    for group, curves in channels.items():
        for index, fc in curves.items():
            factor, offset = fold(group, index)
            keys = []
            for kp in fc.keyframe_points:
                frame, value = kp.co
                left, right = kp.handle_left, kp.handle_right
                keys.append({
                    "f": _r(frame), "v": _r(factor * value + offset), "i": kp.interpolation[0],  # B/L/C
                    "l": [_r(left.x - frame), _r(factor * (left.y - value))],
                    "r": [_r(right.x - frame), _r(factor * (right.y - value))],
                })
            keys.sort(key=lambda k: k["f"])
            out.setdefault(group, {})[str(index)] = keys
    curves = {"order": order, "static": static, "channels": out}
    if offset_matrix is not None:
        curves["offset"] = offset_matrix
    return curves, None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_bridge(context, filepath, mode='EVERYTHING', range_mode='SCENE', c4d_scale=100.0):
    """c4d_scale: centimetres per Blender unit in Cinema 4D; the importer uses it as its Scale."""
    scene, view_layer = context.scene, context.view_layer
    r = scene.render
    start, end = frame_range(scene, range_mode)
    objects = gather_objects(context, mode)
    exported = set(objects)
    warnings = []

    # Collections (only those that hold exported objects, plus their ancestors)
    lcs = layer_collections(view_layer)
    lc_by_col = {lc.collection: (lc, parent) for lc, parent in lcs}

    def home_collection(ob):
        for col in ob.users_collection:
            if col in lc_by_col:
                return col
        return None

    needed = set()
    for ob in objects:
        col = home_collection(ob)
        while col is not None:
            needed.add(col)
            col = lc_by_col[col][1]
    if mode == 'EVERYTHING':
        needed = set(lc_by_col)
    collections = []
    for lc, parent in lcs:
        col = lc.collection
        if col in needed:
            collections.append({
                "name": col.name, "parent": parent.name if parent else None,
                "color": col.color_tag,
                "hide_render": bool(col.hide_render),
                "hide_viewport": bool(col.hide_viewport or lc.hide_viewport),
            })

    tracks = [ObjectTrack(ob, ob.parent in exported) for ob in objects]
    deform_candidates = {ob for ob in objects if ob.type == 'MESH' and (ob.modifiers or ob.data.shape_keys)}
    frames, deforming, topology = bake(context, tracks, start, end, deform_candidates)

    # Geometry at the first exported frame
    scene.frame_set(start)
    depsgraph = context.evaluated_depsgraph_get()
    meshes, blobs, materials = {}, {}, {}
    geo_ids = {}
    for tr in tracks:
        ob = tr.ob
        if tr.kind != "mesh":
            if ob.type in UNSUPPORTED_GEOMETRY:
                warnings.append(f"{ob.name}: {ob.type.lower()} objects aren't supported; imported as a null.")
            continue
        key = geometry_key(ob)
        if key not in geo_ids:
            arrays = mesh_arrays(ob.evaluated_get(depsgraph))
            if arrays is None:
                geo_ids[key] = None
                continue
            mesh_id = f"m{len(meshes)}"
            geo_ids[key] = mesh_id
            digest = hashlib.blake2b(digest_size=12)
            for name, ext in BLOB_EXT.items():
                arr = arrays[name]
                if arr is not None:
                    data = np.ascontiguousarray(arr).tobytes()
                    digest.update(name.encode() + data)
                    blobs[f"mesh/{mesh_id}/{name}.{ext}"] = data
            meshes[mesh_id] = {
                "points": int(len(arrays["points"])), "polys": int(len(arrays["polys"])),
                "uv": arrays["uv"] is not None, "normals": arrays["normals"] is not None,
                "shading": arrays["shading"], "hash": digest.hexdigest(),
            }
        for slot in ob.material_slots:
            if slot.material is not None and slot.material.name not in materials:
                materials[slot.material.name] = material_info(slot.material)

    for ob in sorted(deforming, key=lambda o: o.name):
        why = "changes topology" if ob in topology else "deforms"
        warnings.append(f"{ob.name} {why} over time; exported as a static mesh (frame {start}).")

    out_objects, n_curves, curve_fallbacks = [], 0, []
    for tr in tracks:
        ob = tr.ob
        entry = {
            "name": ob.name, "type": ob.type, "kind": tr.kind,
            "collection": (home_collection(ob).name if home_collection(ob) else None),
            "parent": tr.parent.name if tr.parent else None,
            "m": rle(fill_gaps(tr.matrices)),
            "hide_render": rle(tr.hide_render),
            "hide_viewport": bool(ob.hide_viewport or ob.hide_get(view_layer=view_layer)),
        }
        if tr.kind == "mesh":
            entry["mesh"] = geo_ids.get(geometry_key(ob))
            entry["materials"] = [s.material.name if s.material else None for s in ob.material_slots]
        elif tr.kind == "camera":
            cam = ob.data
            entry["camera"] = {"type": cam.type, "use_dof": bool(cam.dof.use_dof),
                               "channels": {k: pack_channel([s[k] for s in tr.samples]) for k in tr.samples[0]},
                               "curves": data_curves(ob, "camera", *frame_size(scene))}
            if cam.type == 'PANO':
                warnings.append(f"{ob.name}: panoramic cameras import as perspective.")
        elif tr.kind == "light":
            light = ob.data
            entry["light"] = {"type": light.type, "shape": getattr(light, "shape", "SQUARE"),
                              "channels": {k: pack_channel([s[k] for s in tr.samples]) for k in tr.samples[0]},
                              "curves": data_curves(ob, "light", *frame_size(scene))}
        curves, reason = object_curves(ob, tr.kind, tr.parent is not None or ob.parent is None)
        if curves is not None:
            entry["curves"] = curves
            n_curves += 1
        elif reason not in ("no action", "no transform keys", "lights keep baked keys"):
            curve_fallbacks.append(f"{ob.name} ({reason})")
        if ob.type == 'EMPTY':
            entry["empty"] = {"display": ob.empty_display_type, "size": _r(ob.empty_display_size)}
            if ob.instance_type == 'COLLECTION' and ob.instance_collection:
                warnings.append(f"{ob.name}: collection instances aren't expanded; imported as a null.")
        out_objects.append(entry)

    names = {ob.name for ob in objects}
    manifest = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "exporter": ADDON_VERSION,
        "source": {
            "application": "Blender",
            "blender_version": bpy.app.version_string,
            "file": os.path.basename(bpy.data.filepath) or "untitled.blend",
            "scene": scene.name,
        },
        "fps": r.fps / r.fps_base, "fps_int": r.fps, "fps_base": r.fps_base,
        "frame_start": start, "frame_end": end,
        "resolution": [r.resolution_x, r.resolution_y],
        "resolution_percentage": r.resolution_percentage,
        "pixel_aspect": [r.pixel_aspect_x, r.pixel_aspect_y],
        "unit_scale": scene.unit_settings.scale_length,
        "c4d_scale": float(c4d_scale),
        "scene_camera": scene.camera.name if scene.camera else None,
        "cuts": [{"frame": f, "camera": cam.name} for f, cam in camera_cuts(scene) if cam.name in names],
        "markers": [{"frame": m.frame, "name": m.name, "camera": m.camera.name if m.camera else None}
                    for m in sorted(scene.timeline_markers, key=lambda m: m.frame)],
        "collections": collections,
        "materials": sorted(materials.values(), key=lambda m: m["name"]),
        "meshes": meshes,
        "objects": out_objects,
        "warnings": warnings,
    }

    tmp = filepath + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        for name, data in blobs.items():
            zf.writestr(name, data)
    os.replace(tmp, filepath)

    if curve_fallbacks:
        shown = ", ".join(curve_fallbacks[:4]) + ("…" if len(curve_fallbacks) > 4 else "")
        warnings.append(f"{len(curve_fallbacks)} objects keep baked keys instead of their own F-Curves: {shown}")
    stats = {
        "objects": len(objects), "curves": n_curves, "meshes": len(meshes), "cameras": sum(t.kind == "camera" for t in tracks),
        "lights": sum(t.kind == "light" for t in tracks), "cuts": len(manifest["cuts"]),
        "frames": len(frames), "collections": len(collections),
        "points": sum(m["points"] for m in meshes.values()), "bytes": os.path.getsize(filepath),
    }
    return stats, warnings
