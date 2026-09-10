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


def matrix_rows(m):
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

    out_objects = []
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
                               "channels": {k: pack_channel([s[k] for s in tr.samples]) for k in tr.samples[0]}}
            if cam.type == 'PANO':
                warnings.append(f"{ob.name}: panoramic cameras import as perspective.")
        elif tr.kind == "light":
            light = ob.data
            entry["light"] = {"type": light.type, "shape": getattr(light, "shape", "SQUARE"),
                              "channels": {k: pack_channel([s[k] for s in tr.samples]) for k in tr.samples[0]}}
        if ob.type == 'EMPTY':
            entry["empty"] = {"display": ob.empty_display_type, "size": _r(ob.empty_display_size)}
            if ob.instance_type == 'COLLECTION' and ob.instance_collection:
                warnings.append(f"{ob.name}: collection instances aren't expanded; imported as a null.")
        out_objects.append(entry)

    names = {ob.name for ob in objects}
    manifest = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
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

    stats = {
        "objects": len(objects), "meshes": len(meshes), "cameras": sum(t.kind == "camera" for t in tracks),
        "lights": sum(t.kind == "light" for t in tracks), "cuts": len(manifest["cuts"]),
        "frames": len(frames), "collections": len(collections),
        "points": sum(m["points"] for m in meshes.values()), "bytes": os.path.getsize(filepath),
    }
    return stats, warnings
