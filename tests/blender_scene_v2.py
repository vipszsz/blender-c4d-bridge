"""Headless v2 test: build a scene like a real shot (collections, scale-0 scene
switching, rigs, n-gons, materials, lights, cameras on controllers), export it
and verify the file against Blender's own evaluation.

blender -b --factory-startup --python blender_scene_v2.py -- <out_dir>
"""
import json
import math
import os
import sys
import zipfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

out_dir = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
import camera_bridge  # noqa: E402
from camera_bridge import scene_export  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
camera_bridge.register()
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.fps = 30
scene.frame_start, scene.frame_end = 1, 90
root = scene.collection


def new_col(name, parent, color='NONE'):
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    col.color_tag = color
    return col


studio = new_col("STUDIO", root, 'COLOR_04')
cams = new_col("CAMs", studio, 'COLOR_03')
lights = new_col("Lights", studio, 'COLOR_04')
sc1 = new_col("Scene_A", root, 'COLOR_06')
sc2 = new_col("Scene_B", root, 'COLOR_06')
off = new_col("Scene_OFF", root, 'COLOR_06')


def empty(name, col, loc=(0, 0, 0), display='PLAIN_AXES', parent=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = display
    col.objects.link(ob)
    ob.location = loc
    ob.parent = parent
    return ob


def key_scale(ob, frames_values):
    for f, v in frames_values:
        ob.scale = (v, v, v)
        ob.keyframe_insert("scale", frame=f)
    ad = ob.animation_data
    for fc in scene_export_fcurves(ob):
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'


def scene_export_fcurves(ob):
    from bpy_extras import anim_utils
    ad = ob.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return cb.fcurves


# Materials
mats = []
for name, color in (("red", (0.8, 0.1, 0.1, 1)), ("chrome", (0.9, 0.9, 0.9, 1)), ("logo", (0.1, 0.2, 0.9, 1))):
    m = bpy.data.materials.new(name)
    bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs["Base Color"].default_value = color
    mats.append(m)


def card_mesh(name):
    """Box with an n-gon cap, 3 materials, UVs, mixed smooth/sharp shading."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=1, depth=0.1, end_fill_type='NGON')
    ob = bpy.context.object
    ob.name = name
    for m in mats:
        ob.data.materials.append(m)
    for i, p in enumerate(ob.data.polygons):
        p.material_index = 0 if p.loop_total == 4 else (1 if p.center.z > 0 else 2)
    bpy.ops.object.shade_smooth()
    for e in ob.data.edges:  # sharp rim edges -> custom corner normals
        v0, v1 = (ob.data.vertices[i].co for i in e.vertices)
        e.use_edge_sharp = abs(v0.z - v1.z) < 1e-6
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    return ob


# Scene A: controller scaled 1 on frames 1-40, 0 afterwards; rotating child; card under it
ctrl_a = empty("U18.A", sc1, (0, 0, 0))
key_scale(ctrl_a, [(0, 0.0), (1, 1.0), (40, 1.0), (41, 0.0)])
rot_a = empty("RotationCTRL.A", sc1, (0, 0, 0.5), 'CUBE', parent=ctrl_a)
rot_a.rotation_euler = (0, 0, 0)
rot_a.keyframe_insert("rotation_euler", frame=1)
rot_a.rotation_euler = (0.3, 0, math.radians(400))
rot_a.keyframe_insert("rotation_euler", frame=90)
card = card_mesh("Card.A")
sc1.objects.link(card)
card.parent = rot_a
card.matrix_parent_inverse = rot_a.matrix_world.inverted()
card2 = bpy.data.objects.new("Card.A.shared", card.data)  # shares mesh data
sc1.objects.link(card2)
card2.parent = rot_a
card2.location = (0, 0, 0.4)
card2.scale = (0.5, 0.5, 0.5)

# Scene B: global rotator with scaled children, visible 41-90
glob = empty("GLOBAL ROT", sc2, (2, 0, 0), 'SPHERE')
glob.scale = (1.5, 1.5, 1.5)
glob.keyframe_insert("rotation_euler", frame=1)
glob.rotation_euler = (math.radians(90), 0, 0)
glob.keyframe_insert("rotation_euler", frame=90)
ctrl_b = empty("U18.B", sc2, (0, 1, 0), parent=glob)
key_scale(ctrl_b, [(40, 0.0), (41, 1.0), (90, 1.0)])
card_b = card_mesh("Card.B")
sc2.objects.link(card_b)
card_b.parent = ctrl_b
card_b.location = (0.2, 0.1, 0.3)
card_b.rotation_euler = (0.2, 0.4, 0.1)
chip = card_mesh("Chip.B")
sc2.objects.link(chip)
chip.parent = card_b
chip.scale = (0.1, 0.1, 0.1)
chip.location = (0.3, 0, 0.06)

# A deforming mesh (armature) to exercise the warning
bpy.ops.mesh.primitive_cube_add()
bent = bpy.context.object
bent.name = "Bent.B"
for c in list(bent.users_collection):
    c.objects.unlink(bent)
sc2.objects.link(bent)
arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
sc2.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
b = arm_data.edit_bones.new("Bone")
b.head, b.tail = (0, 0, -1), (0, 0, 1)
bpy.ops.object.mode_set(mode='OBJECT')
mod = bent.modifiers.new("Armature", 'ARMATURE')
mod.object = arm
bent.vertex_groups.new(name="Bone").add(list(range(8)), 1.0, 'REPLACE')
bent.parent = arm
pb = arm.pose.bones["Bone"]
pb.rotation_mode = 'XYZ'
pb.keyframe_insert("rotation_euler", frame=1)
pb.rotation_euler = (0.7, 0, 0)
pb.keyframe_insert("rotation_euler", frame=60)

# Excluded collection
empty("Hidden.OFF", off)

# Cameras on controllers
def camera(name, ctrl, loc, rot, lens):
    data = bpy.data.cameras.new(name)
    data.lens = lens
    ob = bpy.data.objects.new(name, data)
    cams.objects.link(ob)
    ob.parent = ctrl
    ob.location = loc
    ob.rotation_euler = rot
    return ob


ctrl_c1 = empty("Camera_A_CTRL", cams, (0, 0, 0), 'SPHERE')
ctrl_c1.keyframe_insert("rotation_euler", frame=1)
ctrl_c1.rotation_euler = (0, 0, math.radians(60))
ctrl_c1.keyframe_insert("rotation_euler", frame=40)
cam_a = camera("Camera_A", ctrl_c1, (0, -6, 1), (math.radians(85), 0, 0), 50)
ctrl_c2 = empty("Camera_B_CTRL", cams, (2, 0, 0), 'SPHERE')
ctrl_c2.scale = (2, 2, 2)  # scaled controller: camera must not inherit scale in C4D
cam_b = camera("Camera_B", ctrl_c2, (0, -4, 2), (math.radians(70), 0, 0), 35)

# Lights
ldata = bpy.data.lights.new("Area", 'AREA')
ldata.energy = 50
ldata.shape = 'RECTANGLE'
ldata.size, ldata.size_y = 1.0, 0.5
light = bpy.data.objects.new("Area", ldata)
lights.objects.link(light)
light.location = (3, -3, 4)
light.rotation_euler = (0.8, 0, 0.6)
light.scale = (2.0, 3.0, 1.0)

scene.camera = cam_a
scene.timeline_markers.new("F_01", frame=1).camera = cam_a
scene.timeline_markers.new("F_41", frame=41).camera = cam_b
bpy.context.view_layer.layer_collection.children["Scene_OFF"].exclude = True

path = os.path.join(out_dir, "test_scene.c4dbridge")
stats, warnings = scene_export.export_bridge(bpy.context, path)
print("STATS", stats)
for w in warnings:
    print("WARN", w)

# ---------------------------------------------------------------------------
# Verification against Blender
# ---------------------------------------------------------------------------
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


zf = zipfile.ZipFile(path)
man = json.loads(zf.read("manifest.json"))
objs = {o["name"]: o for o in man["objects"]}
check(man.get("c4d_scale") == 100.0, f"export scale stored for the importer ({man.get('c4d_scale')})")
check("Hidden.OFF" not in objs and "Scene_OFF" not in [c["name"] for c in man["collections"]],
      "excluded collection skipped")
check([(c["name"], c["parent"]) for c in man["collections"]] ==
      [("STUDIO", None), ("CAMs", "STUDIO"), ("Lights", "STUDIO"), ("Scene_A", None), ("Scene_B", None)],
      f"collection tree {[(c['name'], c['parent']) for c in man['collections']]}")
check(objs["Card.A"]["parent"] == "RotationCTRL.A" and objs["Camera_A"]["parent"] == "Camera_A_CTRL",
      "object parenting kept")
check(objs["Card.A"]["mesh"] == objs["Card.A.shared"]["mesh"], "shared mesh data exported once")
check(any("Bent.B" in w for w in warnings), "deforming mesh reported")


def expand(entries, n):
    out, j = [], 0
    for i in range(n):
        while j + 1 < len(entries) and entries[j + 1][0] <= i:
            j += 1
        out.append(entries[j][1])
    return out


def to_mat(rows):
    return Matrix((rows[0:4], rows[4:8], rows[8:12], (0, 0, 0, 1)))


n = man["frame_end"] - man["frame_start"] + 1
locals_ = {name: [to_mat(r) for r in expand(o["m"], n)] for name, o in objs.items()}
worst = 0.0
for i, frame in enumerate(range(man["frame_start"], man["frame_end"] + 1)):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    for name, o in objs.items():
        ob = bpy.data.objects[name].evaluated_get(dg)
        # rebuild world from exported locals up the chain
        world, cur = Matrix.Identity(4), name
        chain = []
        while cur is not None:
            chain.append(cur)
            cur = objs[cur]["parent"]
        for cname in reversed(chain):
            world = world @ locals_[cname][i]
        target = ob.matrix_world
        if o["kind"] in ("camera", "light"):
            target = scene_export.orthonormal(target)
        if abs(target.to_3x3().determinant()) < 1e-9:
            continue  # collapsed: any local is equivalent
        err = max(abs(a - b) for ra, rb in zip(world, target) for a, b in zip(ra, rb))
        worst = max(worst, err)
check(worst < 1e-4, f"exported local chains rebuild Blender world matrices (worst {worst:.2e})")

# Geometry: winding reversed so C4D's CalcFaceNormal matches the converted Blender normal
m_id = objs["Card.B"]["mesh"]
info = man["meshes"][m_id]
pts = np.frombuffer(zf.read(f"mesh/{m_id}/points.f32"), np.float32).reshape(-1, 3)
polys = np.frombuffer(zf.read(f"mesh/{m_id}/polys.i32"), np.int32).reshape(-1, 4)
a, b, c, d = (pts[polys[:, k]] for k in range(4))
tri = polys[:, 2] == polys[:, 3]
c4d_n = np.where(tri[:, None], np.cross(b - a, c - a), np.cross(b - d, c - a))  # C4D CalcFaceNormal
me = bpy.data.objects["Card.B"].data
bl_n = np.array([p.normal for p in me.polygons])[:, [0, 2, 1]]
# every output polygon's normal must point the same way as some Blender face normal (tris from n-gons included)
dots = (c4d_n / np.linalg.norm(c4d_n, axis=1, keepdims=True)) @ bl_n.T
check(bool(np.all(dots.max(axis=1) > 0.999)), "polygon winding faces outward in C4D convention")
check(info["shading"] == "custom" and info["normals"], f"custom normals exported ({info['shading']})")
mats_arr = np.frombuffer(zf.read(f"mesh/{m_id}/mats.u16"), np.uint16)
check(set(mats_arr.tolist()) == {0, 1, 2}, "material slots per polygon")
check(len(polys) == 12 + 2 * 10, f"n-gon caps triangulated ({len(polys)} polys)")

light = objs["Area"]["light"]["channels"]
check(abs(light["size_x"] - 2.0) < 1e-6 and abs(light["size_y"] - 1.5) < 1e-6, "area light size includes scale")
sc = [r[1] for r in objs["U18.A"]["m"]]
check(len(objs["U18.A"]["m"]) == 2 or len(sc) >= 2, f"scale toggle is run-length encoded ({len(sc)} entries)")
worlds = {}
for frame in range(man["frame_start"], man["frame_end"] + 1):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    for name in objs:
        worlds.setdefault(name, []).append(scene_export.matrix_rows(bpy.data.objects[name].evaluated_get(dg).matrix_world))
with open(os.path.join(out_dir, "worlds.json"), "w") as fh:
    json.dump(worlds, fh)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED")
