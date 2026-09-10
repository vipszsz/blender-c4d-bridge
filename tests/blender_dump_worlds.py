"""Dump Blender world matrices per frame for every object in a .c4dbridge export (ground truth).

blender -b --factory-startup <file.blend> --python blender_dump_worlds.py -- <export.c4dbridge> <worlds.json>
"""
import json
import sys
import zipfile

import bpy

src, out = sys.argv[sys.argv.index("--") + 1:][:2]
man = json.loads(zipfile.ZipFile(src).read("manifest.json"))
scene = bpy.context.scene
worlds = {}
for frame in range(man["frame_start"], man["frame_end"] + 1):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in man["objects"]:
        m = bpy.data.objects[o["name"]].evaluated_get(dg).matrix_world
        worlds.setdefault(o["name"], []).append([round(m[r][c], 6) for r in range(3) for c in range(4)])
with open(out, "w") as fh:
    json.dump(worlds, fh)
print("dumped", len(worlds), "objects")
