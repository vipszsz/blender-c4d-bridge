"""Export an existing .blend headlessly through the real operator (never saves the .blend).

blender -b --factory-startup <file.blend> --python blender_export_file.py -- <out.c4dbridge> [EVERYTHING|SELECTED|CAMERAS] [scale]
"""
import os
import sys
import time
import zipfile
import json

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
mode = args[1] if len(args) > 1 else "EVERYTHING"
scale = float(args[2]) if len(args) > 2 else 100.0
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
import camera_bridge  # noqa: E402

camera_bridge.register()
t0 = time.perf_counter()
result = bpy.ops.export_scene.c4d_bridge(filepath=out, content=mode, c4d_scale=scale)
print(f"EXPORT {result} {time.perf_counter() - t0:.1f}s")
manifest = json.loads(zipfile.ZipFile(out).read("manifest.json"))
print(f"objects {len(manifest['objects'])} c4d_scale {manifest['c4d_scale']}")
