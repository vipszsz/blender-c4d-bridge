"""Headless Cinema 4D verification of the importer (run with c4dpy).

c4dpy c4d_verify_v2.py <export.c4dbridge> <worlds.json> [--scale 50] [--save out.c4d]

Imports the export into a fresh document with the real plugin (at the scale stored
in the export, or --scale), then checks on sampled frames that each object's
evaluated global matrix equals Blender's world matrix converted to Cinema 4D axes,
that the Stage follows the cuts, that geometry and tags exist, and that a
re-import updates in place.
"""
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time

import c4d

HERE = os.path.dirname(os.path.abspath(__file__))
PYP = os.path.join(HERE, "..", "cinema4d", "CameraBridge", "camera_bridge.pyp")
loader = importlib.machinery.SourceFileLoader("bridge_mod", PYP)
spec = importlib.util.spec_from_loader("bridge_mod", loader)
cb = importlib.util.module_from_spec(spec)
loader.exec_module(cb)

argv = sys.argv[1:]
opts = {}
for flag in ("--scale", "--save"):
    if flag in argv:
        i = argv.index(flag)
        opts[flag] = argv[i + 1]
        del argv[i:i + 2]
bundle_path, worlds_path = argv[0], argv[1]
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


bundle = cb.Bundle(bundle_path)
data = bundle.data
worlds = json.load(open(worlds_path))
SCALE = float(opts.get("--scale", data.get("c4d_scale", 100.0)))
print(f"scale {SCALE:g} cm per Blender unit (export says {data.get('c4d_scale')})")

doc = c4d.documents.BaseDocument()
c4d.documents.InsertBaseDocument(doc)
c4d.documents.SetActiveDocument(doc)
opt = cb.Options(scale=SCALE)

t0 = time.perf_counter()
summary, warnings = cb.import_bundle(doc, bundle, opt)
print(f"IMPORT {time.perf_counter() - t0:.1f}s: {summary}")
for w in warnings:
    print("  note:", w)

root = cb.find_top_level(doc, cb.root_name_for(data), c4d.Onull)
check(root is not None, "root null created")
nodes = {cb.get_key(o): o for o in cb.iter_objects(root.GetDown()) if cb.get_key(o)}
objs = {o["name"]: o for o in data["objects"]}
check(all(("obj:" + n) in nodes for n in objs), f"all {len(objs)} objects created")
check(all(("col:" + c["name"]) in nodes for c in data["collections"]), "all collections created")

bad_parent = []
for name, o in objs.items():
    obj = nodes["obj:" + name]
    want = ("obj:" + o["parent"]) if o.get("parent") else (("col:" + o["collection"]) if o.get("collection") else None)
    got = cb.get_key(obj.GetUp()) if obj.GetUp() and obj.GetUp() != root else None
    if want != got:
        bad_parent.append((name, want, got))
check(not bad_parent, f"hierarchy matches Blender {bad_parent[:3]}")

meshes = data["meshes"]
bad_geo, extent = [], 0.0
for name, o in objs.items():
    if o["kind"] == "mesh" and o.get("mesh"):
        obj = nodes["obj:" + name]
        info = meshes[o["mesh"]]
        tags = {t.GetType() for t in obj.GetTags()}
        ok = (obj.GetType() == c4d.Opolygon and obj.GetPointCount() == info["points"]
              and obj.GetPolygonCount() == info["polys"] and (not info["uv"] or c4d.Tuvw in tags)
              and (not info["normals"] or c4d.Tnormal in tags)
              and (c4d.Ttexture in tags or not any(o.get("materials") or [])))
        if not ok:
            bad_geo.append((name, obj.GetPointCount(), info["points"], sorted(tags)))
        extent = max(extent, obj.GetRad().GetLength())
check(not bad_geo, f"meshes have points/polys/UVW/normals/texture tags {bad_geo[:2]}")
print(f"  largest mesh radius {extent:.2f} cm")


def expected(name, i):
    r = list(worlds[name][i])
    kind = objs[name]["kind"] if objs[name]["kind"] != "mesh" or objs[name].get("mesh") else "null"
    if kind in ("camera", "light"):
        for c in range(3):
            length = math.sqrt(sum(r[4 * row + c] ** 2 for row in range(3))) or 1.0
            for row in range(3):
                r[4 * row + c] /= length
    return cb.c4d_local(r, "P", cb.axes_of(kind), SCALE)


def det(m):
    return m.v1.Dot(m.v2.Cross(m.v3))


fps = doc.GetFps()
worst, where, stage_bad = 0.0, "", []
stage = nodes.get("stage")
cuts = sorted(data.get("cuts", []), key=lambda c: c["frame"])
n = data["frame_end"] - data["frame_start"] + 1
frames = sorted(set(range(0, n, max(1, n // 150)))
                | {c["frame"] - data["frame_start"] for c in cuts if c["frame"] >= data["frame_start"]}
                | {c["frame"] - data["frame_start"] - 1 for c in cuts if c["frame"] > data["frame_start"]})
for i in frames:
    frame = data["frame_start"] + i
    doc.SetTime(c4d.BaseTime(frame, fps))
    doc.ExecutePasses(None, True, True, True, c4d.BUILDFLAGS_NONE)
    for name in objs:
        exp = expected(name, i)
        if abs(det(exp)) < 1e-12:
            continue
        got = nodes["obj:" + name].GetMg()
        size = max(exp.v1.GetLength(), exp.v2.GetLength(), exp.v3.GetLength())
        err = max((got.off - exp.off).GetLength() / SCALE,
                  *((a - b).GetLength() / size for a, b in ((got.v1, exp.v1), (got.v2, exp.v2), (got.v3, exp.v3))))
        if err > worst:
            worst, where = err, f"{name} @ {frame}"
    if stage is not None and cuts:
        want = cuts[0]["camera"]
        for c in cuts:
            if c["frame"] <= frame:
                want = c["camera"]
        link = stage[c4d.STAGEOBJECT_CLINK]
        if (link.GetName() if link else None) != want:
            stage_bad.append((frame, link.GetName() if link else None, want))
check(worst < 2e-4, f"evaluated C4D globals match Blender on {len(frames)} frames (worst {worst:.1e}, {where})")
check(not stage_bad, f"Stage camera follows the cuts, incl. frames either side of each cut {stage_bad[:3]}")

n_mats = len(doc.GetMaterials())
before = sum(1 for _ in cb.iter_objects(doc.GetFirstObject()))
t0 = time.perf_counter()
summary2, _ = cb.import_bundle(doc, bundle, opt)
print(f"RE-IMPORT {time.perf_counter() - t0:.1f}s: {summary2}")
after = sum(1 for _ in cb.iter_objects(doc.GetFirstObject()))
check(before == after, f"re-import creates no duplicate objects ({before} -> {after})")
check(len(doc.GetMaterials()) == n_mats, f"re-import creates no duplicate materials ({n_mats} -> {len(doc.GetMaterials())})")

if "--save" in opts:
    ok = c4d.documents.SaveDocument(doc, opts["--save"], c4d.SAVEDOCUMENTFLAGS_NONE, c4d.FORMAT_C4DEXPORT)
    print("saved", opts["--save"], ok)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED")
