"""Channel-level check of the curve rebuild (run with c4dpy).

c4dpy c4d_curve_match.py <export.c4dbridge> <curve_truth.json>

Rebuilds each exported Blender F-Curve as a Cinema 4D track and compares the
track's value on every frame with Blender's own evaluation of that curve.
"""
import importlib.machinery, importlib.util, json, os, sys
import c4d

HERE = os.path.dirname(os.path.abspath(__file__))
loader = importlib.machinery.SourceFileLoader("bridge_mod", os.path.join(HERE, "..", "cinema4d", "CameraBridge", "camera_bridge.pyp"))
cb = importlib.util.module_from_spec(importlib.util.spec_from_loader("bridge_mod", loader))
loader.exec_module(cb)

bundle = cb.Bundle(sys.argv[1])
truth = json.load(open(sys.argv[2]))
data = bundle.data
start, end = data["frame_start"], data["frame_end"]
fps = max(1, int(round(data["fps"])))
doc = c4d.documents.BaseDocument(); doc.SetFps(fps)
keyer = cb.Keyer(list(range(start, end + 1)), fps, 0, 100.0)

worst, where, checked = 0.0, "", 0
for o in data["objects"]:
    curves = o.get("curves")
    if not curves or o["name"] not in truth:
        continue
    for group, indices in curves["channels"].items():
        for index, keys in indices.items():
            expected = truth[o["name"]].get(group, {}).get(index)
            if expected is None:
                continue
            null = c4d.BaseObject(c4d.Onull); doc.InsertObject(null)
            descid = cb.vector_id(c4d.ID_BASEOBJECT_REL_POSITION, c4d.VECTOR_X)
            cb.write_curve_track(null, descid, keys, 1.0, 0.0, keyer)
            curve = null.FindCTrack(descid).GetCurve()
            for i, want in enumerate(expected):
                frame = start + i
                got = curve.GetValue(c4d.BaseTime(frame, fps), fps)  # exact frame time
                err = abs(got - want)
                checked += 1
                if err > worst:
                    worst, where = err, f"{o['name']} {group}[{index}] @ {frame}: C4D {got:.6f} vs Blender {want:.6f}"
            null.Remove()
print(f"compared {checked} samples across rebuilt curves")
print(f"worst difference {worst:.2e}  ({where})")
print("RESULT:", "ALL PASS" if worst < 1e-5 else "MISMATCH")
