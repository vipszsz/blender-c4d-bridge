"""Runs the Cinema 4D importer's pure logic against Blender ground truth, using a
small stand-in for the c4d module (vector/matrix math only).

python check_c4d_logic.py <export.c4dbridge> <worlds.json>

Checks: axis conversion of every local matrix, hierarchy composition, matrix
decomposition, run-length expansion and channel parsing. Extra checks run for
the synthetic scene from blender_scene_v2.py.
"""
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import types

# ---------------------------------------------------------------------------
# Minimal c4d stand-in
# ---------------------------------------------------------------------------


class Vector:
    def __init__(self, x=0.0, y=None, z=None):
        if isinstance(x, Vector):
            x, y, z = x.x, x.y, x.z
        elif y is None:
            y = z = x
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, s):
        if isinstance(s, Vector):
            return self.x * s.x + self.y * s.y + self.z * s.z
        return Vector(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def GetLength(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def GetNormalized(self):
        length = self.GetLength()
        return self * (1.0 / length) if length else Vector(0)


class Matrix:
    def __init__(self, off=None, v1=None, v2=None, v3=None):
        self.off = off or Vector(0)
        self.v1 = v1 or Vector(1, 0, 0)
        self.v2 = v2 or Vector(0, 1, 0)
        self.v3 = v3 or Vector(0, 0, 1)

    def __mul__(self, o):
        if isinstance(o, Vector):
            return self.off + self.v1 * o.x + self.v2 * o.y + self.v3 * o.z
        rot = lambda v: self.v1 * v.x + self.v2 * v.y + self.v3 * v.z
        return Matrix(self * o.off, rot(o.v1), rot(o.v2), rot(o.v3))


c4d = types.ModuleType("c4d")
c4d.Vector, c4d.Matrix = Vector, Matrix
for i, name in enumerate(["Ocamera", "Olight", "Opolygon", "Onull", "Ostage"]):
    setattr(c4d, name, 5100 + i)
c4d.gui = types.ModuleType("c4d.gui")
c4d.gui.GeDialog = object
c4d.plugins = types.ModuleType("c4d.plugins")
c4d.plugins.CommandData = object
sys.modules["c4d"] = c4d
sys.modules["c4d.gui"] = c4d.gui

HERE = os.path.dirname(os.path.abspath(__file__))
PYP = os.path.join(HERE, "..", "cinema4d", "CameraBridge", "camera_bridge.pyp")
loader = importlib.machinery.SourceFileLoader("bridge", PYP)
spec = importlib.util.spec_from_loader("bridge", loader)
cb = importlib.util.module_from_spec(spec)
loader.exec_module(cb)

# ---------------------------------------------------------------------------
bundle = cb.Bundle(sys.argv[1])
data = bundle.data
worlds = json.load(open(sys.argv[2]))
SCALE = 100.0
n = data["frame_end"] - data["frame_start"] + 1
objs = {o["name"]: o for o in data["objects"]}
rows = {name: cb.expand_rle(o["m"], n) for name, o in objs.items()}
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def kind(o):
    return o["kind"] if o["kind"] != "mesh" or o.get("mesh") else "null"


def parent_axes(o):
    return cb.axes_of(kind(objs[o["parent"]])) if o.get("parent") else "P"


def expected_world(name, i):
    """P · W · R straight from Blender's world matrix (scale stripped for cameras/lights)."""
    r = list(worlds[name][i])
    if kind(objs[name]) in ("camera", "light"):
        for c in range(3):
            length = math.sqrt(sum(r[4 * row + c] ** 2 for row in range(3)))
            for row in range(3):
                r[4 * row + c] /= length or 1.0
    return cb.c4d_local(r, "P", cb.axes_of(kind(objs[name])), SCALE)


def local(name, i):
    o = objs[name]
    return cb.c4d_local(rows[name][i], parent_axes(o), cb.axes_of(kind(o)), SCALE)


def c4d_world(name, i, cache):
    if (name, i) not in cache:
        o = objs[name]
        m = local(name, i)
        cache[name, i] = c4d_world(o["parent"], i, cache) * m if o.get("parent") else m
    return cache[name, i]


def rel_err(a, b):
    """max(position error in cm / 100, axis error relative to object size).

    Positions are stored with 6 decimals in Blender units (1 micron), so a
    position error of 0.01 cm maps to 1e-4 here.
    """
    size = max(b.v1.GetLength(), b.v2.GetLength(), b.v3.GetLength(), 1e-9)
    return max((a.off - b.off).GetLength() / 100.0,
               *((p - q).GetLength() / size for p, q in ((a.v1, b.v1), (a.v2, b.v2), (a.v3, b.v3))))


def det(m):
    return m.v1 * Vector(m.v2.y * m.v3.z - m.v2.z * m.v3.y, m.v2.z * m.v3.x - m.v2.x * m.v3.z,
                         m.v2.x * m.v3.y - m.v2.y * m.v3.x)


worst, worst_name, worst_dec, bad_rot, cache, checked = 0.0, "", 0.0, 0, {}, 0
for i in range(n):
    for name in objs:
        exp = expected_world(name, i)
        if abs(det(exp)) < 1e-12:
            continue  # collapsed to zero scale: invisible, any rotation is equivalent
        checked += 1
        err = rel_err(c4d_world(name, i, cache), exp)
        if err > worst:
            worst, worst_name = err, f"{name} @ frame {i + data['frame_start']}"
        m = local(name, i)
        scale, rot = cb.decompose(m)
        if rot is not None:
            rebuilt = Matrix(m.off, rot.v1 * scale.x, rot.v2 * scale.y, rot.v3 * scale.z)
            worst_dec = max(worst_dec, rel_err(rebuilt, m))
            bad_rot += abs(det(rot) - 1) > 1e-6
check(worst < 1e-4, f"C4D hierarchy reproduces Blender world transforms on {checked} object-frames "
                    f"(worst relative error {worst:.1e}, {worst_name})")
check(worst_dec < 1e-5, f"PSR decomposition is exact (worst {worst_dec:.1e})")
check(bad_rot == 0, f"all rotations proper ({bad_rot} bad)")

# Zero-scale switches: an object whose own scale is 0 is invisible exactly where Blender's is
switches = 0
for name, o in objs.items():
    own = [cb.decompose(local(name, i))[0] for i in range(n)]
    vis_c4d = [abs(s.x) > 1e-9 for s in own]
    if len(set(vis_c4d)) > 1:
        switches += 1
        w = worlds[name]
        par = o.get("parent")
        for i in range(n):
            if par is None:
                vis_bl = abs(w[i][0]) + abs(w[i][4]) + abs(w[i][8]) > 1e-9
                if vis_bl != vis_c4d[i]:
                    check(False, f"{name} visibility differs at frame {i + data['frame_start']}")
                    break
print(f"   {switches} objects switch scale to/from zero")

if "U18.A" in objs:
    cam = c4d_world("Camera_B", 0, cache)
    check(all(abs(v.GetLength() - 1) < 1e-6 for v in (cam.v1, cam.v2, cam.v3)), "camera world scale is 1")
    light = c4d_world("Area", 0, cache)
    w = worlds["Area"][0]
    bl_dir = Vector(-w[2], -w[10], -w[6]).GetNormalized()  # Blender -Z axis, converted to C4D axes
    check((light.v3.GetNormalized() - bl_dir).GetLength() < 1e-5, "area light emits the same direction")
    scales = [cb.decompose(local("U18.A", i))[0].x for i in range(n)]
    on = [i + data["frame_start"] for i, s in enumerate(scales) if s > 0.5]
    check((on[0], on[-1]) == (1, 40), f"U18.A visible frames {on[0]}-{on[-1]}")
    check(cb.channel_values({"color": [1, 0.5, 0.2]}, "color", [1, 1, 1], 3) == [[1, 0.5, 0.2]] * 3,
          "constant colour with 3 frames")
    check(cb.channel_values({"lens": [35, 36, 37]}, "lens", 50.0, 3) == [35, 36, 37], "per-frame scalar")

line1, line2, details = cb.describe(data)
print("  ", line1)
print("  ", line2)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED")
