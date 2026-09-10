"""Cinema 4D Bridge - import scenes exported from Blender.

Reads .c4dbridge files (File > Export > Cinema 4D Bridge in Blender) and the
older camera-only .json files.

* Blender's collection tree becomes nested nulls; parenting is kept.
* Every object is keyed from per-frame transforms, so parents, constraints and
  drivers come through exactly. Scale 0/1 visibility switches get step keys.
* Meshes come in with UVs, custom normals and one texture tag per material slot.
* Marker camera switches become a Stage object with camera keys.
* Re-importing the same .blend updates everything in place: objects, tags and
  materials you changed in Cinema 4D keep working, and unchanged meshes are
  skipped.
"""

import array
import json
import os
import traceback
import zipfile

import c4d
from c4d import gui

# IDs 1000001-1000010 are reserved by Maxon for testing. Get a unique ID from
# https://developers.maxon.net before sharing this plugin with others.
PLUGIN_ID = 1000007

FORMAT_ID = "camera-bridge"
FORMAT_VERSION = 2

# Blender exports lens shift as fractions of frame width / height with +Y up.
FILM_OFFSET_X_SIGN = 1.0
FILM_OFFSET_Y_SIGN = -1.0

# Parallel projection: visible frame width (in scene units) at CAMERA_ZOOM 1.
ORTHO_REFERENCE_WIDTH = 1024.0

# Tolerances for lossless key reduction.
EPS_POSITION = 1e-5   # Blender units (0.001 cm at the default scale); scaled with the import
EPS_ROTATION = 1e-6   # radians
EPS_SCALE = 1e-6
EPS_LENS = 1e-4       # millimetres
EPS_OTHER = 1e-4
ZERO_SCALE = 1e-9

LUMENS_PER_WATT = 683.0

# Blender collection colour tags -> icon colours.
COLLECTION_COLORS = {
    "COLOR_01": (0.89, 0.38, 0.36), "COLOR_02": (0.95, 0.64, 0.33), "COLOR_03": (0.90, 0.83, 0.36),
    "COLOR_04": (0.51, 0.77, 0.34), "COLOR_05": (0.36, 0.66, 0.90), "COLOR_06": (0.61, 0.42, 0.86),
    "COLOR_07": (0.89, 0.47, 0.76), "COLOR_08": (0.65, 0.47, 0.31),
}

EMPTY_DISPLAY = {
    "PLAIN_AXES": "NULLOBJECT_DISPLAY_POINT", "ARROWS": "NULLOBJECT_DISPLAY_AXIS",
    "SINGLE_ARROW": "NULLOBJECT_DISPLAY_AXIS", "CIRCLE": "NULLOBJECT_DISPLAY_CIRCLE",
    "CUBE": "NULLOBJECT_DISPLAY_CUBE", "SPHERE": "NULLOBJECT_DISPLAY_SPHERE",
    "CONE": "NULLOBJECT_DISPLAY_PYRAMID", "IMAGE": "NULLOBJECT_DISPLAY_RECTANGLE",
}


class BridgeError(Exception):
    """A problem the user can fix; shown as-is in the dialog."""


def plural(n, word):
    return f"{n} {word if n == 1 else (word + 'es' if word.endswith('sh') else word + 's')}"


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

def expand_rle(entries, n):
    """[[start_index, value], ...] -> one value per frame."""
    out, j = [], 0
    for i in range(n):
        while j + 1 < len(entries) and entries[j + 1][0] <= i:
            j += 1
        out.append(entries[j][1])
    return out


def upgrade_v1(data):
    """Camera-only .json (format 1) -> the scene layout used by format 2."""
    objects = []
    for cam in data["cameras"]:
        objects.append({
            "name": cam["name"], "type": "CAMERA", "kind": "camera", "collection": None, "parent": None,
            "m": [[i, m] for i, m in enumerate(cam["m"])],
            "hide_render": [[0, False]], "hide_viewport": False,
            "camera": {"type": cam.get("type", "PERSP"), "use_dof": cam.get("use_dof", False),
                       "channels": cam.get("channels", {})},
        })
    data.update(objects=objects, collections=[], materials=[], meshes={}, warnings=[], legacy=True)
    return data


class Bundle:
    """An opened export: the manifest plus lazy access to geometry blobs."""

    def __init__(self, path):
        if not path or not os.path.isfile(path):
            raise BridgeError("Choose a file exported from Blender (.c4dbridge).")
        self.path = path
        self.zip = None
        try:
            if zipfile.is_zipfile(path):
                self.zip = zipfile.ZipFile(path)
                data = json.loads(self.zip.read("manifest.json"))
            else:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise BridgeError(f"Couldn't read the file ({exc}).")
        if not isinstance(data, dict) or data.get("format") != FORMAT_ID:
            raise BridgeError("Not a Blender export. In Blender use File › Export › Cinema 4D Bridge.")
        if data.get("version", 0) > FORMAT_VERSION:
            raise BridgeError("This file comes from a newer Blender add-on. Update the Cinema 4D plugin.")
        if data.get("version", 1) == 1:
            if not data.get("cameras"):
                raise BridgeError("The file contains no cameras.")
            data = upgrade_v1(data)
        if not data.get("objects"):
            raise BridgeError("The file contains no objects.")
        self.data = data

    def blob(self, name, typecode):
        arr = array.array(typecode)
        arr.frombytes(self.zip.read(name))
        return arr

    def raw(self, name):
        return self.zip.read(name)

    def close(self):
        if self.zip is not None:
            self.zip.close()
            self.zip = None


def describe(data):
    """Summary lines and details text for the dialog."""
    objs = data["objects"]
    kinds = [o["kind"] for o in objs]
    cuts = data.get("cuts", [])
    w, h = data["resolution"]
    if data.get("legacy"):
        line1 = f"{plural(kinds.count('camera'), 'camera')} · {plural(len(cuts), 'cut')}"
    else:
        line1 = (f"{plural(len(objs), 'object')} · {plural(kinds.count('mesh'), 'mesh')} · "
                 f"{plural(kinds.count('camera'), 'camera')} · {plural(len(cuts), 'cut')} · "
                 f"{plural(kinds.count('light'), 'light')}")
    line1 += (f" · {data['fps']:g} fps · frames {data['frame_start']}–{data['frame_end']} · {w} × {h}")
    src = data.get("source", {})
    line2 = f"{src.get('file', '?')} · scene “{src.get('scene', '?')}” · Blender {src.get('blender_version', '?')}"
    lines = []
    if cuts:
        width = max(len(str(c["frame"])) for c in cuts)
        lines.append("Camera cuts")
        lines += [f"  {str(c['frame']).rjust(width)}   {c['camera']}" for c in cuts]
    else:
        lines.append("No cameras are bound to markers in this scene.")
    if data.get("warnings"):
        lines.append("")
        lines.append("From the Blender export")
        lines += [f"  • {w}" for w in data["warnings"]]
    return line1, line2, "\n".join(lines)


def root_name_for(data):
    src = data.get("source", {})
    stem = os.path.splitext(src.get("file") or "untitled")[0]
    scene = src.get("scene")
    prefix = "Blender Cameras" if data.get("legacy") else "Blender"
    return f"{prefix} · {stem}" + (f" / {scene}" if scene and scene != "Scene" else "")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
#
# Blender is right-handed Z-up, Cinema 4D left-handed Y-up. A Blender world
# matrix W maps to Cinema 4D as P·W·R, where P swaps Y/Z and R is P for regular
# objects (their geometry is converted the same way) or D = diag(1, 1, -1) for
# cameras and lights (Blender points them down -Z, Cinema 4D down +Z). A local
# matrix L relative to a parent therefore converts to R_parent · L · R_child.

def axes_of(kind):
    return "D" if kind in ("camera", "light") else "P"


def c4d_local(rows, parent_axes, own_axes, scale):
    r0, r1, r2 = rows[0:4], rows[4:8], rows[8:12]
    if parent_axes == "P":
        r1, r2 = r2, r1
    else:
        r2 = [-x for x in r2]
    if own_axes == "P":
        cols, signs = (0, 2, 1), (1.0, 1.0, 1.0)
    else:
        cols, signs = (0, 1, 2), (1.0, 1.0, -1.0)
    v = [c4d.Vector(r0[c] * s, r1[c] * s, r2[c] * s) for c, s in zip(cols, signs)]
    return c4d.Matrix(c4d.Vector(r0[3], r1[3], r2[3]) * scale, v[0], v[1], v[2])


def to_c4d_matrix(m, scale):
    """Blender world matrix of a camera (3x4 rows) -> Cinema 4D matrix."""
    return c4d_local(m, "P", "D", scale)


def _cross(a, b):
    return c4d.Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def decompose(m):
    """Matrix -> (scale Vector, orthonormal rotation Matrix or None if undefined)."""
    axes = [c4d.Vector(m.v1), c4d.Vector(m.v2), c4d.Vector(m.v3)]
    lengths = [a.GetLength() for a in axes]
    if _dot(axes[0], _cross(axes[1], axes[2])) < 0:
        lengths[2] = -lengths[2]
        axes[2] = -axes[2]
    scale = c4d.Vector(lengths[0], lengths[1], lengths[2])
    ok = [abs(l) > ZERO_SCALE for l in lengths]
    if sum(ok) < 2:
        return scale, None
    n = [a.GetNormalized() if good else None for a, good in zip(axes, ok)]
    if n[0] is None:
        n[0] = _cross(n[1], n[2])
    if n[1] is None:
        n[1] = _cross(n[2], n[0])
    # Gram-Schmidt keeps the rotation clean even if the source had shear.
    x = n[0].GetNormalized()
    y = (n[1] - x * _dot(x, n[1])).GetNormalized()
    z = _cross(x, y)
    return scale, c4d.Matrix(c4d.Vector(0), x, y, z)


def fill_none(values, default):
    out, last = [], None
    first = next((v for v in values if v is not None), default)
    for v in values:
        if v is not None:
            last = v
        out.append(v if v is not None else (last if last is not None else first))
    return out


def reduce_keys(frames, values, eps):
    """Indices of keys needed so linear interpolation stays within eps of every sample.

    Greedy slope-window ("swinging door") pass, O(n).
    """
    n = len(values)
    if n <= 2:
        return list(range(n))
    keep = [0]
    a = 0
    lo, hi = float("-inf"), float("inf")
    j = 1
    while j < n:
        dt = frames[j] - frames[a]
        slope = (values[j] - values[a]) / dt
        if lo <= slope <= hi:
            lo = max(lo, (values[j] - eps - values[a]) / dt)
            hi = min(hi, (values[j] + eps - values[a]) / dt)
            j += 1
        else:
            a = j - 1
            keep.append(a)
            lo, hi = float("-inf"), float("inf")
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return keep


def channel_values(channels, key, default, n):
    """Channels hold one value, or one value per frame (vector channels: a list of lists)."""
    v = channels.get(key, default)
    per_frame = isinstance(v, list) and len(v) == n and (not isinstance(default, list) or isinstance(v[0], list))
    return list(v) if per_frame else [v] * n


# ---------------------------------------------------------------------------
# Animation tracks
# ---------------------------------------------------------------------------

def vector_id(param, component):
    return c4d.DescID(c4d.DescLevel(param, c4d.DTYPE_VECTOR, 0),
                      c4d.DescLevel(component, c4d.DTYPE_REAL, 0))


def real_id(param):
    return c4d.DescID(c4d.DescLevel(param, c4d.DTYPE_REAL, 0))


def long_id(param):
    return c4d.DescID(c4d.DescLevel(param, c4d.DTYPE_LONG, 0))


def remove_track(obj, descid):
    track = obj.FindCTrack(descid)
    if track is not None:
        track.Remove()


class Keyer:
    def __init__(self, frames, fps, offset, scale=100.0):
        self.frames = [f + offset for f in frames]
        self.fps = fps
        self.offset = offset
        self.pos_eps = EPS_POSITION * scale  # distance tolerance in Cinema 4D units
        self.keys = 0

    def time(self, frame):
        return c4d.BaseTime(frame, self.fps)

    def channel(self, obj, descid, values, eps, step_zero=False, step_all=False):
        """Key a channel; constant channels become a plain value.

        step_zero: a jump to or from zero between two adjacent frames gets a
        step key (Blender visibility switches done with constant scale keys).
        """
        remove_track(obj, descid)
        if max(values) - min(values) <= eps:
            obj.SetParameter(descid, values[0], c4d.DESCFLAGS_SET_NONE)
            return
        track = c4d.CTrack(obj, descid)
        obj.InsertTrackSorted(track)
        curve = track.GetCurve()
        idx = reduce_keys(self.frames, values, eps)
        for n, i in enumerate(idx):
            key = curve.AddKey(self.time(self.frames[i]))["key"]
            key.SetValue(curve, values[i])
            interp = c4d.CINTERPOLATION_LINEAR
            if step_all:
                interp = c4d.CINTERPOLATION_STEP
            elif step_zero and n + 1 < len(idx):
                j = idx[n + 1]
                if j == i + 1 and (abs(values[i]) < ZERO_SCALE) != (abs(values[j]) < ZERO_SCALE):
                    interp = c4d.CINTERPOLATION_STEP
            key.SetInterpolation(curve, interp)
            self.keys += 1


def key_transform(obj, matrices, keyer):
    """Position, rotation (HPB, unwrapped) and scale keys from per-frame local matrices."""
    order = c4d.ROTATIONORDER_DEFAULT
    parts = [decompose(m) for m in matrices]
    hpbs = [c4d.utils.MatrixToHPB(rot, order) if rot is not None else None for _s, rot in parts]
    hpbs = fill_none(hpbs, c4d.Vector(0))
    for i in range(1, len(hpbs)):
        hpbs[i] = c4d.utils.GetOptimalAngle(hpbs[i - 1], hpbs[i], order)
    pos_id, rot_id, scl_id = (c4d.ID_BASEOBJECT_REL_POSITION, c4d.ID_BASEOBJECT_REL_ROTATION,
                              c4d.ID_BASEOBJECT_REL_SCALE)
    for comp, attr in ((c4d.VECTOR_X, "x"), (c4d.VECTOR_Y, "y"), (c4d.VECTOR_Z, "z")):
        keyer.channel(obj, vector_id(pos_id, comp), [getattr(m.off, attr) for m in matrices], keyer.pos_eps)
        keyer.channel(obj, vector_id(rot_id, comp), [getattr(r, attr) for r in hpbs], EPS_ROTATION)
        keyer.channel(obj, vector_id(scl_id, comp), [getattr(s, attr) for s, _r in parts], EPS_SCALE,
                      step_zero=True)


# ---------------------------------------------------------------------------
# Scene objects
# ---------------------------------------------------------------------------

class Options:
    def __init__(self, scale=100.0, frame_offset=0, stage=True, markers=True,
                 match_document=True, clipping=True):
        self.scale = scale
        self.frame_offset = frame_offset
        self.stage = stage
        self.markers = markers
        self.match_document = match_document
        self.clipping = clipping


def get_key(node):
    bc = node.GetDataInstance().GetContainerInstance(PLUGIN_ID)
    return bc.GetString(1) if bc is not None else ""


def get_hash(node):
    bc = node.GetDataInstance().GetContainerInstance(PLUGIN_ID)
    return bc.GetString(2) if bc is not None else ""


def set_key(node, key, digest=None):
    data = node.GetDataInstance()
    sub = data.GetContainer(PLUGIN_ID)
    sub.SetString(1, key)
    if digest is not None:
        sub.SetString(2, digest)
    data.SetContainer(PLUGIN_ID, sub)


def iter_objects(obj):
    while obj:
        yield obj
        yield from iter_objects(obj.GetDown())
        obj = obj.GetNext()


def find_top_level(doc, name, type_id):
    obj = doc.GetFirstObject()
    while obj:
        if obj.GetType() == type_id and obj.GetName() == name:
            return obj
        obj = obj.GetNext()
    return None


def find_child(parent, name, type_id):
    child = parent.GetDown() if parent else None
    while child:
        if child.GetType() == type_id and child.GetName() == name:
            return child
        child = child.GetNext()
    return None


KIND_TYPES = {"camera": c4d.Ocamera, "light": c4d.Olight, "mesh": c4d.Opolygon, "null": c4d.Onull}


def new_object(kind):
    if kind == "mesh":
        return c4d.PolygonObject(0, 0)
    return c4d.BaseObject(KIND_TYPES[kind])


def set_visibility(obj, hide_render, hide_viewport):
    obj[c4d.ID_BASEOBJECT_VISIBILITY_RENDER] = c4d.MODE_OFF if hide_render else c4d.MODE_UNDEF
    obj[c4d.ID_BASEOBJECT_VISIBILITY_EDITOR] = c4d.MODE_OFF if hide_viewport else c4d.MODE_UNDEF


def set_icon_color(obj, rgb):
    try:
        obj[c4d.ID_BASELIST_ICON_COLORIZE_MODE] = c4d.ID_BASELIST_ICON_COLORIZE_MODE_CUSTOM
        obj[c4d.ID_BASELIST_ICON_COLOR] = c4d.Vector(*rgb)
    except (AttributeError, TypeError):
        pass


def apply_camera(obj, cam, keyer, opt, warnings, name):
    n = len(keyer.frames)
    ch = cam.get("channels", {})
    s = opt.scale

    def values(key, default):
        return channel_values(ch, key, default, n)

    kind = cam.get("type", "PERSP")
    ortho = kind == "ORTHO"
    if kind == "PANO":
        warnings.append(f"{name}: panoramic cameras aren't supported; imported as perspective.")

    obj[c4d.CAMERA_PROJECTION] = c4d.Pparallel if ortho else c4d.Pperspective
    obj[c4d.CAMERAOBJECT_APERTURE_PRESET] = c4d.CAMERAOBJECT_APERTURE_PRESET_CUSTOM
    obj[c4d.CAMERAOBJECT_FOCUS_PRESET] = c4d.CAMERAOBJECT_FOCUS_PRESET_CUSTOM
    obj[c4d.CAMERAOBJECT_FNUMBER] = c4d.CAMERAOBJECT_FNUMBER_CUSTOM
    obj[c4d.CAMERAOBJECT_USETARGETOBJECT] = False

    keyer.channel(obj, real_id(c4d.CAMERA_FOCUS),
                  [min(max(v, 1.0), 10000.0) for v in values("lens", 50.0)], EPS_LENS)
    keyer.channel(obj, real_id(c4d.CAMERAOBJECT_APERTURE),
                  [min(max(v, 1.0), 2000.0) for v in values("gate", 36.0)], EPS_LENS)
    keyer.channel(obj, real_id(c4d.CAMERAOBJECT_FILM_OFFSET_X),
                  [FILM_OFFSET_X_SIGN * v for v in values("offset_x", 0.0)], EPS_OTHER)
    keyer.channel(obj, real_id(c4d.CAMERAOBJECT_FILM_OFFSET_Y),
                  [FILM_OFFSET_Y_SIGN * v for v in values("offset_y", 0.0)], EPS_OTHER)
    if ortho:
        keyer.channel(obj, real_id(c4d.CAMERA_ZOOM),
                      [ORTHO_REFERENCE_WIDTH / max(v * s, 1e-6) for v in values("ortho_width", 6.0)], EPS_OTHER)
    keyer.channel(obj, real_id(c4d.CAMERAOBJECT_TARGETDISTANCE),
                  [max(v * s, 0.01) for v in values("focus_distance", 10.0)], keyer.pos_eps)
    keyer.channel(obj, real_id(c4d.CAMERAOBJECT_FNUMBER_VALUE), values("fstop", 2.8), EPS_OTHER)

    near_id, far_id = real_id(c4d.CAMERAOBJECT_NEAR_CLIPPING), real_id(c4d.CAMERAOBJECT_FAR_CLIPPING)
    obj[c4d.CAMERAOBJECT_NEAR_CLIPPING_ENABLE] = opt.clipping
    obj[c4d.CAMERAOBJECT_FAR_CLIPPING_ENABLE] = opt.clipping
    if opt.clipping:
        keyer.channel(obj, near_id, [v * s for v in values("clip_start", 0.1)], keyer.pos_eps)
        keyer.channel(obj, far_id, [v * s for v in values("clip_end", 100.0)], keyer.pos_eps)
    else:
        remove_track(obj, near_id)
        remove_track(obj, far_id)


LIGHT_TYPES = {"POINT": "LIGHT_TYPE_OMNI", "SPOT": "LIGHT_TYPE_SPOT", "SUN": "LIGHT_TYPE_DISTANT",
               "AREA": "LIGHT_TYPE_AREA"}


def apply_light(obj, light, keyer, opt):
    n = len(keyer.frames)
    ch = light.get("channels", {})
    kind = light.get("type", "POINT")
    obj[c4d.LIGHT_TYPE] = getattr(c4d, LIGHT_TYPES.get(kind, "LIGHT_TYPE_OMNI"))
    obj[c4d.LIGHT_SHADOWTYPE] = c4d.LIGHT_SHADOWTYPE_AREA

    colors = channel_values(ch, "color", [1.0, 1.0, 1.0], n)
    for comp, idx in ((c4d.VECTOR_X, 0), (c4d.VECTOR_Y, 1), (c4d.VECTOR_Z, 2)):
        keyer.channel(obj, vector_id(c4d.LIGHT_COLOR, comp), [c[idx] for c in colors], EPS_OTHER)

    energy = channel_values(ch, "energy", 10.0, n)
    if kind == "SUN":
        obj[c4d.LIGHT_PHOTOMETRIC_UNITS] = False
        keyer.channel(obj, real_id(c4d.LIGHT_BRIGHTNESS), energy, EPS_OTHER)
    else:
        obj[c4d.LIGHT_DETAILS_FALLOFF] = c4d.LIGHT_DETAILS_FALLOFF_INVERSESQUARE
        obj[c4d.LIGHT_PHOTOMETRIC_UNITS] = True
        obj[c4d.LIGHT_PHOTOMETRIC_UNIT] = c4d.LIGHT_PHOTOMETRIC_UNIT_LUMEN
        keyer.channel(obj, real_id(c4d.LIGHT_PHOTOMETRIC_INTENSITY),
                      [e * LUMENS_PER_WATT for e in energy], EPS_OTHER)

    if kind == "AREA":
        shape = light.get("shape", "SQUARE")
        obj[c4d.LIGHT_AREADETAILS_SHAPE] = (c4d.LIGHT_AREADETAILS_SHAPE_DISC if shape in ("DISK", "ELLIPSE")
                                            else c4d.LIGHT_AREADETAILS_SHAPE_RECTANGLE)
        keyer.channel(obj, real_id(c4d.LIGHT_AREADETAILS_SIZEX),
                      [v * opt.scale for v in channel_values(ch, "size_x", 1.0, n)], keyer.pos_eps)
        keyer.channel(obj, real_id(c4d.LIGHT_AREADETAILS_SIZEY),
                      [v * opt.scale for v in channel_values(ch, "size_y", 1.0, n)], keyer.pos_eps)
    elif kind == "SPOT":
        sizes = channel_values(ch, "spot_size", 0.785, n)
        blends = channel_values(ch, "spot_blend", 0.15, n)
        obj[c4d.LIGHT_DETAILS_INNERCONE] = True
        keyer.channel(obj, real_id(c4d.LIGHT_DETAILS_OUTERANGLE), sizes, EPS_ROTATION)
        keyer.channel(obj, real_id(c4d.LIGHT_DETAILS_INNERANGLE),
                      [s * (1.0 - b) for s, b in zip(sizes, blends)], EPS_ROTATION)


def apply_empty(obj, empty, opt):
    display = getattr(c4d, EMPTY_DISPLAY.get(empty.get("display"), "NULLOBJECT_DISPLAY_POINT"))
    obj[c4d.NULLOBJECT_DISPLAY] = display
    obj[c4d.NULLOBJECT_RADIUS] = max(float(empty.get("size", 1.0)) * opt.scale, 0.01)
    if empty.get("display") == "CIRCLE":
        obj[c4d.NULLOBJECT_ORIENTATION] = c4d.NULLOBJECT_ORIENTATION_XZ


# ---------------------------------------------------------------------------
# Geometry and materials
# ---------------------------------------------------------------------------

def remove_tags(obj, prefix):
    for tag in obj.GetTags():
        if get_key(tag).startswith(prefix):
            tag.Remove()


def unique_names(names):
    seen, out = {}, []
    for name in names:
        base = name or "Material"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}.{count}")
    return out


def write_geometry(obj, bundle, mesh_id, info, slot_names, scale, warnings):
    """(Re)build points, polygons, UVs, normals and material selections on a polygon object."""
    n_points, n_polys = info["points"], info["polys"]
    base = f"mesh/{mesh_id}/"
    pts = bundle.blob(base + "points.f32", "f")
    polys = bundle.blob(base + "polys.i32", "i")
    obj.ResizeObject(n_points, n_polys)
    vec = c4d.Vector
    obj.SetAllPoints([vec(x * scale, y * scale, z * scale) for x, y, z in zip(pts[0::3], pts[1::3], pts[2::3])])
    poly, set_poly = c4d.CPolygon, obj.SetPolygon
    for i in range(n_polys):
        k = 4 * i
        set_poly(i, poly(polys[k], polys[k + 1], polys[k + 2], polys[k + 3]))

    remove_tags(obj, "geo:")

    if info.get("uv"):
        uv = bundle.blob(base + "uv.f32", "f")
        tag = c4d.UVWTag(n_polys)
        set_uv = tag.SetSlow
        for i in range(n_polys):
            k = 8 * i
            set_uv(i, vec(uv[k], uv[k + 1], 0), vec(uv[k + 2], uv[k + 3], 0),
                   vec(uv[k + 4], uv[k + 5], 0), vec(uv[k + 6], uv[k + 7], 0))
        set_key(tag, "geo:uvw")
        obj.InsertTag(tag)

    shading = info.get("shading", "smooth")
    if shading != "flat":
        phong = obj.MakeTag(c4d.Tphong)
        phong[c4d.PHONGTAG_PHONG_ANGLELIMIT] = False
        set_key(phong, "geo:phong")
    if info.get("normals"):
        try:
            raw = bundle.raw(base + "normals.i16")
            tag = c4d.NormalTag(n_polys)
            mem = tag.GetLowlevelDataAddressW()
            if mem is None:
                raise ValueError("no normal tag memory")
            mem = mem.cast("B") if mem.format != "B" else mem
            if len(mem) != len(raw):
                raise ValueError(f"normal tag size {len(mem)} != {len(raw)}")
            mem[:] = raw
            set_key(tag, "geo:normals")
            obj.InsertTag(tag)
        except Exception as exc:  # keep going with Phong shading only
            warnings.append(f"{obj.GetName()}: custom normals skipped ({exc}).")

    mats = bundle.blob(base + "mats.u16", "H")
    used = sorted(set(mats))
    names = unique_names(slot_names)
    if len(used) > 1:
        for slot in used:
            tag = c4d.SelectionTag(c4d.Tpolygonselection)
            tag.SetName(names[slot] if slot < len(names) else f"Slot {slot + 1}")
            select = tag.GetBaseSelect()
            try:
                select.SetAll([m == slot for m in mats])
            except (AttributeError, TypeError):
                for i, m in enumerate(mats):
                    if m == slot:
                        select.Select(i)
            set_key(tag, f"geo:sel:{slot}")
            obj.InsertTag(tag)
    obj.Message(c4d.MSG_UPDATE)
    return used


class Materials:
    def __init__(self, doc, infos):
        self.doc = doc
        self.infos = {m["name"]: m for m in infos}
        self.created = 0

    def get(self, name):
        mat = self.doc.SearchMaterial(name)
        if mat is not None:
            return mat
        mat = c4d.BaseMaterial(c4d.Mmaterial)
        mat.SetName(name)
        info = self.infos.get(name)
        if info:
            mat[c4d.MATERIAL_COLOR_COLOR] = c4d.Vector(*info["color"])
        self.doc.InsertMaterial(mat)
        self.doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, mat)
        self.created += 1
        return mat


def apply_textures(obj, slot_names, used_slots, materials):
    """One texture tag per used slot. Existing tags keep whatever material you linked in C4D."""
    existing = {get_key(t): t for t in obj.GetTags() if get_key(t).startswith("tex:")}
    names = unique_names(slot_names)
    multi = len(used_slots) > 1
    for slot in used_slots:
        if slot >= len(slot_names) or not slot_names[slot]:
            continue
        key = f"tex:{slot}"
        tag = existing.pop(key, None)
        if tag is None:
            tag = obj.MakeTag(c4d.Ttexture)
            tag[c4d.TEXTURETAG_MATERIAL] = materials.get(slot_names[slot])
            tag[c4d.TEXTURETAG_PROJECTION] = c4d.TEXTURETAG_PROJECTION_UVW
            set_key(tag, key)
        tag[c4d.TEXTURETAG_RESTRICTION] = names[slot] if multi else ""
    for tag in existing.values():
        tag.Remove()


# ---------------------------------------------------------------------------
# Document, Stage, markers
# ---------------------------------------------------------------------------

def apply_document_settings(doc, data, opt):
    fps = max(1, int(round(float(data["fps"]))))
    doc.SetFps(fps)
    start = c4d.BaseTime(data["frame_start"] + opt.frame_offset, fps)
    end = c4d.BaseTime(data["frame_end"] + opt.frame_offset, fps)
    for _ in range(2):  # twice, so neither bound gets clamped by the old other bound
        doc.SetMinTime(start)
        doc.SetMaxTime(end)
        doc.SetLoopMinTime(start)
        doc.SetLoopMaxTime(end)
    rd = doc.GetActiveRenderData()
    doc.AddUndo(c4d.UNDOTYPE_CHANGE, rd)
    w, h = data["resolution"]
    rd[c4d.RDATA_LOCKRATIO] = False
    rd[c4d.RDATA_XRES] = float(w)
    rd[c4d.RDATA_YRES] = float(h)
    rd[c4d.RDATA_FRAMERATE] = float(fps)
    pax, pay = data.get("pixel_aspect", [1.0, 1.0])
    if pay:
        rd[c4d.RDATA_PIXELASPECT] = float(pax) / float(pay)


def remove_bridge_markers(doc, tag):
    marker = c4d.documents.GetFirstMarker(doc)
    while marker:
        nxt = marker.GetNext()
        if marker.GetDataInstance().GetString(PLUGIN_ID) == tag:
            doc.AddUndo(c4d.UNDOTYPE_DELETEOBJ, marker)
            marker.Remove()
        marker = nxt


def add_markers(doc, data, keyer, tag, warnings):
    if not hasattr(c4d.documents, "AddMarker"):
        warnings.append("This Cinema 4D version has no Python marker API; timeline markers were skipped.")
        return 0
    remove_bridge_markers(doc, tag)
    pred, count = None, 0
    for m in sorted(data.get("markers", []), key=lambda m: m["frame"]):
        name = m["name"] or "Marker"
        if m.get("camera"):
            name = f"{name} · {m['camera']}"
        marker = c4d.documents.AddMarker(doc, pred, keyer.time(m["frame"] + keyer.offset), name)
        if marker is None:
            continue
        marker.GetDataInstance().SetString(PLUGIN_ID, tag)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, marker)
        pred = marker
        count += 1
    return count


def key_stage(doc, stage, data, cams, keyer, warnings):
    cuts = [c for c in data.get("cuts", []) if c["camera"] in cams]
    first = cams.get(cuts[0]["camera"]) if cuts else cams.get(data.get("scene_camera") or "")
    link_id = c4d.DescID(c4d.DescLevel(c4d.STAGEOBJECT_CLINK, c4d.DTYPE_BASELISTLINK, 0))
    remove_track(stage, link_id)
    if first is None:
        warnings.append("No cameras are bound to markers in Blender, so the Stage has no cuts.")
        return 0
    stage[c4d.STAGEOBJECT_CLINK] = first
    if len(cuts) > 1:
        track = c4d.CTrack(stage, link_id)
        stage.InsertTrackSorted(track)
        curve = track.GetCurve()
        for cut in cuts:
            # Link tracks hold data keys: they must be filled by the track, not added as value keys.
            key = c4d.CKey()
            track.FillKey(doc, stage, key)
            key.SetTime(curve, keyer.time(cut["frame"] + keyer.offset))
            key.SetGeData(curve, cams[cut["camera"]])
            curve.InsertKey(key)
    others = [o for o in iter_objects(doc.GetFirstObject()) if o.GetType() == c4d.Ostage and o != stage]
    if others:
        warnings.append(f"The scene has another Stage object (“{others[0].GetName()}”). Only one Stage "
                        f"should drive the camera; disable the others.")
    return len(cuts)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def progress(text, fraction):
    c4d.StatusSetText(text)
    c4d.StatusSetBar(int(max(0.0, min(1.0, fraction)) * 100))


def import_bundle(doc, bundle, opt):
    """Build or update the scene. Returns (summary, warnings)."""
    data = bundle.data
    warnings = []
    frames = list(range(data["frame_start"], data["frame_end"] + 1))
    n = len(frames)
    blender_fps = float(data["fps"])
    root_name = root_name_for(data)
    objects = data["objects"]
    by_name = {o["name"]: o for o in objects}

    doc.StartUndo()
    try:
        if opt.match_document:
            apply_document_settings(doc, data, opt)
        fps = doc.GetFps()
        if abs(fps - blender_fps) > 1e-3:
            if round(blender_fps) == fps:
                warnings.append(f"Blender runs at {blender_fps:.3f} fps; Cinema 4D uses whole numbers ({fps}). "
                                f"Frame numbers still match 1:1.")
            else:
                warnings.append(f"This document runs at {fps} fps, Blender at {blender_fps:g} fps. Keys sit on the "
                                f"same frame numbers, so playback speed differs. Turn on “Match FPS, frame range "
                                f"and resolution” to fix.")
        keyer = Keyer(frames, fps, opt.frame_offset, opt.scale)

        # Root and index of what a previous import left behind
        root = find_top_level(doc, root_name, c4d.Onull)
        if root is None:
            root = c4d.BaseObject(c4d.Onull)
            root.SetName(root_name)
            root[c4d.NULLOBJECT_DISPLAY] = c4d.NULLOBJECT_DISPLAY_NONE
            doc.InsertObject(root)
            doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, root)
        index, legacy = {}, {}
        for obj in iter_objects(root.GetDown()):
            key = get_key(obj)
            if key:
                index.setdefault(key, obj)
            else:
                legacy.setdefault((obj.GetName(), obj.GetType()), obj)
        touched = set()
        created = updated = 0

        def claim(key, name, kind):
            nonlocal created, updated
            type_id = KIND_TYPES[kind]
            obj = index.get(key) or legacy.pop((name, type_id), None)
            if obj is not None and obj.GetType() != type_id:
                doc.AddUndo(c4d.UNDOTYPE_DELETEOBJ, obj)
                obj.Remove()
                obj = None
            if obj is None:
                obj = new_object(kind)
                obj.SetName(name)
                set_key(obj, key)
                created += 1
                return obj, True
            doc.AddUndo(c4d.UNDOTYPE_CHANGE, obj)
            obj.SetName(name)
            set_key(obj, key)
            updated += 1
            return obj, False

        nodes, is_new = {}, {}

        # Collections -> nulls
        col_nodes = {}
        for col in data.get("collections", []):
            key = "col:" + col["name"]
            obj, new = claim(key, col["name"], "null")
            obj[c4d.NULLOBJECT_DISPLAY] = c4d.NULLOBJECT_DISPLAY_NONE
            set_visibility(obj, col.get("hide_render"), col.get("hide_viewport"))
            rgb = COLLECTION_COLORS.get(col.get("color"))
            if rgb:
                set_icon_color(obj, rgb)
            col_nodes[col["name"]] = obj
            is_new[key] = new
            nodes[key] = obj

        # Objects
        meshes = data.get("meshes", {})
        for o in objects:
            kind = o["kind"]
            if kind == "mesh" and not (o.get("mesh") and o["mesh"] in meshes):
                kind = "null"
            o["_kind"] = kind
            key = "obj:" + o["name"]
            obj, new = claim(key, o["name"], kind)
            nodes[key] = obj
            is_new[key] = new

        stage = None
        if opt.stage:
            stage = index.get("stage") or find_child(root, "Stage", c4d.Ostage)
            if stage is None:
                stage = c4d.BaseObject(c4d.Ostage)
                stage.SetName("Stage")
                is_new["stage"] = True
            else:
                doc.AddUndo(c4d.UNDOTYPE_CHANGE, stage)
                is_new["stage"] = False
            set_key(stage, "stage")
            nodes["stage"] = stage

        # Hierarchy: collections first, then objects, alphabetical like Blender's outliner
        children = {}

        def parent_key(o):
            if o.get("parent") and o["parent"] in by_name:
                return "obj:" + o["parent"]
            if o.get("collection") and o["collection"] in col_nodes:
                return "col:" + o["collection"]
            return "root"

        if stage is not None:
            children.setdefault("root", []).append("stage")
        for col in data.get("collections", []):
            children.setdefault("col:" + col["parent"] if col.get("parent") in col_nodes else "root",
                                []).append("col:" + col["name"])
        for o in sorted(objects, key=lambda o: o["name"].lower()):
            children.setdefault(parent_key(o), []).append("obj:" + o["name"])

        def place(parent_obj, parent_key_):
            for key in children.get(parent_key_, []):
                obj = nodes[key]
                if not is_new[key]:
                    obj.Remove()
                obj.InsertUnderLast(parent_obj)
                if is_new[key]:
                    doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, obj)
                touched.add(key)
                place(obj, key)

        place(root, "root")

        # Animation, parameters, geometry
        materials = Materials(doc, data.get("materials", []))
        built, skipped_geo = {}, 0
        cams = {}
        total = len(objects)
        for count, o in enumerate(objects):
            progress(f"Importing {o['name']}…", count / max(total, 1))
            key = "obj:" + o["name"]
            obj = nodes[key]
            kind = o["_kind"]
            parent = by_name.get(o.get("parent") or "")
            parent_axes = axes_of(parent["_kind"]) if parent else "P"
            rows = expand_rle(o["m"], n)
            mats = [c4d_local(r, parent_axes, axes_of(kind), opt.scale) for r in rows]
            key_transform(obj, mats, keyer)

            vis = expand_rle(o.get("hide_render", [[0, False]]), n)
            vis_id = long_id(c4d.ID_BASEOBJECT_VISIBILITY_RENDER)
            if len(set(vis)) > 1:
                keyer.channel(obj, vis_id, [float(c4d.MODE_OFF if v else c4d.MODE_UNDEF) for v in vis], 0.5,
                              step_all=True)
            else:
                remove_track(obj, vis_id)
                obj[c4d.ID_BASEOBJECT_VISIBILITY_RENDER] = c4d.MODE_OFF if vis[0] else c4d.MODE_UNDEF
            obj[c4d.ID_BASEOBJECT_VISIBILITY_EDITOR] = c4d.MODE_OFF if o.get("hide_viewport") else c4d.MODE_UNDEF

            if kind == "camera":
                apply_camera(obj, o["camera"], keyer, opt, warnings, o["name"])
                cams[o["name"]] = obj
            elif kind == "light":
                apply_light(obj, o["light"], keyer, opt)
            elif kind == "null" and o.get("empty"):
                apply_empty(obj, o["empty"], opt)
            elif kind == "mesh":
                mesh_id = o["mesh"]
                info = meshes[mesh_id]
                digest = f"{info['hash']}@{opt.scale:g}"
                slot_names = o.get("materials", [])
                if get_hash(obj) == digest and obj.GetPointCount() == info["points"]:
                    skipped_geo += 1
                    used = sorted(set(bundle.blob(f"mesh/{mesh_id}/mats.u16", "H")))
                else:
                    used = write_geometry(obj, bundle, mesh_id, info, slot_names, opt.scale, warnings)
                    set_key(obj, key, digest)
                apply_textures(obj, slot_names, used, materials)
            obj.Message(c4d.MSG_UPDATE)

        n_cuts = key_stage(doc, stage, data, cams, keyer, warnings) if stage is not None else 0
        n_markers = add_markers(doc, data, keyer, root_name, warnings) if opt.markers else 0

        stale = [obj for key, obj in index.items() if key not in touched and key.startswith(("obj:", "col:"))]
        if stale:
            warnings.append(f"{plural(len(stale), 'object')} from an earlier import no longer exist in Blender "
                            f"and were left untouched (e.g. “{stale[0].GetName()}”).")
        if any(o.get("camera", {}).get("type") == "ORTHO" for o in objects):
            warnings.append("Orthographic cameras were imported as Parallel. Check their framing against Blender.")
    finally:
        doc.EndUndo()
        c4d.StatusClear()

    parts = []
    if created:
        parts.append(f"{created} new")
    if updated:
        parts.append(f"{updated} updated")
    if skipped_geo:
        parts.append(f"{skipped_geo} meshes unchanged")
    summary = (f"Imported {plural(len(objects), 'object')} ({', '.join(parts)}), {plural(n_cuts, 'cut')}, "
               f"{plural(n_markers, 'marker')} and {plural(keyer.keys, 'key')} into “{root_name}”.")
    if materials.created:
        summary += f" Created {plural(materials.created, 'material')}."
    return summary, warnings


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

ID_PATH, ID_BROWSE = 1001, 1002
ID_SUMMARY, ID_SOURCE, ID_DETAILS = 1010, 1011, 1012
ID_SCALE, ID_OFFSET, ID_SCALE_NOTE = 1020, 1021, 1022
SCALE_NOTE = "cm per Blender unit (100 = real-world size)"
ID_STAGE, ID_MARKERS, ID_MATCH, ID_CLIP = 1030, 1031, 1032, 1033
ID_STATUS, ID_CANCEL, ID_IMPORT = 1040, 1041, 1042

DEFAULTS = {"path": "", "scale": 100.0, "offset": 0, "stage": True, "markers": True,
            "match": True, "clip": True}


def load_prefs():
    bc = c4d.plugins.GetWorldPluginData(PLUGIN_ID)
    prefs = dict(DEFAULTS)
    if bc:
        raw = bc.GetString(1)
        if raw:
            try:
                prefs.update(json.loads(raw))
            except ValueError:
                pass
    return prefs


def save_prefs(prefs):
    bc = c4d.BaseContainer()
    bc.SetString(1, json.dumps(prefs))
    c4d.plugins.SetWorldPluginData(PLUGIN_ID, bc, add=False)


class ImportDialog(gui.GeDialog):
    def __init__(self):
        super().__init__()
        self.bundle = None

    def _section(self, title):
        self.AddStaticText(0, c4d.BFH_LEFT, name=title, borderstyle=c4d.BORDER_WITH_TITLE_BOLD)

    def CreateLayout(self):
        self.SetTitle("Import from Blender")
        self.GroupBegin(0, c4d.BFH_SCALEFIT | c4d.BFV_SCALEFIT, cols=1)
        self.GroupBorderSpace(14, 12, 14, 12)
        self.GroupSpace(0, 6)

        self._section("Blender Export")
        self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2)
        self.GroupSpace(6, 0)
        self.AddEditText(ID_PATH, c4d.BFH_SCALEFIT, initw=340)
        self.AddButton(ID_BROWSE, c4d.BFH_RIGHT, name="Choose…")
        self.GroupEnd()
        self.AddStaticText(ID_SUMMARY, c4d.BFH_SCALEFIT, name="")
        self.AddStaticText(ID_SOURCE, c4d.BFH_SCALEFIT, name="")
        self.AddMultiLineEditText(ID_DETAILS, c4d.BFH_SCALEFIT | c4d.BFV_SCALEFIT, inith=110,
                                  style=c4d.DR_MULTILINE_READONLY | c4d.DR_MULTILINE_MONOSPACED)

        self.AddSeparatorH(0, c4d.BFH_SCALEFIT)

        self._section("Options")
        self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=3)
        self.GroupSpace(8, 4)
        self.AddStaticText(0, c4d.BFH_LEFT, initw=96, name="Scale")
        self.AddEditNumberArrows(ID_SCALE, c4d.BFH_LEFT, initw=90)
        self.AddStaticText(ID_SCALE_NOTE, c4d.BFH_SCALEFIT, name=SCALE_NOTE)
        self.AddStaticText(0, c4d.BFH_LEFT, initw=96, name="Frame offset")
        self.AddEditNumberArrows(ID_OFFSET, c4d.BFH_LEFT, initw=90)
        self.AddStaticText(0, c4d.BFH_SCALEFIT, name="added to every key, cut and marker")
        self.GroupEnd()

        self.AddCheckbox(ID_STAGE, c4d.BFH_LEFT, 0, 0, name="Create Stage with camera cuts")
        self.AddCheckbox(ID_MARKERS, c4d.BFH_LEFT, 0, 0, name="Add timeline markers")
        self.AddCheckbox(ID_MATCH, c4d.BFH_LEFT, 0, 0, name="Match FPS, frame range and resolution")
        self.AddCheckbox(ID_CLIP, c4d.BFH_LEFT, 0, 0, name="Match camera clipping")

        self.AddSeparatorH(0, c4d.BFH_SCALEFIT)

        self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=3)
        self.GroupSpace(6, 0)
        self.AddStaticText(ID_STATUS, c4d.BFH_SCALEFIT, name="")
        self.AddButton(ID_CANCEL, c4d.BFH_RIGHT, initw=90, name="Cancel")
        self.AddButton(ID_IMPORT, c4d.BFH_RIGHT, initw=110, name="Import")
        self.GroupEnd()

        self.GroupEnd()
        return True

    def InitValues(self):
        prefs = load_prefs()
        self.SetString(ID_PATH, prefs["path"])
        try:
            self.SetString(ID_PATH, "Choose a .c4dbridge file exported from Blender", flags=c4d.EDITTEXT_HELPTEXT)
        except (TypeError, AttributeError):
            pass
        self.SetFloat(ID_SCALE, prefs["scale"], min=0.0001, max=1e6, step=1.0, format=c4d.FORMAT_FLOAT)
        self.SetInt32(ID_OFFSET, int(prefs["offset"]), min=-100000, max=100000)
        self.SetBool(ID_STAGE, prefs["stage"])
        self.SetBool(ID_MARKERS, prefs["markers"])
        self.SetBool(ID_MATCH, prefs["match"])
        self.SetBool(ID_CLIP, prefs["clip"])
        self._load(prefs["path"], quiet=True)
        return True

    def _load(self, path, quiet=False):
        if self.bundle is not None:
            self.bundle.close()
        self.bundle = None
        try:
            self.bundle = Bundle(path)
        except BridgeError as exc:
            for gid in (ID_SUMMARY, ID_SOURCE, ID_DETAILS):
                self.SetString(gid, "")
            self.SetString(ID_STATUS, "" if quiet and not path else str(exc))
        else:
            line1, line2, details = describe(self.bundle.data)
            self.SetString(ID_SUMMARY, line1)
            self.SetString(ID_SOURCE, line2)
            self.SetString(ID_DETAILS, details)
            self.SetString(ID_STATUS, "Ready to import.")
            # The scale chosen in Blender's export wins; it can still be changed here before importing.
            blender_scale = self.bundle.data.get("c4d_scale")
            if blender_scale:
                self.SetFloat(ID_SCALE, float(blender_scale), min=0.0001, max=1e6, step=1.0, format=c4d.FORMAT_FLOAT)
            self.SetString(ID_SCALE_NOTE, "cm per Blender unit (set in Blender's export)" if blender_scale
                           else SCALE_NOTE)
        self.Enable(ID_IMPORT, self.bundle is not None)
        self.LayoutChanged(0)

    def _options(self):
        return Options(scale=self.GetFloat(ID_SCALE), frame_offset=self.GetInt32(ID_OFFSET),
                       stage=self.GetBool(ID_STAGE), markers=self.GetBool(ID_MARKERS),
                       match_document=self.GetBool(ID_MATCH), clipping=self.GetBool(ID_CLIP))

    def _save(self):
        save_prefs({"path": self.GetString(ID_PATH), "scale": self.GetFloat(ID_SCALE),
                    "offset": self.GetInt32(ID_OFFSET), "stage": self.GetBool(ID_STAGE),
                    "markers": self.GetBool(ID_MARKERS), "match": self.GetBool(ID_MATCH),
                    "clip": self.GetBool(ID_CLIP)})

    def Command(self, cid, msg):
        if cid == ID_BROWSE:
            current = self.GetString(ID_PATH)
            folder = os.path.dirname(current) if current else ""
            path = c4d.storage.LoadDialog(type=c4d.FILESELECTTYPE_ANYTHING, title="Choose a Blender export",
                                          flags=c4d.FILESELECT_LOAD, def_path=folder)
            if path:
                self.SetString(ID_PATH, path)
                self._load(path)
        elif cid == ID_PATH:
            path = self.GetString(ID_PATH).strip().strip('"')
            if path.lower().endswith((".c4dbridge", ".json")) and os.path.isfile(path):
                self._load(path)
        elif cid == ID_CANCEL:
            self._save()
            self.Close()
        elif cid == ID_IMPORT and self.bundle is not None:
            self._import()
        return True

    def _import(self):
        doc = c4d.documents.GetActiveDocument()
        self.SetString(ID_STATUS, "Importing…")
        try:
            summary, warnings = import_bundle(doc, self.bundle, self._options())
        except BridgeError as exc:
            self.SetString(ID_STATUS, str(exc))
            return
        except Exception as exc:  # show anything unexpected instead of failing silently
            traceback.print_exc()
            gui.MessageDialog(f"Import failed: {exc}\n\nDetails are in the Python console.")
            return
        self._save()
        c4d.EventAdd()
        print(f"[Blender Bridge] {summary}")
        for w in warnings:
            print(f"[Blender Bridge] Note: {w}")
        c4d.StatusSetText(summary)
        self.Close()
        if warnings:
            gui.MessageDialog(summary + "\n\n" + "\n\n".join(f"• {w}" for w in warnings))

    def DestroyWindow(self):
        if self.bundle is not None:
            self.bundle.close()
            self.bundle = None


class BridgeCommand(c4d.plugins.CommandData):
    dialog = None

    def Execute(self, doc):
        if self.dialog is None:
            self.dialog = ImportDialog()
        return self.dialog.Open(dlgtype=c4d.DLG_TYPE_ASYNC, pluginid=PLUGIN_ID, defaultw=560, defaulth=0)

    def RestoreLayout(self, sec_ref):
        if self.dialog is None:
            self.dialog = ImportDialog()
        return self.dialog.Restore(pluginid=PLUGIN_ID, secret=sec_ref)


if __name__ == "__main__":
    c4d.plugins.RegisterCommandPlugin(
        id=PLUGIN_ID,
        str="Import from Blender…",
        info=0,
        icon=c4d.bitmaps.InitResourceBitmap(c4d.Ocamera),
        help="Import scenes, cameras and marker camera cuts exported from Blender",
        dat=BridgeCommand(),
    )
