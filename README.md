# Cinema 4D Bridge — Blender → Cinema 4D

A replacement for Alembic between Blender and Cinema 4D. It brings over the scene hierarchy, meshes,
animation, cameras, lights and marker camera cuts. The cuts become a Stage object.

## Install

Download both zips from the [latest release](../../releases/latest).

**Blender 5.0 / 5.2**: Edit › Preferences › Get Extensions › ⌄ › *Install from Disk…* → `camera_bridge_c4d-2.0.1.zip`

**Cinema 4D 2026**: unzip `CameraBridge_C4D-2.0.1.zip` into your Cinema 4D `plugins` folder (or add its folder under
Preferences › Plugins) and restart. It appears as **Extensions › Import from Blender…**

## Use

1. Blender: **File › Export › Cinema 4D Bridge (.c4dbridge)**
   - *Everything* (default): every object in enabled collections. You can also choose *Selected* or *Cameras Only*.
     Parents always come along, so rigs and controllers stay intact.
   - **Scale**: centimetres per Blender unit in Cinema 4D (100 = real-world size). The importer fills in this
     value automatically, and you can still change it there before importing.
2. Cinema 4D: **Extensions › Import from Blender…** → Choose… → Import.
   - Re-import the same .blend whenever animation changes. Everything updates in place, meshes that
     didn't change are skipped, and tags and materials you set up in Cinema 4D are kept.

## What comes across

| Blender | Cinema 4D |
|---|---|
| Collections (colour tags, render/viewport visibility) | Nested nulls with matching icon colours and visibility |
| Object parenting | Same parenting |
| Per-frame transforms (parents, constraints, drivers, NLA) | PSR keys, lossless reduction, no Euler flips |
| Scale 0 ↔ 1 switches (constant keys) | Step keys on the exact same frames |
| Meshes: modifiers applied, n-gons, UVs, sharp/custom normals | Polygon objects with UVW, Normal and Phong tags |
| Material slots | Polygon selections + texture tags; materials reused by name (new ones get the base colour) |
| Meshes sharing data | Exported once |
| Empties (display type/size) | Nulls with matching display |
| Cameras (lens, sensor fit, shift, DOF, clipping) | Cameras |
| Area/point/spot/sun lights (colour, size incl. scale, cone) | Lights (intensity in lumens = W × 683, approximate) |
| Markers bound to cameras | Stage camera keys |
| All markers | Timeline markers |
| FPS, frame range, resolution | Document + render settings |

**Not yet:** deforming meshes (armature skinning, shape keys), which come in static at the first frame with a warning,
and collection instances (they come in as nulls).

## Versions

- **2.0.1**: Scale setting in the Blender exporter, picked up by the importer. Stage camera keys are now
  created the way Cinema 4D expects (no more console warnings during playback).
- 2.0.0: full scene bridge.
- The 2.1.0 Redshift shader-library experiment is archived in `_archive/2.1.0`.

## Tests

- `tests/blender_scene_v2.py`: builds a shot-like scene headless, exports it and checks it against Blender's own evaluation
- `tests/check_c4d_logic.py`: runs the importer's transform logic against Blender's world matrices, frame by frame
- `tests/blender_export_file.py` + `tests/blender_dump_worlds.py`: the same checks on any real .blend
- `tests/c4d_verify_v2.py` (c4dpy): imports inside Cinema 4D and checks every sampled frame, the Stage and re-imports
