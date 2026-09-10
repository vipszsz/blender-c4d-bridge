# SPDX-License-Identifier: GPL-3.0-or-later
"""Cinema 4D Bridge: export a Blender scene - hierarchy, geometry, animation,
cameras, lights and marker camera cuts - for the Cinema 4D importer.

Every object is baked per frame (parents, constraints, drivers and NLA all come
through), the collection tree becomes nested nulls, and Blender's marker camera
switches become a Stage object.
"""

import os
import time

import bpy
from bpy.props import EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from . import scene_export


def plural(n, word):
    return f"{n} {word if n == 1 else (word + 'es' if word.endswith('sh') else word + 's')}"


class EXPORT_OT_c4d_bridge(bpy.types.Operator, ExportHelper):
    """Export the scene for Cinema 4D: hierarchy, meshes, animation, cameras, lights and camera cuts"""
    bl_idname = "export_scene.c4d_bridge"
    bl_label = "Export for Cinema 4D"
    bl_options = {'PRESET'}

    filename_ext = ".c4dbridge"
    filter_glob: StringProperty(default="*.c4dbridge", options={'HIDDEN'})

    content: EnumProperty(
        name="Include",
        items=(
            ('EVERYTHING', "Everything", "All objects in enabled collections"),
            ('SELECTED', "Selected", "Selected objects and their parents"),
            ('CAMERAS', "Cameras Only", "Cameras bound to markers, with their parent rigs"),
        ),
        default='EVERYTHING',
    )
    range_mode: EnumProperty(
        name="Frames",
        items=(
            ('SCENE', "Scene Range", "Scene start to end frame"),
            ('PREVIEW', "Preview Range", "Timeline preview range, if enabled"),
        ),
        default='SCENE',
    )
    c4d_scale: FloatProperty(
        name="Scale",
        description="Size in Cinema 4D, in centimetres per Blender unit (100 keeps real-world size). "
                    "The Cinema 4D importer picks this up automatically",
        default=100.0, min=0.0001, soft_min=0.01, soft_max=10000.0, step=100, precision=3,
    )

    def invoke(self, context, event):
        if not self.filepath or os.path.basename(self.filepath).startswith("untitled"):
            base = os.path.splitext(bpy.path.basename(bpy.data.filepath))[0] or "untitled"
            self.filepath = base + self.filename_ext
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "content")
        layout.prop(self, "range_mode")
        layout.prop(self, "c4d_scale")

        scene = context.scene
        objects = scene_export.gather_objects(context, self.content)
        kinds = [scene_export.object_kind(ob) for ob in objects]
        start, end = scene_export.frame_range(scene, self.range_mode)
        names = {ob.name for ob in objects}
        cuts = [c for c in scene_export.camera_cuts(scene) if c[1].name in names]
        fps = scene.render.fps / scene.render.fps_base

        col = layout.box().column(align=True)
        col.label(text=f"{plural(len(objects), 'object')} · {plural(kinds.count('mesh'), 'mesh')}",
                  icon='OUTLINER_OB_MESH')
        col.label(text=f"{plural(kinds.count('camera'), 'camera')} · {plural(len(cuts), 'cut')} · "
                       f"{plural(kinds.count('light'), 'light')}", icon='CAMERA_DATA')
        col.label(text=f"Frames {start}–{end} · {fps:g} fps", icon='TIME')
        col.label(text=f"1 Blender unit = {self.c4d_scale:g} cm in Cinema 4D", icon='EMPTY_ARROWS')
        if not objects:
            col.label(text="Nothing to export", icon='ERROR')

    def execute(self, context):
        scene = context.scene
        if not scene_export.gather_objects(context, self.content):
            self.report({'ERROR'}, "Nothing to export. Select objects, or bind cameras to markers (Ctrl+B).")
            return {'CANCELLED'}
        start, end = scene_export.frame_range(scene, self.range_mode)
        if end < start:
            self.report({'ERROR'}, "Frame range is empty")
            return {'CANCELLED'}

        t0 = time.perf_counter()
        stats, warnings = scene_export.export_bridge(context, self.filepath, self.content, self.range_mode,
                                                     c4d_scale=self.c4d_scale)
        elapsed = time.perf_counter() - t0
        for w in warnings:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, f"Exported {plural(stats['objects'], 'object')} ({stats['points']:,} points), "
                              f"{plural(stats['cameras'], 'camera')}, {plural(stats['cuts'], 'cut')}, "
                              f"{stats['frames']} frames in {elapsed:.1f}s → {os.path.basename(self.filepath)} "
                              f"({stats['bytes'] / 1e6:.1f} MB)")
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_c4d_bridge.bl_idname, text="Cinema 4D Bridge (.c4dbridge)")


def register():
    bpy.utils.register_class(EXPORT_OT_c4d_bridge)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(EXPORT_OT_c4d_bridge)
