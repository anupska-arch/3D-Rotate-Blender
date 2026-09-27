import bpy
import gpu
import blf
import math
import mathutils
from mathutils.kdtree import KDTree
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils

bl_info = {
    "name": "3D Rotate",
    "author": "Anupska",
    "version": (1, 0, 3),
    "blender": (4, 0, 0),
    "location": "Object > Transform | Right-Click Menu | N-Panel (Item)",
    "description": "CAD-style 3D Reference Rotation with Scene-Relative Pivot Raycasting.",
    "category": "Object",
}

NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE', 'NDOF_MOTION'
}

TITLES = [
    "Step 1/4: Set Base Point",
    "Step 2/4: Set Rotation Axis [{}]",
    "Step 3/4: Set Reference Angle",
    "Step 4/4: Set Target Angle"
]

DESCS = [
    "Click a vertex or surface to set rotation pivot.",
    "Pick second point or use hotkeys to lock axis.",
    "Click a point to set the 0° start direction.",
    "Click target point OR type numeric angle directly."
]

SHORTCUTS = [
    ["• Left Click: Select Base Point", "• Hold CTRL: Toggle Snapping On/Off", "• Right Click / ESC: Cancel"],
    ["• Left Click: Confirm Axis Direction", "• Press X / Y / Z: Lock Global Axis", "• Press C: Freehand Axis", "• Hold CTRL: Toggle Snapping On/Off", "• Right Click / ESC: Cancel"],
    ["• Left Click: Confirm Start Direction", "• Hold CTRL: Toggle Snapping On/Off", "• Right Click / ESC: Cancel"],
    ["• Left Click: Set Target Point", "• Type Number + Enter: Set Exact Angle (e.g. 45)", "• Press X / Y / Z: Lock Target Alignment", "• Press C: Freehand Target Snap", "• Hold CTRL: Toggle Snapping On/Off", "• Right Click / ESC: Cancel"]
]

GUIDES = [
    "Guide: Red Dot = Base Point",
    "Guides: Red = Base | Magenta Line = Axis",
    "Guides: Red = Base | Magenta = Axis | Cyan = Ref",
    "Guides: Red = Base | Mag = Axis | Cyan = Ref | Yel = Target"
]

addon_keymaps = []
_active_draw_handlers = []

def remove_all_draw_handlers():
    """Defensively removes any orphan GPU draw handlers from memory."""
    global _active_draw_handlers
    for h3d, h2d in list(_active_draw_handlers):
        if h3d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(h3d, 'WINDOW')
            except Exception:
                pass
        if h2d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(h2d, 'WINDOW')
            except Exception:
                pass
    _active_draw_handlers.clear()

def tag_3d_viewports_redraw(context):
    """Forces a redraw across all 3D Viewport areas on screen."""
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def get_3d_region_and_rv3d(context):
    """Safely retrieves active 3D Viewport region and camera data using passed context."""
    if context and context.area and context.area.type == 'VIEW_3D':
        for r in context.area.regions:
            if r.type == 'WINDOW':
                space = context.area.spaces.active
                rv3d = space.region_3d if space else None
                if rv3d:
                    return r, rv3d

    if context and context.screen:
        for a in context.screen.areas:
            if a.type == 'VIEW_3D':
                for r in a.regions:
                    if r.type == 'WINDOW':
                        space = a.spaces.active
                        rv3d = space.region_3d if space else None
                        if rv3d:
                            return r, rv3d
    return None, None

def point_to_segment_2d(p, a, b):
    ab = b - a
    length_sq = ab.length_squared
    if length_sq < 1e-6:
        return a, 0.0
    t = (p - a).dot(ab) / length_sq
    t = max(0.0, min(1.0, t))
    return a + t * ab, t

def get_compass_basis(base_pt, axis_end, ref_pt):
    axis_vec = axis_end - base_pt
    if axis_vec.length_squared == 0:
        return None, None, None, 1.0
    u = axis_vec.normalized()
    v_ref = ref_pt - base_pt
    v_ref_proj = v_ref - (v_ref.dot(u)) * u
    r = v_ref_proj.length
    if r < 1e-4:
        r = 1.0
        e1 = u.cross(mathutils.Vector((0, 0, 1)))
        if e1.length_squared < 1e-4:
            e1 = u.cross(mathutils.Vector((1, 0, 0)))
        e1.normalize()
    else:
        e1 = v_ref_proj / r
    e2 = u.cross(e1)
    return u, e1, e2, r

# --- NATIVE BLENDER 2D SHAPE DRAWING HELPERS ---
def draw_snap_marker_circle(x, y, radius=7, color=(1.0, 0.6, 0.0, 1.0)):
    verts = []
    num_segs = 16
    for i in range(num_segs):
        a1 = (2.0 * math.pi * i) / num_segs
        a2 = (2.0 * math.pi * (i + 1)) / num_segs
        verts.extend([
            (x + radius * math.cos(a1), y + radius * math.sin(a1)),
            (x + radius * math.cos(a2), y + radius * math.sin(a2))
        ])
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    batch.draw(shader)

def draw_snap_marker_triangle(x, y, size=8, color=(0.2, 0.9, 1.0, 1.0)):
    top = (x, y + size)
    left = (x - size * 0.866, y - size * 0.5)
    right = (x + size * 0.866, y - size * 0.5)
    verts = [top, left, left, right, right, top]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    batch.draw(shader)

def draw_snap_marker_square(x, y, size=7, color=(1.0, 0.8, 0.2, 1.0)):
    p1 = (x - size, y + size)
    p2 = (x + size, y + size)
    p3 = (x + size, y - size)
    p4 = (x - size, y - size)
    verts = [p1, p2, p2, p3, p3, p4, p4, p1]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    batch.draw(shader)

def draw_snap_marker_perp(x, y, size=8, color=(1.0, 0.3, 0.9, 1.0)):
    verts = [
        (x - size, y - size), (x + size, y - size),
        (x, y - size), (x, y + size),
        (x, y), (x + size * 0.5, y)
    ]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    batch.draw(shader)

def draw_hud_box(x, y, width, height, color=(0.08, 0.10, 0.14, 0.88)):
    vertices = [(x, y), (x + width, y), (x + width, y - height), (x, y - height)]
    indices = [(0, 1, 2), (0, 2, 3)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    batch.draw(shader)

def draw_hud_border(x, y, width, height, color=(0.3, 0.4, 0.55, 0.9)):
    vertices = [(x, y), (x + width, y), (x + width, y - height), (x, y - height)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(1.5)
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
    batch.draw(shader)

def draw_callback_3d(self, context):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    gpu.state.point_size_set(12.0)

    if self.base_pt:
        shader.uniform_float("color", (1.0, 0.2, 0.2, 1.0))
        batch = batch_for_shader(shader, 'POINTS', {"pos": [self.base_pt]})
        batch.draw(shader)

    axis_end_pt = self.axis_end if self.axis_end else (self.hover_pt if self.step == 1 else None)
    if self.base_pt and axis_end_pt:
        gpu.state.line_width_set(3.0)
        shader.uniform_float("color", (1.0, 0.0, 1.0, 1.0))
        batch_axis = batch_for_shader(shader, 'LINE_STRIP', {"pos": [self.base_pt, axis_end_pt]})
        batch_axis.draw(shader)
        batch_pt = batch_for_shader(shader, 'POINTS', {"pos": [axis_end_pt]})
        batch_pt.draw(shader)

    if self.base_pt and self.ref_pt:
        gpu.state.line_width_set(3.0)
        shader.uniform_float("color", (0.2, 0.8, 1.0, 1.0))
        batch_line = batch_for_shader(shader, 'LINE_STRIP', {"pos": [self.base_pt, self.ref_pt]})
        batch_line.draw(shader)
        batch_pt = batch_for_shader(shader, 'POINTS', {"pos": [self.ref_pt]})
        batch_pt.draw(shader)

    if self.base_pt and self.step == 3:
        tgt_pt = self.target_pt if self.target_pt else self.hover_pt
        if tgt_pt:
            gpu.state.line_width_set(3.0)
            shader.uniform_float("color", (1.0, 0.9, 0.2, 0.9))
            batch_tgt = batch_for_shader(shader, 'LINE_STRIP', {"pos": [self.base_pt, tgt_pt]})
            batch_tgt.draw(shader)
            batch_tgt_pt = batch_for_shader(shader, 'POINTS', {"pos": [tgt_pt]})
            batch_tgt_pt.draw(shader)

            u, e1, e2, R = get_compass_basis(self.base_pt, self.axis_end, self.ref_pt)
            if u:
                ring_verts = []
                num_segs = 64
                for i in range(num_segs):
                    a1 = (2.0 * math.pi * i) / num_segs
                    a2 = (2.0 * math.pi * (i + 1)) / num_segs
                    p1 = self.base_pt + R * (math.cos(a1) * e1 + math.sin(a1) * e2)
                    p2 = self.base_pt + R * (math.cos(a2) * e1 + math.sin(a2) * e2)
                    ring_verts.extend([p1, p2])
                
                gpu.state.line_width_set(1.5)
                shader.uniform_float("color", (0.3, 0.7, 1.0, 0.45))
                batch_ring = batch_for_shader(shader, 'LINES', {"pos": ring_verts})
                batch_ring.draw(shader)

                tick_verts = []
                for deg in range(0, 360, 15):
                    rad = math.radians(deg)
                    t_len = R * 0.15 if (deg % 90 == 0) else (R * 0.08 if deg % 45 == 0 else R * 0.04)
                    p_in = self.base_pt + (R - t_len) * (math.cos(rad) * e1 + math.sin(rad) * e2)
                    p_out = self.base_pt + (R + t_len) * (math.cos(rad) * e1 + math.sin(rad) * e2)
                    tick_verts.extend([p_in, p_out])

                shader.uniform_float("color", (0.4, 0.8, 1.0, 0.75))
                batch_ticks = batch_for_shader(shader, 'LINES', {"pos": tick_verts})
                batch_ticks.draw(shader)

                angle_rad = math.radians(self.current_angle_deg)
                if abs(angle_rad) > 1e-4:
                    arc_segs = max(1, int(abs(angle_rad) / (math.pi / 36)))
                    fan_verts = []
                    for i in range(arc_segs):
                        a1 = angle_rad * (i / arc_segs)
                        a2 = angle_rad * ((i + 1) / arc_segs)
                        p1 = self.base_pt + R * (math.cos(a1) * e1 + math.sin(a1) * e2)
                        p2 = self.base_pt + R * (math.cos(a2) * e1 + math.sin(a2) * e2)
                        fan_verts.extend([self.base_pt, p1, p2])

                    shader.uniform_float("color", (1.0, 0.85, 0.2, 0.22))
                    batch_fan = batch_for_shader(shader, 'TRIS', {"pos": fan_verts})
                    batch_fan.draw(shader)

def draw_callback_2d(self, context):
    region, rv3d = get_3d_region_and_rv3d(context)
    if not region or not rv3d:
        return

    font_id = 0
    blf.size(font_id, 13)

    curr_title = TITLES[self.step].format(self.axis_mode) if "{}" in TITLES[self.step] else TITLES[self.step]
    curr_desc = DESCS[self.step]
    curr_scs = SHORTCUTS[self.step]
    curr_guide = GUIDES[self.step]

    line_count = 3 + len(curr_scs) + (1 if self.step == 3 else 0)
    box_w, box_h = 370, 35 + (line_count * 18)
    box_x, box_y = 20, 20 + box_h

    draw_hud_box(box_x, box_y, box_w, box_h)
    draw_hud_border(box_x, box_y, box_w, box_h)

    line_y = box_y - 22
    blf.size(font_id, 14)
    blf.position(font_id, box_x + 15, line_y, 0)
    blf.color(font_id, 1.0, 0.85, 0.2, 1.0)
    blf.draw(font_id, f"3D Rotate  [{curr_title}]")

    line_y -= 20
    blf.size(font_id, 12)
    blf.position(font_id, box_x + 15, line_y, 0)
    blf.color(font_id, 0.3, 0.85, 1.0, 1.0)
    blf.draw(font_id, curr_desc)

    if self.step == 3:
        line_y -= 18
        blf.position(font_id, box_x + 15, line_y, 0)
        if self.typed_angle:
            blf.color(font_id, 1.0, 0.5, 0.0, 1.0)
            blf.draw(font_id, f"Typed Angle: [ {self.typed_angle}° ]  (Press ENTER)")
        else:
            blf.color(font_id, 1.0, 0.9, 0.2, 1.0)
            blf.draw(font_id, f"Live Angle: {self.current_angle_deg:.2f}°")

    blf.color(font_id, 0.85, 0.85, 0.85, 1.0)
    for sc in curr_scs:
        line_y -= 18
        blf.position(font_id, box_x + 15, line_y, 0)
        blf.draw(font_id, sc)

    line_y -= 20
    blf.position(font_id, box_x + 15, line_y, 0)
    blf.color(font_id, 0.6, 0.65, 0.7, 1.0)
    blf.draw(font_id, curr_guide)

    def draw_label(pt, text, color=(1.0, 1.0, 1.0, 1.0), offset_x=14, offset_y=14):
        p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, pt)
        if p2d:
            blf.position(font_id, p2d.x + offset_x, p2d.y + offset_y, 0)
            blf.color(font_id, *color)
            blf.draw(font_id, text)

    if self.base_pt:
        draw_label(self.base_pt, "[BASE]", (1.0, 0.3, 0.3, 1.0), offset_y=-16)

    if self.axis_end:
        if self.step == 1:
            mode_str = f"[AXIS: {self.axis_mode}]"
            draw_label(self.axis_end, mode_str, (1.0, 0.3, 1.0, 1.0), offset_y=22)
        else:
            draw_label(self.axis_end, "[AXIS END]", (1.0, 0.3, 1.0, 1.0), offset_y=16)

    if self.ref_pt:
        draw_label(self.ref_pt, "[REF (0°)]", (0.3, 0.8, 1.0, 1.0), offset_y=16)

    if self.step == 3 and self.base_pt and self.axis_end and self.ref_pt:
        u, e1, e2, R = get_compass_basis(self.base_pt, self.axis_end, self.ref_pt)
        if u:
            cardinals = [(0, "0°"), (90, "90°"), (180, "180°"), (270, "-90°")]
            for deg, text in cardinals:
                rad = math.radians(deg)
                pos_3d = self.base_pt + (R * 1.12) * (math.cos(rad) * e1 + math.sin(rad) * e2)
                draw_label(pos_3d, text, (0.4, 0.85, 1.0, 0.85), offset_y=2)

    if self.step == 3:
        tgt_pt = self.target_pt if self.target_pt else self.hover_pt
        if tgt_pt:
            lock_str = f" [LOCK: {self.target_axis_mode}]" if self.target_axis_mode != 'NONE' else ""
            angle_str = f" ({self.typed_angle}°)" if self.typed_angle else f" ({self.current_angle_deg:.2f}°)"
            draw_label(tgt_pt, f"[TARGET{lock_str}]{angle_str}", (1.0, 0.9, 0.2, 1.0), offset_y=2)
            
    # --- RENDER NATIVE 2D SNAP MARKER SHAPES ---
    if self.hover_pt:
        p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, self.hover_pt)
        if p2d:
            if self.is_snapped:
                if self.snap_type == "Vertex":
                    draw_snap_marker_circle(p2d.x, p2d.y, radius=7, color=(1.0, 0.6, 0.0, 1.0))
                elif self.snap_type == "Edge Center":
                    draw_snap_marker_triangle(p2d.x, p2d.y, size=8, color=(0.2, 0.9, 1.0, 1.0))
                elif self.snap_type == "Edge Perpendicular":
                    draw_snap_marker_perp(p2d.x, p2d.y, size=8, color=(1.0, 0.3, 0.9, 1.0))
                else:
                    draw_snap_marker_square(p2d.x, p2d.y, size=6, color=(1.0, 0.85, 0.2, 1.0))
                
                blf.position(font_id, p2d.x + 14, p2d.y + 2, 0)
                blf.color(font_id, 0.0, 1.0, 0.4, 1.0)
                blf.draw(font_id, f"SNAP [{self.snap_type}]")
            else:
                col = (0.3, 0.85, 1.0, 1.0) if getattr(self, 'snap_enabled', False) else (1.0, 0.8, 0.2, 0.9)
                status = "Cursor (Snapping Active)" if getattr(self, 'snap_enabled', False) else "Cursor (Snap OFF - Hold CTRL)"
                blf.position(font_id, p2d.x + 14, p2d.y + 2, 0)
                blf.color(font_id, *col)
                blf.draw(font_id, f"<- {status}")

class OBJECT_OT_rhino_3d_rotate_live(bpy.types.Operator):
    """CAD-style reference 3D rotation tool."""
    bl_idname = "object.rhino_3d_rotate_live"
    bl_label = "3D Rotate"
    bl_description = "CAD-style reference 3D rotation tool."
    bl_options = {'REGISTER', 'UNDO'}

    SNAP_PIXEL_THRESHOLD = 28.0

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(context.selected_objects)

    def execute(self, context):
        self.report({'WARNING'}, "3D Rotate is an interactive tool and must be invoked from a 3D Viewport.")
        return {'CANCELLED'}

    def get_vertex_kdtree(self, obj):
        if obj.name in self._kd_v_cache:
            return self._kd_v_cache[obj.name]
        mesh = obj.data
        if not mesh or len(mesh.vertices) == 0:
            return None
        mw = obj.matrix_world
        kd_v = KDTree(len(mesh.vertices))
        for i, v in enumerate(mesh.vertices):
            kd_v.insert(mw @ v.co, i)
        kd_v.balance()
        self._kd_v_cache[obj.name] = kd_v
        return kd_v

    def get_edge_kdtree(self, obj):
        if obj.name in self._kd_e_cache:
            return self._kd_e_cache[obj.name]
        mesh = obj.data
        if not mesh or len(mesh.edges) == 0:
            return None
        mw = obj.matrix_world
        kd_e = KDTree(len(mesh.edges))
        verts = mesh.vertices
        for i, e in enumerate(mesh.edges):
            mid = mw @ ((verts[e.vertices[0]].co + verts[e.vertices[1]].co) * 0.5)
            kd_e.insert(mid, i)
        kd_e.balance()
        self._kd_e_cache[obj.name] = kd_e
        return kd_e

    def get_mouse_snap_target(self, context, event, base_pt=None):
        region, rv3d = get_3d_region_and_rv3d(context)
        if not region or not rv3d:
            return mathutils.Vector((0, 0, 0)), False, "Off", False

        ts = context.tool_settings
        snap_enabled = ts.use_snap if not (event and event.ctrl) else not ts.use_snap

        mouse_x = event.mouse_x if event else region.x + region.width // 2
        mouse_y = event.mouse_y if event else region.y + region.height // 2
        mouse_2d = mathutils.Vector((mouse_x - region.x, mouse_y - region.y))
        
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
        ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d)
        
        depsgraph = context.evaluated_depsgraph_get()
        result, location, normal, index, hit_obj, matrix = context.scene.ray_cast(depsgraph, ray_origin, ray_target)
        
        is_rotating = (self.step == 3)
        is_selected_hit = (result and hit_obj in context.selected_objects) if is_rotating else False

        # --- SCENE-RELATIVE FALLBACK PLANE INTERSECTION ---
        # Anchors raycast miss fallback plane to base_pt (if set) or viewport pivot (rv3d.view_location)
        pivot_pt = base_pt if base_pt else (rv3d.view_location if rv3d else mathutils.Vector((0, 0, 0)))

        if result and not is_selected_hit:
            unsnapped_pt = location
        else:
            if abs(ray_target.z) > 0.05:
                t = (pivot_pt.z - ray_origin.z) / ray_target.z
                unsnapped_pt = ray_origin + t * ray_target
            elif rv3d:
                view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
                denom = ray_target.dot(view_dir)
                if abs(denom) > 1e-4:
                    t = (pivot_pt - ray_origin).dot(view_dir) / denom
                    unsnapped_pt = ray_origin + t * ray_target
                else:
                    unsnapped_pt = ray_origin + ray_target * (pivot_pt - ray_origin).length
            else:
                unsnapped_pt = ray_origin + ray_target * 5.0

        if not snap_enabled:
            return unsnapped_pt, False, "Off", False

        elements = ts.snap_elements if hasattr(ts, "snap_elements") else {'VERTEX'}
        elem_set = {str(e).upper() for e in elements}

        has_vertex = 'VERTEX' in elem_set
        has_edge_center = 'EDGE_MIDPOINT' in elem_set
        has_face_center = any(k in elem_set for k in ('FACE_PROJECT', 'FACE_NEAREST', 'FACE', 'FACE_MIDPOINT'))
        has_edge_perp = 'EDGE_PERPENDICULAR' in elem_set
        has_edge = 'EDGE' in elem_set
        has_face = any(k in elem_set for k in ('FACE', 'FACE_PROJECT', 'FACE_NEAREST', 'FACE_MIDPOINT'))
        has_inc = 'INCREMENT' in elem_set

        min_pixel_dist = self.SNAP_PIXEL_THRESHOLD

        if result and hit_obj and hit_obj.type == 'MESH' and not is_selected_hit:
            mw = hit_obj.matrix_world
            eval_obj = hit_obj.evaluated_get(depsgraph)
            mesh = eval_obj.data if eval_obj else hit_obj.data
            verts = mesh.vertices
            num_verts = len(verts)

            if 0 <= index < len(mesh.polygons):
                poly = mesh.polygons[index]

                if has_vertex:
                    best_dist, best_pt = min_pixel_dist, None
                    for v_idx in poly.vertices:
                        if v_idx < num_verts:
                            v_world = mw @ verts[v_idx].co
                            v_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v_world)
                            if v_2d:
                                dist = (v_2d - mouse_2d).length
                                if dist < best_dist:
                                    best_dist, best_pt = dist, v_world
                    if best_pt:
                        return best_pt, True, "Vertex", True

                if has_edge_center:
                    best_dist, best_pt = min_pixel_dist, None
                    for e_idx in poly.edge_keys:
                        if e_idx[0] < num_verts and e_idx[1] < num_verts:
                            mid_3d = mw @ ((verts[e_idx[0]].co + verts[e_idx[1]].co) * 0.5)
                            mid_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, mid_3d)
                            if mid_2d:
                                dist = (mid_2d - mouse_2d).length
                                if dist < best_dist:
                                    best_dist, best_pt = dist, mid_3d
                    if best_pt:
                        return best_pt, True, "Edge Center", True

                if has_face_center:
                    fc_3d = mw @ poly.center
                    fc_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, fc_3d)
                    if fc_2d and (fc_2d - mouse_2d).length < min_pixel_dist:
                        return fc_3d, True, "Face Center", True

                if has_edge_perp and base_pt:
                    best_dist, best_pt = min_pixel_dist, None
                    for e_idx in poly.edge_keys:
                        if e_idx[0] < num_verts and e_idx[1] < num_verts:
                            v1_world = mw @ verts[e_idx[0]].co
                            v2_world = mw @ verts[e_idx[1]].co
                            edge_vec = v2_world - v1_world
                            if edge_vec.length_squared > 1e-8:
                                u_edge = edge_vec.normalized()
                                t = (base_pt - v1_world).dot(u_edge)
                                perp_3d = v1_world + t * u_edge
                                perp_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, perp_3d)
                                if perp_2d:
                                    dist = (perp_2d - mouse_2d).length
                                    if dist < best_dist:
                                        best_dist, best_pt = dist, perp_3d
                    if best_pt:
                        return best_pt, True, "Edge Perpendicular", True

                if has_edge:
                    best_dist, best_pt = min_pixel_dist, None
                    for e_idx in poly.edge_keys:
                        if e_idx[0] < num_verts and e_idx[1] < num_verts:
                            v1_world = mw @ verts[e_idx[0]].co
                            v2_world = mw @ verts[e_idx[1]].co
                            p1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v1_world)
                            p2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v2_world)
                            if p1_2d and p2_2d:
                                proj_2d, t = point_to_segment_2d(mouse_2d, p1_2d, p2_2d)
                                dist = (proj_2d - mouse_2d).length
                                if dist < best_dist:
                                    best_dist, best_pt = dist, v1_world + t * (v2_world - v1_world)
                    if best_pt:
                        return best_pt, True, "Edge", True

        max_dist_sq = (getattr(self, 'max_ray_distance', 1000.0)) ** 2
        
        if is_rotating:
            raw_candidates = (
                o for o in context.visible_objects 
                if o.type == 'MESH' and not o.hide_viewport and o not in context.selected_objects
            )
        else:
            raw_candidates = (
                o for o in context.visible_objects 
                if o.type == 'MESH' and not o.hide_viewport
            )
        
        sorted_candidates = sorted(
            raw_candidates,
            key=lambda o: (o.matrix_world.translation - unsnapped_pt).length_squared
        )

        candidates = [
            o for o in sorted_candidates 
            if (o.matrix_world.translation - unsnapped_pt).length_squared <= max_dist_sq
        ]

        for c_obj in candidates[:3]:
            if has_vertex:
                kd_v = self.get_vertex_kdtree(c_obj)
                if kd_v:
                    near_verts = kd_v.find_n(unsnapped_pt, 5)
                    best_dist, best_pt = min_pixel_dist, None
                    for co, index_v, d_3d in near_verts:
                        v_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co)
                        if v_2d:
                            dist = (v_2d - mouse_2d).length
                            if dist < best_dist:
                                best_dist, best_pt = dist, co
                    if best_pt:
                        return best_pt, True, "Vertex", True

            if has_edge_center:
                kd_e = self.get_edge_kdtree(c_obj)
                if kd_e:
                    near_edges = kd_e.find_n(unsnapped_pt, 5)
                    best_dist, best_pt = min_pixel_dist, None
                    for co, index_e, d_3d in near_edges:
                        e_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co)
                        if e_2d:
                            dist = (e_2d - mouse_2d).length
                            if dist < best_dist:
                                best_dist, best_pt = dist, co
                    if best_pt:
                        return best_pt, True, "Edge Center", True

        if has_face and result and not is_selected_hit:
            return location, True, "Face", True

        if has_inc:
            return mathutils.Vector((round(unsnapped_pt.x), round(unsnapped_pt.y), round(unsnapped_pt.z))), True, "Grid Increment", True

        if result and not is_selected_hit:
            return location, True, "Surface", True

        return unsnapped_pt, False, "Off", True

    def update_axis_end_position(self, context, event):
        if self.step != 1 or not self.base_pt:
            return

        if self.axis_mode == 'CUSTOM':
            self.axis_end = self.hover_pt
        else:
            dirs = {'X': mathutils.Vector((1,0,0)), 'Y': mathutils.Vector((0,1,0)), 'Z': mathutils.Vector((0,0,1))}
            u = dirs[self.axis_mode]
            
            if self.is_snapped:
                v = self.hover_pt - self.base_pt
                dist = v.dot(u)
                if abs(dist) < 0.1:
                    dist = 2.0 if dist >= 0 else -2.0
                self.axis_end = self.base_pt + u * dist
            else:
                region, rv3d = get_3d_region_and_rv3d(context)
                if region and rv3d and event:
                    mouse_2d = mathutils.Vector((event.mouse_x - region.x, event.mouse_y - region.y))
                    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
                    ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d).normalized()
                    
                    A = self.base_pt
                    B = ray_origin
                    v = ray_target
                    w0 = A - B
                    
                    b = u.dot(v)
                    denom = 1.0 - b * b
                    max_dist = getattr(self, 'max_ray_distance', 1000.0)
                    if abs(denom) > 1e-4:
                        d = u.dot(w0)
                        e = v.dot(w0)
                        t = (b * e - d) / denom
                        t = max(-max_dist, min(max_dist, t))
                    else:
                        t = 2.0
                        
                    if abs(t) < 0.1:
                        t = 2.0 if t >= 0 else -2.0
                        
                    self.axis_end = self.base_pt + u * t
                else:
                    self.axis_end = self.base_pt + u * 2.0

    def update_target_position(self, context, event):
        if not self.base_pt or not self.axis_end:
            self.target_pt = self.hover_pt
            return

        axis_vec = self.axis_end - self.base_pt
        if axis_vec.length_squared < 1e-8:
            self.target_pt = self.hover_pt
            return

        u_axis = axis_vec.normalized()
        max_dist = getattr(self, 'max_ray_distance', 1000.0)

        if self.target_axis_mode != 'NONE':
            dirs = {'X': mathutils.Vector((1,0,0)), 'Y': mathutils.Vector((0,1,0)), 'Z': mathutils.Vector((0,0,1))}
            u_lock = dirs[self.target_axis_mode]

            if self.is_snapped:
                v = self.hover_pt - self.base_pt
                dist = v.dot(u_lock)
                if abs(dist) < 1e-4:
                    dist = 1.0
                self.target_pt = self.base_pt + u_lock * dist
            else:
                region, rv3d = get_3d_region_and_rv3d(context)
                if region and rv3d and event:
                    mouse_2d = mathutils.Vector((event.mouse_x - region.x, event.mouse_y - region.y))
                    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
                    ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d).normalized()
                    
                    A = self.base_pt
                    B = ray_origin
                    v = ray_target
                    w0 = A - B
                    
                    b = u_lock.dot(v)
                    denom = 1.0 - b * b
                    if abs(denom) > 1e-4:
                        d = u_lock.dot(w0)
                        e = v.dot(w0)
                        t = (b * e - d) / denom
                        t = max(-max_dist, min(max_dist, t))
                    else:
                        t = 1.0
                        
                    if abs(t) < 1e-4:
                        t = 1.0
                        
                    self.target_pt = self.base_pt + u_lock * t
                else:
                    self.target_pt = self.base_pt + u_lock * 1.0
            return

        if self.is_snapped:
            self.target_pt = self.hover_pt
            return

        region, rv3d = get_3d_region_and_rv3d(context)
        if region and rv3d and event:
            mouse_2d = mathutils.Vector((event.mouse_x - region.x, event.mouse_y - region.y))
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
            ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d).normalized()

            denom = ray_target.dot(u_axis)
            if abs(denom) > 1e-4:
                t = (self.base_pt - ray_origin).dot(u_axis) / denom
                t = max(-max_dist, min(max_dist, t))
                self.target_pt = ray_origin + t * ray_target
            else:
                self.target_pt = self.hover_pt
        else:
            self.target_pt = self.hover_pt

    def calculate_angle_rad(self, current_target_pt):
        if not current_target_pt or not self.base_pt or not self.axis_end or not self.ref_pt:
            return 0.0

        axis_vec = self.axis_end - self.base_pt
        if axis_vec.length_squared == 0:
            return 0.0
        u = axis_vec.normalized()

        v_ref = self.ref_pt - self.base_pt
        v_tgt = current_target_pt - self.base_pt

        v_ref_proj = v_ref - (v_ref.dot(u)) * u
        v_tgt_proj = v_tgt - (v_tgt.dot(u)) * u

        if v_ref_proj.length_squared < 1e-8 or v_tgt_proj.length_squared < 1e-8:
            return 0.0

        v_ref_proj.normalize()
        v_tgt_proj.normalize()

        cos_a = max(-1.0, min(1.0, v_ref_proj.dot(v_tgt_proj)))
        sin_a = (v_ref_proj.cross(v_tgt_proj)).dot(u)
        return math.atan2(sin_a, cos_a)

    def apply_rotation_matrix(self, context, angle_rad):
        axis_vec = self.axis_end - self.base_pt
        if axis_vec.length_squared == 0:
            return
        u = axis_vec.normalized()

        rot_mat = mathutils.Matrix.Rotation(angle_rad, 4, u)
        pivot = self.base_pt

        trans_mat = mathutils.Matrix.Translation(pivot)
        trans_inv = mathutils.Matrix.Translation(-pivot)

        M = trans_mat @ rot_mat @ trans_inv
        for obj, init_matrix in self.initial_matrices.items():
            obj.matrix_world = M @ init_matrix

    def modal(self, context, event):
        if event.type in NAV_EVENTS:
            return {'PASS_THROUGH'}

        tag_3d_viewports_redraw(context)

        if getattr(self, "activated_from_menu", False):
            dist_moved = math.hypot(event.mouse_x - self.init_mouse_pos[0], event.mouse_y - self.init_mouse_pos[1])
            if dist_moved > 5:
                self.activated_from_menu = False
            elif event.type == 'LEFTMOUSE':
                return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' or event.type in {'LEFT_CTRL', 'RIGHT_CTRL'}:
            self.hover_pt, self.is_snapped, self.snap_type, self.snap_enabled = self.get_mouse_snap_target(context, event, getattr(self, 'base_pt', None))
            if self.step == 1:
                self.update_axis_end_position(context, event)
            elif self.step == 3 and not self.typed_angle:
                self.update_target_position(context, event)
                angle_rad = self.calculate_angle_rad(self.target_pt)
                self.current_angle_deg = math.degrees(angle_rad)
                self.apply_rotation_matrix(context, angle_rad)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.step == 0:
                self.base_pt = self.hover_pt
                self.step = 1
                self.axis_mode = 'CUSTOM'
                self.update_axis_end_position(context, event)
                
            elif self.step == 1:
                self.update_axis_end_position(context, event)
                if (self.axis_end - self.base_pt).length_squared < 0.0001:
                    self.report({'WARNING'}, "Axis End must be different from Axis Start.")
                    return {'RUNNING_MODAL'}
                self.step = 2
                
            elif self.step == 2:
                self.ref_pt = self.hover_pt
                self.step = 3
                self.target_axis_mode = 'NONE'
                self.update_target_position(context, event)
                angle_rad = self.calculate_angle_rad(self.target_pt)
                self.current_angle_deg = math.degrees(angle_rad)

            elif self.step == 3:
                if self.typed_angle:
                    try:
                        val = float(self.typed_angle)
                        self.apply_rotation_matrix(context, math.radians(val))
                    except ValueError:
                        self.report({'WARNING'}, f"Invalid angle: '{self.typed_angle}'. Enter a valid number.")
                        return {'RUNNING_MODAL'}
                else:
                    self.update_target_position(context, event)
                    angle_rad = self.calculate_angle_rad(self.target_pt)
                    self.apply_rotation_matrix(context, angle_rad)

                self.cleanup(context)
                self.report({'INFO'}, "3D Rotate complete.")
                return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.restore_initial_transforms()
            self.cleanup(context)
            self.report({'INFO'}, "3D Rotate cancelled.")
            return {'CANCELLED'}

        elif event.type in {'X', 'Y', 'Z', 'C'} and event.value == 'PRESS':
            key = event.type
            if self.step == 1:
                self.axis_mode = 'CUSTOM' if (key == 'C' or self.axis_mode == key) else key
                self.update_axis_end_position(context, event)
            elif self.step == 3:
                self.target_axis_mode = 'NONE' if (key == 'C' or self.target_axis_mode == key) else key
                if not self.typed_angle:
                    self.update_target_position(context, event)
                    angle_rad = self.calculate_angle_rad(self.target_pt)
                    self.current_angle_deg = math.degrees(angle_rad)
                    self.apply_rotation_matrix(context, angle_rad)

        elif self.step == 3 and event.value == 'PRESS':
            char_map = {
                'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
                'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9',
                'NUMPAD_0': '0', 'NUMPAD_1': '1', 'NUMPAD_2': '2', 'NUMPAD_3': '3',
                'NUMPAD_4': '4', 'NUMPAD_5': '5', 'NUMPAD_6': '6', 'NUMPAD_7': '7',
                'NUMPAD_8': '8', 'NUMPAD_9': '9', 'PERIOD': '.', 'NUMPAD_PERIOD': '.',
                'MINUS': '-', 'NUMPAD_MINUS': '-'
            }
            if event.type in char_map:
                char = char_map[event.type]
                if not (char == '-' and len(self.typed_angle) > 0) and not (char == '.' and '.' in self.typed_angle):
                    self.typed_angle += char
                    try:
                        val = float(self.typed_angle)
                        self.current_angle_deg = val
                        self.apply_rotation_matrix(context, math.radians(val))
                    except ValueError:
                        pass
                return {'RUNNING_MODAL'}

            elif event.type == 'BACK_SPACE':
                if len(self.typed_angle) > 0:
                    self.typed_angle = self.typed_angle[:-1]
                    if self.typed_angle:
                        try:
                            val = float(self.typed_angle)
                            self.current_angle_deg = val
                            self.apply_rotation_matrix(context, math.radians(val))
                        except ValueError:
                            pass
                    else:
                        self.update_target_position(context, event)
                        angle_rad = self.calculate_angle_rad(self.target_pt)
                        self.current_angle_deg = math.degrees(angle_rad)
                        self.apply_rotation_matrix(context, angle_rad)
                return {'RUNNING_MODAL'}

            elif event.type in {'RET', 'NUMPAD_ENTER'}:
                if self.typed_angle:
                    try:
                        val = float(self.typed_angle)
                        self.apply_rotation_matrix(context, math.radians(val))
                        self.cleanup(context)
                        self.report({'INFO'}, f"3D Rotate complete ({val}°).")
                        return {'FINISHED'}
                    except ValueError:
                        self.report({'WARNING'}, f"Invalid angle input: '{self.typed_angle}'. Enter a valid number.")
                        return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if getattr(self, "_is_running", False):
            return {'CANCELLED'}
        self._is_running = True

        self.initial_matrices = {obj: obj.matrix_world.copy() for obj in context.selected_objects}

        self.init_mouse_pos = (event.mouse_x, event.mouse_y) if event else (0, 0)
        self.activated_from_menu = True

        self.step = 0
        self.axis_mode = 'CUSTOM'
        self.target_axis_mode = 'NONE'
        self.typed_angle = ""
        self.current_angle_deg = 0.0
        self.base_pt = None
        self.axis_end = None
        self.ref_pt = None
        self.target_pt = None
        
        max_dist = 1000.0
        if context.visible_objects:
            try:
                bbs = [(o.matrix_world.translation.length + o.dimensions.length) for o in context.visible_objects if o.type == 'MESH']
                if bbs:
                    max_dist = max(max_dist, max(bbs) * 3.0)
            except Exception:
                pass
        self.max_ray_distance = max_dist

        self._kd_v_cache = {}
        self._kd_e_cache = {}

        self.hover_pt, self.is_snapped, self.snap_type, self.snap_enabled = self.get_mouse_snap_target(context, event)
        if self.hover_pt is None:
            self.hover_pt = mathutils.Vector((0, 0, 0))

        args = (self, context)
        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d, args, 'WINDOW', 'POST_VIEW')
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_2d, args, 'WINDOW', 'POST_PIXEL')
        
        global _active_draw_handlers
        _active_draw_handlers.append((self._handle_3d, self._handle_2d))

        context.window_manager.modal_handler_add(self)
        tag_3d_viewports_redraw(context)
        return {'RUNNING_MODAL'}

    def restore_initial_transforms(self):
        for obj, matrix in self.initial_matrices.items():
            obj.matrix_world = matrix.copy()

    def cleanup(self, context):
        global _active_draw_handlers
        pair = (getattr(self, "_handle_3d", None), getattr(self, "_handle_2d", None))
        if pair in _active_draw_handlers:
            _active_draw_handlers.remove(pair)

        if getattr(self, "_handle_3d", None) is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
            except Exception:
                pass
            self._handle_3d = None

        if getattr(self, "_handle_2d", None) is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
            except Exception:
                pass
            self._handle_2d = None
        
        self._is_running = False
        self._kd_v_cache.clear()
        self._kd_e_cache.clear()
        tag_3d_viewports_redraw(context)

class VIEW3D_PT_3d_rotate_panel(bpy.types.Panel):
    bl_label = "CAD Transforms"
    bl_idname = "VIEW3D_PT_3d_rotate_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Item'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("object.rhino_3d_rotate_live", text="3D Rotate", icon='ORIENTATION_GIMBAL')

def menu_func_transform(self, context):
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator("object.rhino_3d_rotate_live", text="3D Rotate", icon='ORIENTATION_GIMBAL')

def menu_func_context(self, context):
    self.layout.separator()
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator("object.rhino_3d_rotate_live", text="3D Rotate (CAD)", icon='ORIENTATION_GIMBAL')

def register():
    bpy.utils.register_class(OBJECT_OT_rhino_3d_rotate_live)
    bpy.utils.register_class(VIEW3D_PT_3d_rotate_panel)

    bpy.types.VIEW3D_MT_transform_object.append(menu_func_transform)
    bpy.types.VIEW3D_MT_object_context_menu.append(menu_func_context)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new(
            "object.rhino_3d_rotate_live", 
            type='R', 
            value='PRESS', 
            ctrl=True, 
            shift=True
        )
        addon_keymaps.append((km, kmi))

def unregister():
    remove_all_draw_handlers()

    bpy.types.VIEW3D_MT_transform_object.remove(menu_func_transform)
    bpy.types.VIEW3D_MT_object_context_menu.remove(menu_func_context)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(VIEW3D_PT_3d_rotate_panel)
    bpy.utils.unregister_class(OBJECT_OT_rhino_3d_rotate_live)

if __name__ == "__main__":
    register()