#!/usr/bin/env python3
"""Render ALICE detector OBJ geometry and sampled trajectories in Blender

Coordinates are stored in Blender meters: (X, Y, Z) = ALICE (Z, X, Y) / 100
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector

# Geometry material styles
STYLE = {
    'TPC_CAGE':       {'color': (0.10, 0.66, 0.79), 'alpha': 0.28, 'roughness': 0.44},
    'TPC_ENDCAP':     {'color': (0.28, 0.82, 0.86), 'alpha': 0.44, 'roughness': 0.40},
    'TPC_SHELL':      {'color': (0.08, 0.45, 0.58), 'alpha': 0.18, 'roughness': 0.52},
    'TPC_ACTIVE':     {'color': (0.16, 0.73, 0.75), 'alpha': 0.08, 'roughness': 0.32},
    'ITS':            {'color': (0.41, 0.86, 0.63), 'alpha': 0.72, 'roughness': 0.36},
    'TRD':            {'color': (0.44, 0.53, 0.87), 'alpha': 0.86, 'roughness': 0.42},
    'TOF':            {'color': (0.94, 0.43, 0.39), 'alpha': 0.90, 'roughness': 0.35},
    'EMCAL':          {'color': (0.56, 0.84, 0.72), 'alpha': 0.76, 'roughness': 0.42},
    'DCAL':           {'color': (0.38, 0.74, 0.64), 'alpha': 0.76, 'roughness': 0.42},
    'EMCAL_CELLS':    {'color': (0.76, 0.95, 0.63), 'alpha': 0.84, 'roughness': 0.28},
    'DCAL_CELLS':     {'color': (0.60, 0.90, 0.68), 'alpha': 0.84, 'roughness': 0.28},
    'L3_MAGNET':      {'color': (0.95, 0.06, 0.10), 'alpha': 0.92, 'roughness': 0.50},
    'DIPOLE_COIL':    {'color': (0.96, 0.72, 0.34), 'alpha': 0.54, 'roughness': 0.34},
    'DIPOLE_YOKE':    {'color': (0.45, 0.50, 0.58), 'alpha': 0.24, 'roughness': 0.68},
    'FRONT_ABSORBER': {'color': (0.46, 0.47, 0.50), 'alpha': 0.26, 'roughness': 0.74},
    'MUON_FILTER':    {'color': (0.42, 0.43, 0.46), 'alpha': 0.24, 'roughness': 0.78},
    'BEAM_SHIELD':    {'color': (0.59, 0.75, 0.70), 'alpha': 0.26, 'roughness': 0.54},
    'BEAM_PIPE':      {'color': (0.90, 0.93, 0.95), 'alpha': 0.96, 'roughness': 0.22, 'metallic': 0.22},
    'MFT':            {'color': (0.66, 0.36, 0.60), 'alpha': 0.82, 'roughness': 0.42},
    'MCH':            {'color': (0.71, 0.38, 0.60), 'alpha': 0.88, 'roughness': 0.46},
    'MCH_PANELS':     {'color': (0.61, 0.42, 0.68), 'alpha': 0.70, 'roughness': 0.42},
    'MID':            {'color': (0.72, 0.40, 0.59), 'alpha': 0.88, 'roughness': 0.46},
    'FT0A':           {'color': (0.98, 0.51, 0.49), 'alpha': 0.90, 'roughness': 0.34},
    'FT0C':           {'color': (0.95, 0.41, 0.52), 'alpha': 0.90, 'roughness': 0.34},
    'FV0A':           {'color': (0.98, 0.64, 0.26), 'alpha': 0.84, 'roughness': 0.34},
    'FDDA':           {'color': (0.84, 0.72, 0.34), 'alpha': 0.82, 'roughness': 0.34},
    'FDDC':           {'color': (0.84, 0.72, 0.34), 'alpha': 0.82, 'roughness': 0.34},
    'OTHER':          {'color': (0.72, 0.76, 0.81), 'alpha': 0.34, 'roughness': 0.58},

    # Trajectory styling, sizes in meters
    'TRACK_BARREL':   {'color': (1.00, 0.95, 0.68), 'thickness': 0.012, 'glow': 1.28, 'alpha': 1.0},
    'TRACK_FORWARD':  {'color': (1.00, 0.95, 0.68), 'thickness': 0.012, 'glow': 1.28, 'alpha': 1.0},

    # Optional TPC spotlight
    'TPC_SPOTLIGHT': {
        'color': (0.88, 0.92, 1.0),
        'energy': 350.0,
        'radial_distance': 3.2,
        'x': 0.25,
        'y_offset': 0.0,
        'z_offset': 0.20,
        'target': (0.0, 0.0, 0.0),
        'cone_angle': 88.0,
        'blend': 0.82,
        'soft_size': 0.75,
    },
}

kMFT, kMCH, kMID = 0x10, 0x20, 0x40

DEFAULT_CUTOUT_GROUPS = (
    'L3_MAGNET', 'TPC_CAGE', 'TPC_ENDCAP', 'TPC_SHELL',
    'TPC_ACTIVE', 'ITS', 'TRD', 'TOF', 'EMCAL', 'DCAL',
    'EMCAL_CELLS', 'DCAL_CELLS',
)
DEFAULT_CAMERA_BARREL_GROUPS = (
    'L3_MAGNET', 'TPC_CAGE', 'TPC_ENDCAP', 'TPC_SHELL', 'TPC_ACTIVE',
    'BEAM_PIPE', 'ITS', 'TRD', 'TOF', 'EMCAL', 'DCAL',
    'EMCAL_CELLS', 'DCAL_CELLS',
)

# Light positions and sizes are relative to the scene or target detector group
LIGHTING = {
    'world': {'color': (0.0, 0.0, 0.0), 'strength': 0.16},
    'key':       {'position': (0.55, -1.15, 1.20), 'energy_per_radius': 720, 'diameter_per_radius': 1.35},
    'fill':      {'position': (-1.10, 0.95, 0.72), 'energy_per_radius': 280, 'diameter_per_radius': 1.85},
    'rim':       {'position': (-0.15, -1.40, 1.55), 'energy_per_radius': 420, 'diameter_per_radius': 1.05},
    'forward':   {'position': (1.65, 0.50, 0.28), 'energy_per_radius': 260, 'diameter_per_radius': 1.80},
}


def merge_settings(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_settings(base[key], value)
        else:
            base[key] = value
    return base


def load_card(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as file:
        card = json.load(file)
    parent = card.pop('extends', None)
    return merge_settings(load_card(path.parent / parent), card) if parent else card


CARD_OPTIONS = {
    'geometry': {
        'inputs': 'inputs', 'prefix': 'prefix', 'groups': 'groups', 'exclude': 'exclude',
        'opacity': 'opacity', 'solidify': 'solidify', 'alpha_mode': 'alpha_mode',
        'shade_smooth': 'shade_smooth',
    },
    'tracks': {'file': 'tracks_npz', 'bevel_resolution': 'track_bevel_resolution'},
    'camera': {
        'azimuth': 'azimuth', 'elevation': 'elevation', 'margin': 'margin',
        'zoom': 'camera_zoom', 'center_mode': 'camera_center_mode',
        'barrel_groups': 'camera_barrel_groups',
    },
    'cutout': {
        'mode': 'cutout', 'method': 'cutout_method', 'radius': 'cutout_radius',
        'xmin': 'cutout_xmin', 'xmax': 'cutout_xmax', 'segments': 'cutout_segments',
        'hole_tolerant': 'cutout_hole_tolerant', 'epsilon': 'cutout_epsilon',
        'groups': 'cutout_groups', 'phi_center': 'cutout_phi_center',
        'phi_width': 'cutout_phi_width',
    },
    'render': {
        'output': 'output', 'save_blend': 'save_blend', 'engine': 'engine',
        'width': 'width', 'height': 'height', 'samples': 'samples',
        'view_transform': 'view_transform', 'transparent': 'film_transparent',
        'ambient_occlusion': 'ambient_occlusion', 'bloom': 'bloom',
        'bloom_intensity': 'bloom_intensity', 'bloom_radius': 'bloom_radius',
        'bloom_threshold': 'bloom_threshold',
    },
}


def card_arguments(card):
    defaults = {}
    for section, mapping in CARD_OPTIONS.items():
        for key, dest in mapping.items():
            if key not in card.get(section, {}):
                continue
            value = card[section][key]
            if dest in ('opacity', 'solidify') and isinstance(value, dict):
                value = ','.join(f'{k}:{v}' for k, v in value.items())
            elif dest in ('groups', 'exclude', 'camera_barrel_groups', 'cutout_groups') and isinstance(value, list):
                value = ','.join(value)
            defaults[dest] = value
    return defaults


def group_from_filename(path, prefix='display'):
    stem = Path(path).stem
    if stem.startswith(prefix + '_'):
        return stem[len(prefix) + 1:].upper()
    if stem.startswith('alice_geom_'):
        return stem[len('alice_geom_'):].upper()
    return stem.upper()


def parse_mapping(arg, name, min_value=0.0):
    mapping = {}
    if not arg.strip():
        return mapping
    for entry in arg.split(','):
        if not entry.strip():
            continue
        key, value = entry.split(':', 1)
        key = key.strip().upper()
        val = float(value)
        if val < min_value:
            raise ValueError(f'Bad --{name} entry {entry!r}: value must be >= {min_value}')
        mapping[key] = val
    return mapping


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-card', default='', help='JSON display configuration')
    parser.add_argument('--inputs', default='display_*.obj', help='quoted file glob')
    parser.add_argument('--prefix', default='display', help='OBJ filename prefix before GROUP.obj')
    parser.add_argument('--groups', default='', help='optional comma-separated inclusion groups')
    parser.add_argument('--exclude', default='', help='comma-separated exclusion groups')
    parser.add_argument('--opacity', default='', help='per-group alpha overrides, e.g. MUON_FILTER:0.6')
    parser.add_argument('--solidify', default='TOF:0.03,TRD:0.012,ITS:0.008',
                        help='per-group display thickness in meters; empty string disables')
    parser.add_argument('--tracks-npz', default='', help='NPZ file with sampled track trajectories')
    parser.add_argument('--track-bevel-resolution', type=int, default=5)
    parser.add_argument('--output', default='geometry.png')
    parser.add_argument('--save-blend', default='')
    parser.add_argument('--engine', choices=['auto', 'eevee', 'cycles', 'workbench'], default='auto')
    parser.add_argument('--view-transform', default='Standard')
    parser.add_argument('--film-transparent', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ambient-occlusion', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bloom', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bloom-intensity', type=float, default=0.05)
    parser.add_argument('--bloom-radius', type=float, default=5.5)
    parser.add_argument('--bloom-threshold', type=float, default=0.62)
    parser.add_argument('--alpha-mode', choices=['HASHED', 'BLEND', 'OPAQUE'], default='HASHED')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--azimuth', type=float, default=-70.0)
    parser.add_argument('--elevation', type=float, default=24.0)
    parser.add_argument('--shade-smooth', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--margin', type=float, default=1.12)
    parser.add_argument('--camera-zoom', type=float, default=1.0)
    parser.add_argument('--camera-center-mode', choices=['auto', 'barrel'], default='auto')
    parser.add_argument('--camera-barrel-groups', default=','.join(DEFAULT_CAMERA_BARREL_GROUPS))
    parser.add_argument('--cutout', choices=['none', 'sector'], default='none')
    parser.add_argument('--cutout-method', choices=['bisect', 'boolean'], default='bisect')
    parser.add_argument('--cutout-radius', type=float, default=None)
    parser.add_argument('--cutout-xmin', type=float, default=None)
    parser.add_argument('--cutout-xmax', type=float, default=None)
    parser.add_argument('--cutout-segments', type=int, default=96)
    parser.add_argument('--cutout-hole-tolerant', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--cutout-epsilon', type=float, default=1.0e-6)
    parser.add_argument('--cutout-groups', default=','.join(DEFAULT_CUTOUT_GROUPS))
    parser.add_argument('--cutout-phi-center', type=float, default=None)
    parser.add_argument('--cutout-phi-width', type=float, default=85.0)
    parser.add_argument('--tpc-light-energy', type=float, default=None,
                        help='override STYLE[TPC_SPOTLIGHT] energy in watts, 0 disables')
    parser.add_argument('--tpc-light-x', type=float, default=None,
                        help='override spotlight axial X position in meters')
    parser.add_argument('--tpc-light-y-offset', type=float, default=0.0,
                        help='extra Blender Y offset from the spotlight default position, meters')
    parser.add_argument('--tpc-light-z-offset', type=float, default=0.0,
                        help='extra Blender Z offset from the spotlight default position, meters')

    card_path = argparse.ArgumentParser(add_help=False)
    card_path.add_argument('--input-card', default='')
    path = card_path.parse_known_args(argv)[0].input_card
    if path:
        card = load_card(path)
        parser.set_defaults(**card_arguments(card))
        merge_settings(STYLE, card.get('styles', {}))
        merge_settings(LIGHTING, card.get('lighting', {}))
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.samples) <= 0 or args.margin < 1:
        parser.error('width, height, samples must be >0 and margin must be >=1')
    if args.camera_zoom <= 0:
        parser.error('--camera-zoom must be positive')
    if args.cutout == 'sector' and not (0.0 < args.cutout_phi_width < 180.0):
        parser.error('--cutout-phi-width must be between 0 and 180 degrees')
    return args


def configure_engine(scene, args):
    choices = {
        'auto': ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT', 'CYCLES', 'BLENDER_WORKBENCH'),
        'eevee': ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'),
        'cycles': ('CYCLES',),
        'workbench': ('BLENDER_WORKBENCH',),
    }[args.engine]
    errors = []
    for choice in choices:
        try:
            scene.render.engine = choice
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            break
    else:
        raise RuntimeError('None of the requested renderers is present. ' + '; '.join(errors))
    engine = scene.render.engine
    if engine == 'CYCLES':
        scene.cycles.samples = args.samples
    elif engine.startswith('BLENDER_EEVEE'):
        if hasattr(scene, 'eevee'):
            if hasattr(scene.eevee, 'taa_render_samples'):
                scene.eevee.taa_render_samples = args.samples
            if hasattr(scene.eevee, 'use_gtao'):
                scene.eevee.use_gtao = args.ambient_occlusion
                if hasattr(scene.eevee, 'gtao_distance'):
                    scene.eevee.gtao_distance = 2.0
            if hasattr(scene.eevee, 'use_bloom'):
                scene.eevee.use_bloom = args.bloom
                if hasattr(scene.eevee, 'bloom_intensity'):
                    scene.eevee.bloom_intensity = args.bloom_intensity
                if hasattr(scene.eevee, 'bloom_radius'):
                    scene.eevee.bloom_radius = args.bloom_radius
                if hasattr(scene.eevee, 'bloom_threshold'):
                    scene.eevee.bloom_threshold = args.bloom_threshold
    else:
        scene.display.shading.color_type = 'MATERIAL'
    print('Renderer:', engine)


def make_material(group, alpha_mode, alpha_overrides):
    style = STYLE.get(group, STYLE['OTHER'])
    rgb, alpha = style['color'], style['alpha']
    alpha = alpha_overrides.get(group, alpha)
    if alpha_mode == 'OPAQUE':
        alpha = 1.0
    mat = bpy.data.materials.new('ALICE_' + group)
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], alpha)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs['Alpha'].default_value = alpha
    bsdf.inputs['Metallic'].default_value = style.get('metallic', 0.08)
    bsdf.inputs['Roughness'].default_value = style.get('roughness', 0.46)
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = style.get('specular', 0.32)
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = style.get('specular', 0.32)

    emission_color = style.get('emission')
    emission_strength = style.get('emission_strength', 0.0)
    if emission_color is not None and emission_strength > 0.0:
        emission_value = (emission_color[0], emission_color[1], emission_color[2], 1.0)
        if 'Emission Color' in bsdf.inputs:
            bsdf.inputs['Emission Color'].default_value = emission_value
        elif 'Emission' in bsdf.inputs:
            bsdf.inputs['Emission'].default_value = emission_value
        if 'Emission Strength' in bsdf.inputs:
            bsdf.inputs['Emission Strength'].default_value = emission_strength

    if hasattr(mat, 'blend_method'):
        mat.blend_method = ('OPAQUE' if alpha >= 1 else alpha_mode)
    if hasattr(mat, 'shadow_method') and alpha < 1.0 and alpha_mode == 'BLEND':
        mat.shadow_method = 'NONE'
    if hasattr(mat, 'use_backface_culling'):
        mat.use_backface_culling = False
    return mat


def make_emission_material(name, color, glow=1.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (220, 0)
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.location = (30, 0)
    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    transp.location = (-180, -80)
    emiss = nt.nodes.new('ShaderNodeEmission')
    emiss.location = (-180, 80)
    emiss.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    emiss.inputs['Strength'].default_value = glow
    mix.inputs[0].default_value = alpha
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(emiss.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    if hasattr(mat, 'blend_method'):
        mat.blend_method = 'BLEND' if alpha < 1.0 else 'OPAQUE'
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'
    return mat


def import_obj(path):
    filepath = str(Path(path).resolve())
    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='Y', up_axis='Z')
    else:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='Y', axis_up='Z')
    return [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']


def sector_mesh_data(xmin, xmax, radius, phi_center_deg, phi_width_deg, segments):
    start = math.radians(phi_center_deg - 0.5 * phi_width_deg)
    sweep = math.radians(phi_width_deg)
    angles = [start + sweep * i / segments for i in range(segments + 1)]
    vertices = [(xmin, 0.0, 0.0), (xmax, 0.0, 0.0)]
    lower = []
    upper = []
    for angle in angles:
        lower.append(len(vertices))
        vertices.append((xmin, radius * math.cos(angle), radius * math.sin(angle)))
        upper.append(len(vertices))
        vertices.append((xmax, radius * math.cos(angle), radius * math.sin(angle)))
    faces = []
    for index in range(segments):
        faces.append((0, lower[index + 1], lower[index]))
        faces.append((1, upper[index], upper[index + 1]))
        faces.append((lower[index], lower[index + 1], upper[index + 1], upper[index]))
    faces.append((0, lower[0], upper[0], 1))
    faces.append((lower[-1], 0, 1, upper[-1]))
    return vertices, faces


def apply_boolean_sector_cutout(geometry_by_group, args):
    selected = {group.strip().upper() for group in args.cutout_groups.split(',') if group.strip()}
    targets = [obj for group, objects in geometry_by_group.items() if group in selected for obj in objects]
    if not targets:
        print('WARNING: sector cutout skipped: none of the selected groups were imported')
        return
    corners = [obj.matrix_world @ Vector(corner) for obj in targets for corner in obj.bound_box]
    minx = min(point.x for point in corners)
    maxx = max(point.x for point in corners)
    maxradius = max(math.hypot(point.y, point.z) for point in corners)
    margin = max(0.05, 0.03 * (maxx - minx))
    xmin = args.cutout_xmin if args.cutout_xmin is not None else minx - margin
    xmax = args.cutout_xmax if args.cutout_xmax is not None else maxx + margin
    radius = args.cutout_radius if args.cutout_radius is not None else maxradius * 1.15 + 0.05
    if xmin >= xmax:
        raise ValueError('Invalid cutter range: xmin >= xmax')
    if args.cutout_phi_center is None:
        azimuth = math.radians(args.azimuth)
        elevation = math.radians(args.elevation)
        cam_y = math.cos(elevation) * math.sin(azimuth)
        cam_z = math.sin(elevation)
        phi_center = 90.0 if abs(cam_y) + abs(cam_z) < 1e-8 else math.degrees(math.atan2(cam_z, cam_y))
    else:
        phi_center = args.cutout_phi_center
    verts, faces = sector_mesh_data(xmin, xmax, radius, phi_center, args.cutout_phi_width, args.cutout_segments)
    mesh = bpy.data.meshes.new('ALICE_sector_cutter_mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    cutter = bpy.data.objects.new('ALICE_sector_cutter', mesh)
    bpy.context.collection.objects.link(cutter)
    cutter.display_type = 'WIRE'
    cutter.hide_render = True
    for obj in targets:
        modifier = obj.modifiers.new('ALICE_sector_cutout', 'BOOLEAN')
        modifier.operation = 'DIFFERENCE'
        modifier.solver = 'EXACT'
        modifier.object = cutter
        if args.cutout_hole_tolerant and hasattr(modifier, 'use_hole_tolerant'):
            modifier.use_hole_tolerant = True
    print('Sector cutout: {} object(s), groups={}, phi={:.1f}±{:.1f} deg, radius={:.2f} m, X=[{:.2f}, {:.2f}] m'.format(
        len(targets), ','.join(sorted(set(geometry_by_group) & selected)),
        phi_center, args.cutout_phi_width / 2, radius, xmin, xmax))


def sector_angles(args):
    if args.cutout_phi_center is not None:
        return args.cutout_phi_center
    azimuth = math.radians(args.azimuth)
    elevation = math.radians(args.elevation)
    cam_y = math.cos(elevation) * math.sin(azimuth)
    cam_z = math.sin(elevation)
    if abs(cam_y) + abs(cam_z) < 1.0e-8:
        return 90.0
    return math.degrees(math.atan2(cam_z, cam_y))


def bisect_halfspace(mesh_bm, plane_co, plane_no, positive, epsilon):
    geometry = list(mesh_bm.verts) + list(mesh_bm.edges) + list(mesh_bm.faces)
    if not geometry:
        return
    bmesh.ops.bisect_plane(mesh_bm, geom=geometry, dist=epsilon, plane_co=plane_co, plane_no=plane_no,
                           clear_inner=positive, clear_outer=not positive)


def mesh_piece_from_bmesh(original, bm, name):
    if not bm.faces:
        return None
    unused_verts = [vert for vert in bm.verts if not vert.link_faces]
    if unused_verts:
        bmesh.ops.delete(bm, geom=unused_verts, context='VERTS')
    result = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(result)
    result.update()
    for material in original.materials:
        if material is not None:
            result.materials.append(material)
    return result


def clip_obj_to_sector(obj, start_deg, end_deg, epsilon):
    angle0, angle1 = math.radians(start_deg), math.radians(end_deg)
    normal_start = Vector((0.0, -math.sin(angle0), math.cos(angle0)))
    normal_end = Vector((0.0, -math.sin(angle1), math.cos(angle1)))
    matrix = obj.matrix_world.copy()
    co_local = matrix.inverted() @ Vector((0.0, 0.0, 0.0))
    start_local = (matrix.to_3x3().transposed() @ normal_start).normalized()
    end_local = (matrix.to_3x3().transposed() @ normal_end).normalized()
    original = obj.data
    bm_a = bmesh.new(); bm_a.from_mesh(original)
    bisect_halfspace(bm_a, co_local, start_local, False, epsilon)
    piece_a = mesh_piece_from_bmesh(original, bm_a, obj.name + '_sector_A_mesh'); bm_a.free()
    bm_b = bmesh.new(); bm_b.from_mesh(original)
    bisect_halfspace(bm_b, co_local, start_local, True, epsilon)
    bisect_halfspace(bm_b, co_local, end_local, True, epsilon)
    piece_b = mesh_piece_from_bmesh(original, bm_b, obj.name + '_sector_B_mesh'); bm_b.free()
    pieces = []
    for tag, mesh in (('A', piece_a), ('B', piece_b)):
        if mesh is None:
            continue
        if not pieces:
            obj.name = obj.name + '_sector_' + tag
            obj.data = mesh
            pieces.append(obj)
        else:
            new_obj = bpy.data.objects.new(obj.name.rsplit('_sector_', 1)[0] + '_sector_' + tag, mesh)
            bpy.context.collection.objects.link(new_obj)
            new_obj.matrix_world = matrix
            pieces.append(new_obj)
    if not pieces:
        bpy.data.objects.remove(obj, do_unlink=True)
    if original.users == 0:
        bpy.data.meshes.remove(original)
    return pieces


def apply_bisect_sector_cutout(geometry_by_group, args):
    selected = {name.strip().upper() for name in args.cutout_groups.split(',') if name.strip()}
    phi_center = sector_angles(args)
    start = phi_center - args.cutout_phi_width / 2.0
    end = phi_center + args.cutout_phi_width / 2.0
    groups_cut = []
    for group, objects in list(geometry_by_group.items()):
        if group not in selected:
            continue
        rebuilt = []
        for obj in objects:
            rebuilt.extend(clip_obj_to_sector(obj, start, end, args.cutout_epsilon))
        geometry_by_group[group] = rebuilt
        groups_cut.append(group)
    if groups_cut:
        print('BMesh sector face clipping: groups={}, phi={:.1f}±{:.1f} deg'.format(','.join(groups_cut), phi_center, args.cutout_phi_width / 2.0))


def apply_group_solidify(geometry_by_group, solidify_map):
    for group, thickness in solidify_map.items():
        objects = geometry_by_group.get(group, [])
        for obj in objects:
            modifier = obj.modifiers.new('ALICE_solidify', 'SOLIDIFY')
            modifier.thickness = thickness
            modifier.offset = 0.0
            modifier.use_even_offset = True
            modifier.use_quality_normals = True
            modifier.use_rim = True
        if objects:
            print(f'Solidify: {group} {thickness:.4f} m on {len(objects)} object(s)')


def make_light(name, position, target, energy, diameter):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'DISK'
    data.size = diameter
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = position
    light.rotation_euler = (target - light.location).to_track_quat('-Z', 'Y').to_euler()
    return light


def setup_camera(scene, geometry_by_group, args):
    all_geometry = [obj for objects in geometry_by_group.values() for obj in objects if obj.name in bpy.data.objects]
    if not all_geometry:
        raise RuntimeError('All imported geometry disappeared after clipping')
    barrel_groups = [group.strip().upper() for group in args.camera_barrel_groups.split(',') if group.strip()]
    barrel_geometry = [obj for group in barrel_groups for obj in geometry_by_group.get(group, []) if obj.name in bpy.data.objects]
    if args.camera_center_mode == 'barrel' and barrel_geometry:
        framing_geometry = barrel_geometry
        center = Vector((0.0, 0.0, 0.0))
    else:
        framing_geometry = all_geometry
        all_corners = [obj.matrix_world @ Vector(corner) for obj in framing_geometry for corner in obj.bound_box]
        minimum = Vector([min(point[i] for point in all_corners) for i in range(3)])
        maximum = Vector([max(point[i] for point in all_corners) for i in range(3)])
        center = (minimum + maximum) * 0.5
    frame_corners = [obj.matrix_world @ Vector(corner) for obj in framing_geometry for corner in obj.bound_box]
    minimum = Vector([min(point[i] for point in frame_corners) for i in range(3)])
    maximum = Vector([max(point[i] for point in frame_corners) for i in range(3)])
    radius = max((maximum - minimum).length * 0.5, 1.0)
    print('Geometry bounds [m]:', tuple(round(x, 3) for x in minimum), tuple(round(x, 3) for x in maximum))
    azimuth = math.radians(args.azimuth)
    elevation = math.radians(args.elevation)
    direction = Vector((math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)))
    data = bpy.data.cameras.new('ALICE_orthographic_camera')
    camera = bpy.data.objects.new('ALICE_orthographic_camera', data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + 3.5 * radius * direction
    camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
    data.type = 'ORTHO'
    data.sensor_fit = 'HORIZONTAL'
    inv_rotation = camera.rotation_euler.to_matrix().inverted()
    screen = [inv_rotation @ (point - center) for point in frame_corners]
    projected_width = max(point.x for point in screen) - min(point.x for point in screen)
    projected_height = max(point.y for point in screen) - min(point.y for point in screen)
    aspect = args.width / float(args.height)
    base_scale = max(projected_width, projected_height * aspect, 0.1) * args.margin
    data.ortho_scale = base_scale / args.camera_zoom
    scene.camera = camera
    return center, radius


def add_event_tracks(tracks_npz, scene, bevel_resolution):
    data = np.load(tracks_npz)
    ids = data['track_id']
    offsets = data['offsets']
    points = data['points_m']
    masks = data['detector_mask']
    collection = bpy.data.collections.new('EVENT')
    scene.collection.children.link(collection)
    materials = {}
    count = 0
    for index, tid in enumerate(ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end - start < 2:
            continue
        group = 'TRACK_FORWARD' if int(masks[index]) & (kMCH | kMID | kMFT) else 'TRACK_BARREL'
        style = STYLE[group]
        if group not in materials:
            materials[group] = make_emission_material('MAT_' + group, style['color'], style['glow'], style.get('alpha', 1.0))
        name = f'EVENT_track_{int(tid)}'
        curve = bpy.data.curves.new(name, type='CURVE')
        curve.dimensions = '3D'
        curve.bevel_depth = style['thickness']
        curve.bevel_resolution = bevel_resolution
        spline = curve.splines.new('POLY')
        spline.points.add(end - start - 1)
        for point, xyz in zip(spline.points, points[start:end]):
            point.co = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
        curve.materials.append(materials[group])
        collection.objects.link(bpy.data.objects.new(name, curve))
        count += 1
    print(f'Added {count} trajectories from {tracks_npz}')


def add_tpc_spotlight(args):
    style = STYLE['TPC_SPOTLIGHT']
    energy = style['energy'] if args.tpc_light_energy is None else args.tpc_light_energy
    if energy <= 0:
        return None
    phi = math.radians(sector_angles(args))
    position = Vector((
        style['x'] if args.tpc_light_x is None else args.tpc_light_x,
        style['radial_distance'] * math.cos(phi) + style['y_offset'] + args.tpc_light_y_offset,
        style['radial_distance'] * math.sin(phi) + style['z_offset'] + args.tpc_light_z_offset,
    ))
    target = Vector(style['target'])
    light_data = bpy.data.lights.new('TPC_spot_fill', 'SPOT')
    light_data.energy = energy
    light_data.color = style['color']
    light_data.spot_size = math.radians(style['cone_angle'])
    light_data.spot_blend = style['blend']
    light_data.shadow_soft_size = style['soft_size']
    obj = bpy.data.objects.new('TPC_spot_fill', light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (target - position).to_track_quat('-Z', 'Y').to_euler()
    print('TPC spotlight: position={} target={} energy={:.0f} W'.format(
        tuple(round(float(value), 2) for value in position),
        tuple(round(float(value), 2) for value in target), energy))
    return obj


def main():
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    requested = {group.strip().upper() for group in args.groups.split(',') if group.strip()}
    excluded = {group.strip().upper() for group in args.exclude.split(',') if group.strip()}
    alpha_overrides = parse_mapping(args.opacity, 'opacity', min_value=0.0)
    solidify_map = parse_mapping(args.solidify, 'solidify', min_value=0.0)
    if requested:
        files = [path for path in files if group_from_filename(path, args.prefix) in requested]
    files = [path for path in files if group_from_filename(path, args.prefix) not in excluded]

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    materials = {}
    geometry_by_group = {}
    for path in files:
        group = group_from_filename(path, args.prefix)
        if group not in materials:
            materials[group] = make_material(group, args.alpha_mode, alpha_overrides)
        imported = import_obj(path)
        for obj in imported:
            obj.name = 'ALICE_' + group
            obj.data.materials.clear()
            obj.data.materials.append(materials[group])
            if args.shade_smooth and hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(35)
            for polygon in obj.data.polygons:
                polygon.use_smooth = args.shade_smooth
            geometry_by_group.setdefault(group, []).append(obj)
        print('Imported {}: {} mesh object(s)'.format(Path(path).name, len(imported)))

    if geometry_by_group and args.cutout == 'sector':
        if args.cutout_method == 'bisect':
            apply_bisect_sector_cutout(geometry_by_group, args)
        else:
            apply_boolean_sector_cutout(geometry_by_group, args)
    apply_group_solidify(geometry_by_group, solidify_map)

    scene = bpy.context.scene
    configure_engine(scene, args)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    scene.render.film_transparent = args.film_transparent
    scene.view_settings.view_transform = args.view_transform

    if scene.world is None:
        scene.world = bpy.data.worlds.new('ALICE_world')
    scene.world.use_nodes = True
    world_bg = scene.world.node_tree.nodes.get('Background')
    world_bg.inputs['Color'].default_value = (*LIGHTING['world']['color'], 1.0)
    world_bg.inputs['Strength'].default_value = LIGHTING['world']['strength']

    center = Vector((0.0, 0.0, 0.0))
    radius = 4.0
    if geometry_by_group:
        center, radius = setup_camera(scene, geometry_by_group, args)
    else:
        data = bpy.data.cameras.new('ALICE_orthographic_camera')
        camera = bpy.data.objects.new('ALICE_orthographic_camera', data)
        bpy.context.collection.objects.link(camera)
        azimuth = math.radians(args.azimuth)
        elevation = math.radians(args.elevation)
        direction = Vector((math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)))
        camera.location = center + 12.0 * direction
        camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
        data.type = 'ORTHO'
        data.sensor_fit = 'HORIZONTAL'
        data.ortho_scale = 8.0 / args.camera_zoom
        scene.camera = camera

    for key, light in LIGHTING.items():
        if key == 'world' or light.get('energy_per_radius', 0) <= 0:
            continue
        target, scale = center, radius
        group = light.get('target_group')
        if group:
            objects = geometry_by_group.get(group.upper(), [])
            if not objects:
                print(f'Light {key}: no geometry for {group}')
                continue
            corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
            lower = Vector([min(point[i] for point in corners) for i in range(3)])
            upper = Vector([max(point[i] for point in corners) for i in range(3)])
            target, scale = (lower + upper) * 0.5, max((upper - lower).length * 0.5, 1.0)
        position = target + scale * Vector(light['position'])
        make_light('Area_' + key, position, target,
                   light['energy_per_radius'] * scale, light['diameter_per_radius'] * scale)
        print(f'Light {key}: target={tuple(round(x, 2) for x in target)}, scale={scale:.2f} m')
    add_tpc_spotlight(args)
    if args.tracks_npz:
        add_event_tracks(args.tracks_npz, scene, args.track_bevel_resolution)

    if args.save_blend:
        blendfile = Path(args.save_blend).expanduser().resolve()
        blendfile.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blendfile))
        print('Saved Blender scene:', blendfile)
    bpy.ops.render.render(write_still=True)
    print('Saved image:', output)


if __name__ == '__main__':
    main()
