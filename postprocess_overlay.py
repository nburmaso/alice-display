#!/usr/bin/env python3
"""Add ALICE logo, run metadata, and MathText annotations to a detector render"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch


DEFAULT_CARD = {
    'layout': {
        'corner': 'top-right', 'margin_px': 54, 'padding_px': 18,
        'gap_px': 16, 'line_gap_px': 6, 'logo_width_px': 170,
        'panel_radius_px': 16,
    },
    'fonts': {
        'family': 'Computer Modern', 'family_bold': 'Computer Modern',
        'math_fontset': 'cm', 'title_size': 42,
        'body_size': 24, 'small_size': 22,
    },
    'style': {
        'text_color': '#ffffff', 'accent_color': '#ff3b3b',
        'panel_color': [0, 0, 0, 96], 'shadow_color': [0, 0, 0, 170],
        'shadow_offset_px': 2, 'line_color': [255, 255, 255, 90],
        'line_width_px': 2,
    },
    'metadata': {
        'experiment': '', 'run_number': '', 'period': '',
        'system': '', 'energy': '', 'time_zone': 'UTC',
        'timestamp_key': 'candidate_timestampMs',
        'timestamp_candidates': ['candidate_timestampMs', 'timestamp_ms', 'candidate_timestamp_ms'],
    },
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
    card = json.loads(path.read_text(encoding='utf-8'))
    parent = card.pop('extends', None)
    base = load_card(path.parent / parent) if parent else json.loads(json.dumps(DEFAULT_CARD))
    return merge_settings(base, card)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay-card', default='')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tracks-npz', default='')
    parser.add_argument('--logo', default='', help='Override the logo path from the overlay card')
    parser.add_argument('--run-number', default='')
    parser.add_argument('--period', default='')
    parser.add_argument('--system', default='')
    parser.add_argument('--energy', default='')
    parser.add_argument('--corner', default='', choices=['', 'top-left', 'top-right', 'bottom-left', 'bottom-right'])
    parser.add_argument('--timestamp-key', default='')
    parser.add_argument('--timestamp-ms', default='')
    return parser.parse_args()


def rgba(value):
    if isinstance(value, str):
        return to_rgba(value)
    values = np.asarray(value, dtype=float)
    if values.max() > 1.0:
        values /= 255.0
    return tuple(values) if len(values) == 4 else (*values, 1.0)


def font_property(name, bold=False, unicode_text=False):
    # Matplotlib bundles the Computer Modern TTF files used for MathText
    if name.lower() in ('computer modern', 'computer modern roman', 'cmr10', 'cmu serif'):
        if unicode_text:
            return font_manager.FontProperties(family='STIXGeneral', weight='bold' if bold else 'normal')
        font = 'cmb10.ttf' if bold else 'cmr10.ttf'
        path = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / font
        return font_manager.FontProperties(fname=str(path))
    path = Path(name)
    if path.is_file():
        return font_manager.FontProperties(fname=str(path))
    legacy_names = {'DejaVuSans.ttf': 'DejaVu Sans', 'DejaVuSans-Bold.ttf': 'DejaVu Sans'}
    family = legacy_names.get(name, name)
    return font_manager.FontProperties(family=family, weight='bold' if bold else 'normal')


def timestamp_from_npz(path, preferred, candidates):
    if not path:
        return None
    with np.load(path, allow_pickle=False) as data:
        for key in dict.fromkeys([preferred, *candidates]):
            if key and key in data and data[key].size:
                return int(np.asarray(data[key]).reshape(-1)[0])
    raise KeyError(f'No timestamp field found in {path}: searched {[preferred, *candidates]}')


def timestamp_label(timestamp_ms, time_zone):
    if timestamp_ms is None:
        return ''
    zone = ZoneInfo(time_zone)
    date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).astimezone(zone)
    return date.strftime('%Y-%m-%d %H:%M:%S.') + f'{timestamp_ms % 1000:03d} {time_zone}'


def math_label(value):
    # MathText parses TeX commands inside dollar delimiters
    value = str(value)
    if '\\' in value and '$' not in value:
        return f'${value}$'
    return value


def metadata_lines(meta, timestamp):
    lines = []
    if meta.get('run_number'):
        lines.append(('Run ' + str(meta['run_number']), 'body'))
    if meta.get('period'):
        lines.append(('Period ' + str(meta['period']), 'body'))
    system, energy = str(meta.get('system', '')).strip(), math_label(meta.get('energy', ''))
    if system and energy:
        lines.append((f'{system}, {energy}', 'body'))
    elif system or energy:
        lines.append((system or energy, 'body'))
    if timestamp:
        lines.append((timestamp, 'small'))
    return lines


def box_position(image_size, panel_size, corner, margin):
    width, height = image_size
    pw, ph = panel_size
    x = margin if corner.endswith('left') else width - margin - pw
    y = margin if corner.startswith('top') else height - margin - ph
    return x, y


def main():
    args = parse_args()
    card = load_card(args.overlay_card) if args.overlay_card else json.loads(json.dumps(DEFAULT_CARD))
    meta, layout, fonts, style = (card[name] for name in ('metadata', 'layout', 'fonts', 'style'))
    for arg, key in (('run_number', 'run_number'), ('period', 'period'), ('system', 'system'),
                     ('energy', 'energy'), ('timestamp_key', 'timestamp_key')):
        value = getattr(args, arg)
        if value:
            meta[key] = value
    if args.corner:
        layout['corner'] = args.corner

    timestamp = int(args.timestamp_ms) if args.timestamp_ms else timestamp_from_npz(
        args.tracks_npz, meta.get('timestamp_key', ''), meta.get('timestamp_candidates', []))
    lines = metadata_lines(meta, timestamp_label(timestamp, meta.get('time_zone', 'UTC')))
    title = str(meta.get('experiment', '')).strip()

    background = mpimg.imread(args.image)
    height, width = background.shape[:2]
    dpi = 100
    plt.rcParams['mathtext.fontset'] = fonts.get('math_fontset', 'cm')
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.imshow(background, extent=(0, width, height, 0), interpolation='nearest', zorder=0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    logo_path = args.logo or card.get('logo', 'assets/logo/ALICE-logo.png')
    logo = mpimg.imread(logo_path) if logo_path and Path(logo_path).is_file() else None
    logo_width = int(layout['logo_width_px']) if logo is not None else 0
    logo_height = round(logo_width * logo.shape[0] / logo.shape[1]) if logo is not None else 0
    gap, line_gap, padding = (int(layout[key]) for key in ('gap_px', 'line_gap_px', 'padding_px'))

    entries = ([('title', title)] if title else []) + [(role, text) for text, role in lines]
    # Font sizes are specified in pixels in the JSON card
    text_artists = []
    for role, value in entries:
        is_title = role == 'title'
        size = int(fonts['title_size' if is_title else ('small_size' if role == 'small' else 'body_size')])
        family = fonts.get('family_bold', fonts['family']) if is_title else fonts['family']
        face = font_property(family, is_title, any(ord(ch) > 127 for ch in value))
        artist = ax.text(0, 0, value, fontproperties=face, fontsize=size * 72 / dpi,
                         ha='left', va='top', color=rgba(style['accent_color'] if is_title else style['text_color']),
                         zorder=5)
        text_artists.append(artist)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    sizes = [(artist.get_window_extent(renderer).width, artist.get_window_extent(renderer).height)
             for artist in text_artists]
    text_width = max((size[0] for size in sizes), default=0)
    content_width = max(text_width, logo_width)
    panel_width = content_width + 2 * padding
    text_height = sum(size[1] for size in sizes) + line_gap * max(0, len(sizes) - 1)
    block_gap = gap if logo is not None and entries else 0
    panel_height = logo_height + block_gap + text_height + 2 * padding
    if not entries and logo is None:
        raise ValueError('Nothing to draw: supply a logo, text metadata, or a timestamp')

    x, y = box_position((width, height), (panel_width, panel_height), layout['corner'], int(layout['margin_px']))
    panel = FancyBboxPatch((x, y), panel_width, panel_height,
                           boxstyle=f'round,pad=0,rounding_size={layout["panel_radius_px"]}',
                           facecolor=rgba(style['panel_color']), edgecolor='none', zorder=2)
    ax.add_patch(panel)
    cursor = y + padding
    if logo is not None:
        logo_x = x + (panel_width - logo_width) / 2
        ax.imshow(logo, extent=(logo_x, logo_x + logo_width, cursor + logo_height, cursor),
                  interpolation='hanning', zorder=3)
        cursor += logo_height + block_gap

    for index, (artist, (tw, th)) in enumerate(zip(text_artists, sizes)):
        artist.set_position((x + (panel_width - tw) / 2, cursor))
        cursor += th + (line_gap if index < len(sizes) - 1 else 0)

    if title and len(entries) > 1:
        # The title is visually separated from the metadata by a thin rule
        divider_y = y + padding + logo_height + block_gap + sizes[0][1] + line_gap / 2
        ax.plot([x + padding, x + panel_width - padding], [divider_y, divider_y],
                color=rgba(style['line_color']), linewidth=int(style['line_width_px']) * 72 / dpi,
                zorder=4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, pad_inches=0, metadata={'Software': None})
    plt.close(figure)
    print(f'Saved overlay image: {output}')


if __name__ == '__main__':
    main()
