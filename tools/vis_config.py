"""Strict, shared publication configuration for visualization tools."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.text import Text

COLORS = {
    'red': '#e74c3c', 'blue': '#3498db', 'green': '#2ecc71',
    'orange': '#f39c12', 'purple': '#9b59b6', 'teal': '#1abc9c',
    'pink': '#e91e63', 'brown': '#8d6e63', 'gray': '#95a5a6',
}
COLORS_CATEGORICAL = [plt.get_cmap('turbo')(x) for x in __import__('numpy').linspace(0, 1, 60)]
COLORS_STAGES = [COLORS['red'], COLORS['blue'], COLORS['green'], COLORS['orange']]
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']


def resolve_font_path(font_path=None):
    """Resolve and verify an explicitly supplied Times New Roman font file."""
    value = font_path or os.environ.get('TPFS_FONT_PATH')
    if not value:
        raise RuntimeError(
            'Times New Roman is required: pass --font-path or set TPFS_FONT_PATH'
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Times New Roman font file not found: {path}')
    try:
        family = font_manager.FontProperties(fname=str(path)).get_name()
    except Exception as exc:
        raise RuntimeError(f'cannot read font file {path}: {exc}') from exc
    if family.casefold() != 'times new roman':
        raise RuntimeError(
            f'font must be Times New Roman, but {path} identifies as {family!r}'
        )
    return path, family


def setup_journal_style(font_path=None, fontsize=10):
    """Apply style only after strict Times New Roman validation."""
    path, family = resolve_font_path(font_path)
    font_manager.fontManager.addfont(str(path))
    rcParams.update({
        'font.family': family, 'font.serif': [family], 'font.size': fontsize,
        'axes.labelsize': fontsize + 1, 'axes.titlesize': fontsize + 1,
        'axes.titleweight': 'bold', 'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1, 'legend.fontsize': fontsize - 1,
        'figure.titlesize': fontsize + 2, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'text.usetex': False, 'axes.labelpad': 5,
        'axes.titlepad': 8, 'axes.axisbelow': True, 'grid.alpha': 0.25,
        'lines.linewidth': 1.6, 'legend.frameon': True,
        'legend.framealpha': 1.0, 'legend.edgecolor': 'gray',
    })
    return str(path)


def _inside(inner, outer, tolerance=1.0):
    return (inner.x0 >= outer.x0 - tolerance and
            inner.y0 >= outer.y0 - tolerance and
            inner.x1 <= outer.x1 + tolerance and
            inner.y1 <= outer.y1 + tolerance)


def validate_layout(fig, tolerance=1.0):
    """Fail on clipped axes/text/legends or legends overlapping plot panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    failures = []
    for index, ax in enumerate(fig.axes):
        box = ax.get_tightbbox(renderer)
        if box is not None and not _inside(box, canvas, tolerance):
            failures.append(f'axes[{index}] bounds leave the figure')
    tick_text_ids = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_text_ids.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(match=lambda item: isinstance(item, (Text, Legend))):
        if (not artist.get_visible() or not artist.get_in_layout() or
                (isinstance(artist, Text) and not artist.get_text())):
            continue
        # Axis tight bounds already account for tick labels.  Checking every
        # tick Text independently produces false positives at axes margins.
        if id(artist) in tick_text_ids:
            continue
        try:
            box = artist.get_window_extent(renderer)
        except (AttributeError, RuntimeError, ValueError):
            continue
        if box.width > 0 and box.height > 0 and not _inside(box, canvas, tolerance):
            label = artist.get_text() if isinstance(artist, Text) else 'legend'
            failures.append(f'artist out of bounds: {label!r}')
    legends = list(fig.legends)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legends.append(legend)
    for i, legend in enumerate(legends):
        box = legend.get_window_extent(renderer)
        for j, ax in enumerate(fig.axes):
            if box.overlaps(ax.bbox):
                failures.append(f'legend[{i}] overlaps axes[{j}] plot area')
        for other in legends[i + 1:]:
            if box.overlaps(other.get_window_extent(renderer)):
                failures.append(f'legend[{i}] overlaps another legend')
    if failures:
        raise RuntimeError('layout validation failed: ' + '; '.join(dict.fromkeys(failures)))
    return True


def safe_tight_layout(fig, pad=1.2, rect=None):
    fig.tight_layout(pad=pad, rect=rect)


def add_text_box(ax, text, x=0.02, y=0.98, **kwargs):
    defaults = dict(ha='left', va='top', fontsize=9)
    defaults.update(kwargs)
    return ax.text(x, y, text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='gray', alpha=0.95), **defaults)


def save_figure(fig, path, dpi=300):
    """Validate layout and write paired vector/raster outputs."""
    validate_layout(fig)
    fig.savefig(f'{path}.pdf', dpi=dpi)
    fig.savefig(f'{path}.png', dpi=dpi)
    print(f'[saved] {path}.pdf + .png')
