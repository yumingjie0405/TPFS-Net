"""
Publication-grade matplotlib defaults shared by all tools.

Call ``apply()`` at the top of every visualization script.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE_OURS = '#d62728'
PALETTE_BASE = '#1f77b4'
PALETTE_DELTA = '#2ca02c'
SEMANTIC_PARTS = {
    'head':       '#e6194B',
    'torso':      '#3cb44b',
    'left_arm':   '#4363d8',
    'right_arm':  '#f58231',
    'left_leg':   '#911eb4',
    'right_leg':  '#42d4f4',
}


def apply(fontsize=11):
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'cm',
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize + 1,
        'legend.fontsize': fontsize - 1,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'axes.linewidth': 0.9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'lines.linewidth': 1.6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def save_dual(fig, path_no_ext):
    """Save both .pdf (vector) and .png (300 dpi) for paper use."""
    fig.savefig(f'{path_no_ext}.pdf')
    fig.savefig(f'{path_no_ext}.png')
    plt.close(fig)
