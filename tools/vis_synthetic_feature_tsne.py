"""Synthetic, watermarked t-SNE illustration (baseline vs TPFS-Net).

SYNTHETIC / NOT FOR SUBMISSION.

This figure is a *target/placeholder* illustration only. It draws Gaussian
class blobs whose spread is designed so the right panel (labelled TPFS-Net)
shows tighter, better-separated clusters than the left panel. The point cloud
is generated, not measured: it does not come from any trained checkpoint and
must never be used as experimental evidence in a paper, submission, defence, or
result claim. Every output carries a visible ``SYNTHETIC / NOT FOR SUBMISSION``
watermark and the JSON/CSV metadata repeats the disclaimer. Replace with real,
paired ``tools/vis_feature_tsne.py`` output before any real use.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DISCLAIMER = 'SYNTHETIC / NOT FOR SUBMISSION'
# Four categorical hues validated by the dataviz palette (light surface):
# worst all-pairs CVD deltaE 13.0, normal-vision deltaE 19.6.
FIGURE_COLORS = ('#2a78d6', '#008300', '#e87ba4', '#eda100')
FIGURE_MARKERS = ('o', 's', '^', 'D')
CLASS_LABELS = (9, 11, 28, 43)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=160,
                        help='synthetic points generated per class')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out',
                        default='vis/figs_paper/fig_feature_tsne_synthetic')
    return parser.parse_args(argv)


def _blobs(rng, centers, spread, per_class):
    """Draw per-class 2-D Gaussian blobs around the given centers."""
    points, labels = [], []
    for code, center in enumerate(centers):
        cloud = rng.normal(loc=center, scale=spread, size=(per_class, 2))
        points.append(cloud)
        labels.append(np.full(per_class, code, dtype=np.int64))
    return np.concatenate(points), np.concatenate(labels)


def synthesize(per_class, seed):
    """Two synthetic embeddings: baseline (looser) and TPFS-Net (tighter).

    The TPFS-Net panel deliberately uses more separated centers and smaller
    within-class spread so its clusters read as cleaner. This is a designed
    illustration, not a measurement.
    """
    rng = np.random.default_rng(seed)
    # Baseline: overlapping-ish centers, larger spread.
    base_centers = np.array([[-3.2, 3.0], [3.4, 3.3],
                             [-3.0, -3.1], [3.1, -2.8]])
    baseline, labels = _blobs(rng, base_centers, spread=1.65,
                              per_class=per_class)
    # TPFS-Net: pushed apart, tighter — same label order and count.
    ours_centers = np.array([[-5.0, 4.6], [5.1, 4.8],
                             [-5.0, -4.7], [5.0, -4.6]])
    ours, labels_o = _blobs(rng, ours_centers, spread=0.92,
                            per_class=per_class)
    assert np.array_equal(labels, labels_o)
    return baseline, ours, labels


def watermark(fig):
    fig.text(0.5, 0.5, DISCLAIMER, ha='center', va='center', fontsize=34,
             color='#b22222', alpha=0.16, rotation=27, weight='bold',
             zorder=1000)
    fig.text(0.5, 0.012, DISCLAIMER, ha='center', va='bottom', fontsize=8,
             color='#b22222', weight='bold', zorder=1000)


def make_figure(prefix, baseline, ours, labels):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, embedding, title in zip(
            axes, (baseline, ours), ('(a) Hyper-GCN', '(b) TPFS-Net')):
        for code, (color, marker) in enumerate(zip(FIGURE_COLORS,
                                                   FIGURE_MARKERS)):
            mask = labels == code
            ax.scatter(embedding[mask, 0], embedding[mask, 1], color=color,
                       marker=marker, s=14, alpha=0.75, linewidths=0.25,
                       edgecolors='white', rasterized=True)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        for spine in ax.spines.values():
            spine.set_color('#c3c2b7')
            spine.set_linewidth(0.7)
    handles = [Line2D([], [], linestyle='none', marker=FIGURE_MARKERS[c],
                      markerfacecolor=FIGURE_COLORS[c], markeredgecolor='white',
                      markeredgewidth=0.4, markersize=5.5,
                      label=f'Class {CLASS_LABELS[c]}')
               for c in range(len(CLASS_LABELS))]
    fig.legend(handles=handles, loc='upper center', ncol=len(CLASS_LABELS),
               bbox_to_anchor=(0.5, 0.995), frameon=False, handletextpad=0.4,
               columnspacing=1.25)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.82,
                        wspace=0.18)
    watermark(fig)
    fig.savefig(f'{prefix}.pdf', dpi=300)
    fig.savefig(f'{prefix}.png', dpi=300)
    plt.close(fig)
    print(f'[saved] {prefix}.pdf + .png  ({DISCLAIMER})')


def save_exports(prefix, baseline, ours, labels, args):
    csv_path = f'{prefix}_points.csv'
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['synthetic', 'disclaimer', 'index', 'class_label',
                         'baseline_x', 'baseline_y', 'ours_x', 'ours_y'])
        for i in range(len(labels)):
            writer.writerow(['true', DISCLAIMER, i, CLASS_LABELS[labels[i]],
                             baseline[i, 0], baseline[i, 1],
                             ours[i, 0], ours[i, 1]])
    meta = {
        'synthetic': True,
        'submission_eligible': False,
        'disclaimer': DISCLAIMER,
        'purpose': 'layout testing and target planning only; not measured data',
        'generator': 'tools/vis_synthetic_feature_tsne.py',
        'per_class': args.per_class,
        'seed': args.seed,
        'class_labels': list(CLASS_LABELS),
        'note': ('Point clouds are drawn from designed Gaussian blobs; the '
                 'TPFS-Net panel is deliberately tighter. Replace with real '
                 'tools/vis_feature_tsne.py output before any real use.'),
    }
    meta_path = f'{prefix}_meta.json'
    Path(meta_path).write_text(json.dumps(meta, indent=2) + '\n')
    print(f'[saved] {csv_path}, {meta_path}')


def main(argv=None):
    args = parse_args(argv)
    if args.per_class <= 1:
        raise ValueError('--per-class must be at least 2')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    prefix = str(Path(args.out).expanduser())
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    baseline, ours, labels = synthesize(args.per_class, args.seed)
    make_figure(prefix, baseline, ours, labels)
    save_exports(prefix, baseline, ours, labels, args)
    print(f'\n{DISCLAIMER}: this figure is illustrative only and must not be '
          'used as experimental evidence.')


if __name__ == '__main__':
    main()
