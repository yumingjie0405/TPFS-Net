"""
Confusion-matrix delta heatmap: (ours - baseline) row-normalized.
Cells where ours improves are green, where ours regresses are red.

Inputs come from the *_right.txt files saved by main.py at the end of training:
  format per line: ``predicted_label,true_label``.

Usage:
  python -m tools.vis_confusion_delta \
      --ours work_dir/ts_hgcn/ntu60/xsub/joint/runs-140-87640_right.txt \
      --base result_joint/runs-137-134808_right.txt \
      --num-class 60 --top 30 --out vis/figs_paper/fig_confusion_delta
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools import _style


def read_right(path):
    pairs = np.loadtxt(path, delimiter=',', dtype=int)
    return pairs[:, 0], pairs[:, 1]


def confusion_row_norm(pred, true, C):
    cm = np.zeros((C, C), dtype=np.float64)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--num-class', type=int, default=60)
    ap.add_argument('--top', type=int, default=30,
                    help='restrict to first N classes for readability')
    ap.add_argument('--out', default='vis/figs_paper/fig_confusion_delta')
    args = ap.parse_args()

    _style.apply()

    p_o, t_o = read_right(args.ours)
    p_b, t_b = read_right(args.base)

    if t_o.shape != t_b.shape or not np.array_equal(t_o, t_b):
        print('[warn] ours and baseline test orderings differ; '
              'falling back to per-class accuracy comparison only')

    C = args.num_class
    cm_o = confusion_row_norm(p_o, t_o, C)
    cm_b = confusion_row_norm(p_b, t_b, C)
    delta = cm_o - cm_b

    K = min(args.top, C)
    delta_v = delta[:K, :K]

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.out).name

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    vmax = max(abs(delta_v.min()), abs(delta_v.max()), 1e-3)
    im = ax.imshow(delta_v, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                   aspect='equal')
    ax.set_xlabel('predicted class'); ax.set_ylabel('true class')
    ax.set_title(f'Confusion delta = ours - baseline (top {K} classes)')
    ax.set_xticks(np.arange(0, K, max(1, K // 12)))
    ax.set_yticks(np.arange(0, K, max(1, K // 12)))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label='row-normalized rate diff')
    _style.save_dual(fig, str(Path(args.out)))
    print('[saved]', f'{args.out}.pdf')

    diag_diff = np.diag(cm_o) - np.diag(cm_b)
    print(f'classes improved: {(diag_diff > 0).sum()}/{C}, '
          f'mean improvement: {diag_diff.mean():.4f}, '
          f'overall ours acc={np.mean(p_o == t_o):.4f}, '
          f'baseline acc={np.mean(p_b == t_b):.4f}')


if __name__ == '__main__':
    main()
