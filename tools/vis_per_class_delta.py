"""
Per-class accuracy delta bar chart: ours vs baseline.

Reads the same *_right.txt files. Sorts by delta and shows the top/bottom
``--top`` classes so reviewers see where the gain comes from.

Usage:
  python -m tools.vis_per_class_delta \
      --ours work_dir/ts_hgcn/ntu60/xsub/joint/runs-140-87640_right.txt \
      --base result_joint/runs-137-134808_right.txt \
      --num-class 60 --top 15 --out vis/figs_paper/fig_per_class_delta
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools import _style


def per_class_acc(path, C):
    pairs = np.loadtxt(path, delimiter=',', dtype=int)
    pred, true = pairs[:, 0], pairs[:, 1]
    acc = np.zeros(C)
    cnt = np.zeros(C)
    for p, t in zip(pred, true):
        cnt[t] += 1
        if p == t:
            acc[t] += 1
    return acc / np.maximum(cnt, 1.0), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--num-class', type=int, default=60)
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out', default='vis/figs_paper/fig_per_class_delta')
    args = ap.parse_args()

    _style.apply()

    a_o, _ = per_class_acc(args.ours, args.num_class)
    a_b, _ = per_class_acc(args.base, args.num_class)
    delta = a_o - a_b

    order = np.argsort(delta)
    bot = order[:args.top]
    top = order[-args.top:][::-1]

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw=dict(wspace=0.3))
    for ax, idx, title, color in [
        (axes[0], top, f'(a) top-{args.top} gains', _style.PALETTE_OURS),
        (axes[1], bot, f'(b) bottom-{args.top} (regressions)',
         _style.PALETTE_BASE),
    ]:
        ax.barh(np.arange(len(idx)), delta[idx] * 100,
                color=color, edgecolor='k', linewidth=0.5)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([f'cls {c}' for c in idx])
        ax.invert_yaxis()
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel(r'$\Delta$ accuracy (ours - baseline) [%]')
        ax.set_title(title, loc='left')

    fig.suptitle(f'Per-class accuracy delta '
                 f'(overall {a_o.mean()*100:.2f}% vs {a_b.mean()*100:.2f}%)')
    _style.save_dual(fig, str(Path(args.out)))
    print('[saved]', f'{args.out}.pdf')


if __name__ == '__main__':
    main()
