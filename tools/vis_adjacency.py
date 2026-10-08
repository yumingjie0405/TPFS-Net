"""
Layer-wise adjacency comparison: ours vs (optionally) baseline.

If --baseline-config and --baseline-work-dir are given, plots a 2x9 grid
(top: ours; bottom: baseline) with shared colormap, plus per-layer
Frobenius distance to the physical adjacency.

If no baseline is given, plots a 1x9 row of "ours" only.

Usage:
  python -m tools.vis_adjacency \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --baseline-config config/base/nturgbd-cross-subject/hyper_joint.yaml \
      --baseline-work-dir result_joint \
      --out vis/figs_paper/fig_adjacency
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from tools import _style
from tools._io import (build_model, ensure_outdir, latest_or_best_ckpt,
                       load_config, load_weights)


def collect_layer_adj(model):
    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]
    nrn = model.num_point
    out = []
    for layer in layers:
        gcn = layer.gcn1
        A = (gcn.edge_importance * gcn.PA).detach().cpu().numpy()
        A = A.mean(axis=0)[:nrn, :nrn]
        out.append(A)
    return out


def physical_A(model):
    nrn = model.num_point
    A = model.l1.gcn1.PA.detach().cpu().numpy().mean(axis=0)[:nrn, :nrn]
    return A


def normalize(A):
    return A / max(A.max(), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--baseline-config', default=None)
    ap.add_argument('--baseline-work-dir', default=None)
    ap.add_argument('--baseline-ckpt', default=None)
    ap.add_argument('--out', default='vis/figs_paper/fig_adjacency')
    args = ap.parse_args()

    _style.apply()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.config)
    ours = build_model(cfg, device)
    load_weights(ours, args.ckpt or latest_or_best_ckpt(args.work_dir), device)
    ours_mats = [normalize(A) for A in collect_layer_adj(ours)]
    A_phys = normalize(physical_A(ours))

    has_base = args.baseline_config and args.baseline_work_dir
    if has_base:
        bcfg = load_config(args.baseline_config)
        base = build_model(bcfg, device)
        load_weights(base,
                     args.baseline_ckpt or latest_or_best_ckpt(args.baseline_work_dir),
                     device)
        base_mats = [normalize(A) for A in collect_layer_adj(base)]
    else:
        base_mats = None

    out = Path(ensure_outdir(Path(args.out).parent))
    base_name = Path(args.out).name

    rows = 2 if has_base else 1
    fig, axes = plt.subplots(rows, 9, figsize=(13.5, 1.7 * rows + 1.2),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.1, hspace=0.18))
    axes = np.atleast_2d(axes)

    for i, A in enumerate(ours_mats):
        ax = axes[0, i]
        im = ax.imshow(A, cmap='magma', vmin=0, vmax=1)
        d = np.linalg.norm(A - A_phys)
        ax.set_title(f'L{i+1}\n$\\Vert\\cdot\\Vert_F^{{\\Delta}}$={d:.2f}',
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Ours', fontsize=11)

    if has_base:
        for i, A in enumerate(base_mats):
            ax = axes[1, i]
            ax.imshow(A, cmap='magma', vmin=0, vmax=1)
            d = np.linalg.norm(A - A_phys)
            ax.set_title(f'$\\Vert\\cdot\\Vert_F^{{\\Delta}}$={d:.2f}',
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel('Baseline', fontsize=11)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.012, pad=0.012)
    cbar.set_label('normalized weight', rotation=90, labelpad=4)
    fig.suptitle('Layer-wise learned adjacency vs physical skeleton',
                 fontsize=12, y=1.02)
    _style.save_dual(fig, str(out / base_name))

    # quantitative bar
    fig2, ax2 = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(9)
    w = 0.36
    o_d = [np.linalg.norm(A - A_phys) for A in ours_mats]
    ax2.bar(x - (w/2 if has_base else 0), o_d, w if has_base else 0.6,
            color=_style.PALETTE_OURS, label='Ours')
    if has_base:
        b_d = [np.linalg.norm(A - A_phys) for A in base_mats]
        ax2.bar(x + w/2, b_d, w, color=_style.PALETTE_BASE, label='Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i+1}' for i in range(9)])
    ax2.set_ylabel(r'$\Vert A_l - A_{\mathrm{phys}}\Vert_F$')
    ax2.set_title('Distance from physical skeleton (lower = more topology-preserving)')
    ax2.legend(frameon=False)
    _style.save_dual(fig2, str(out / f'{base_name}_distance'))

    print('[saved]', out / f'{base_name}.pdf')
    print('[saved]', out / f'{base_name}_distance.pdf')


if __name__ == '__main__':
    main()
