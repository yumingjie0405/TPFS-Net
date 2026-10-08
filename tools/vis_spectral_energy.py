"""
Spectral energy retention curve.

For each layer, projects the row-normalized adjacency onto the eigenbasis
of the physical skeleton Laplacian and plots the cumulative energy ratio
captured by the first k eigenvectors. A "topology-preserving" model
should keep most of the energy in the lowest few modes.

Usage:
  python -m tools.vis_spectral_energy \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --baseline-config config/base/nturgbd-cross-subject/hyper_joint.yaml \
      --baseline-work-dir result_joint \
      --out vis/figs_paper/fig_spectral_energy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tools import _style
from tools._io import (build_model, ensure_outdir, latest_or_best_ckpt,
                       load_config, load_weights)


def laplacian_basis(model):
    nrn = model.num_point
    A = model.l1.gcn1.PA.detach().cpu().numpy().mean(axis=0)[:nrn, :nrn]
    A = (A + A.T) * 0.5
    np.fill_diagonal(A, 0)
    A = (A > 0.05).astype(np.float64)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(nrn) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs


def per_layer_energy(model, eigvecs):
    nrn = model.num_point
    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]
    out = []
    for layer in layers:
        gcn = layer.gcn1
        A = (gcn.edge_importance * gcn.PA).detach().cpu().numpy().mean(axis=0)
        A = A[:nrn, :nrn]
        d = np.maximum(np.abs(A).sum(axis=1, keepdims=True), 1e-8)
        An = A / d
        coeff = eigvecs.T @ An @ eigvecs
        e = np.diag(coeff) ** 2
        e = e / max(e.sum(), 1e-8)
        out.append(e)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--baseline-config', default=None)
    ap.add_argument('--baseline-work-dir', default=None)
    ap.add_argument('--baseline-ckpt', default=None)
    ap.add_argument('--out', default='vis/figs_paper/fig_spectral_energy')
    args = ap.parse_args()

    _style.apply()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    ours = build_model(cfg, device)
    load_weights(ours, args.ckpt or latest_or_best_ckpt(args.work_dir), device)

    eigvals, eigvecs = laplacian_basis(ours)
    E_o = per_layer_energy(ours, eigvecs)

    has_base = args.baseline_config and args.baseline_work_dir
    if has_base:
        bcfg = load_config(args.baseline_config)
        base = build_model(bcfg, device)
        load_weights(base,
                     args.baseline_ckpt or latest_or_best_ckpt(args.baseline_work_dir),
                     device)
        E_b = per_layer_energy(base, eigvecs)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw=dict(wspace=0.32))

    cum_o = np.cumsum(E_o, axis=1)
    L_count, V = cum_o.shape
    cmap = plt.get_cmap('plasma', L_count)
    for l in range(L_count):
        axes[0].plot(np.arange(1, V + 1), cum_o[l], color=cmap(l),
                     linewidth=1.4, label=f'L{l+1}')
    axes[0].set_xlabel('# eigenvectors (low to high freq)')
    axes[0].set_ylabel('cumulative energy ratio')
    axes[0].set_title('(a) ours: spectral energy retention')
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc='lower right')

    if has_base:
        cum_b = np.cumsum(E_b, axis=1)
        ratio_o = cum_o[:, 4]
        ratio_b = cum_b[:, 4]
        x = np.arange(L_count)
        w = 0.36
        axes[1].bar(x - w/2, ratio_o, w, color=_style.PALETTE_OURS,
                    label='Ours')
        axes[1].bar(x + w/2, ratio_b, w, color=_style.PALETTE_BASE,
                    label='Baseline')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'L{i+1}' for i in range(L_count)])
        axes[1].set_ylabel('low-freq energy (top-5 modes)')
        axes[1].set_title('(b) low-frequency energy share per layer')
        axes[1].legend(frameon=False)
    else:
        ratio_o = cum_o[:, 4]
        x = np.arange(L_count)
        axes[1].bar(x, ratio_o, 0.6, color=_style.PALETTE_OURS)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'L{i+1}' for i in range(L_count)])
        axes[1].set_ylabel('low-freq energy (top-5 modes)')
        axes[1].set_title('(b) low-frequency energy share per layer')

    _style.save_dual(fig, str(Path(args.out)))
    print('[saved]', f'{args.out}.pdf')


if __name__ == '__main__':
    main()
