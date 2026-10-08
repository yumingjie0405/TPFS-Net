"""
Direction 1: K-stage hyper-edge grid.

For each of N selected NTU action classes, picks one representative test
sample, forwards it through the model, and reads the SPG stage attention
alpha_k(t). For each stage k it re-computes the layer-L hyper-graph H using
alpha_k as the temporal weighting (instead of mean over time). The top
hyper-edges of each stage-specific H are highlighted on a 3D skeleton.

Layout: rows = action classes, columns = K stages (+ 1 trailing column with
the alpha_k(t) curves). Joints inside any selected hyper-edge are red.

Usage:
  python -m tools.vis_stage_hyperedge \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --layer 9 --top-edges 3 --num-neighbors 6 \
      --classes 0 8 9 26 36 46 \
      --out vis/figs_paper/fig_stage_hyperedge
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from graph.ntu_rgb_d import inward as NTU_BONES
from tools import _style
from tools._io import (build_model, build_test_loader, ensure_outdir,
                       latest_or_best_ckpt, load_config, load_weights)
from tools.vis_hyperedge_action import (NTU60_ACTIONS, collect_samples_per_class,
                                        extract_coords, select_hyper_edges)


def compute_H_weighted(gcn, x, w):
    """compute_H but with explicit temporal weights w of shape (N, T)."""
    N, C, T, V = x.size()
    h_x = gcn.hyper_joint.T.unsqueeze(1)
    x_aug = torch.cat([x, h_x.repeat(N, 1, T, 1)], dim=-1)

    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    t_x = torch.einsum('nt,nctv->ncv', w, x_aug)

    v_x = gcn.to_V(t_x)
    Vtot = x_aug.size(-1)
    dis_v_x = v_x.view(N, gcn.num_subset, gcn.hidden_channels, Vtot)
    dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous()
    distance = torch.cdist(dis_v_x, dis_v_x)
    H = torch.zeros_like(distance)

    topk_v, topk_idx = torch.topk(distance, 9, largest=False)
    topk_v = gcn.softmax(-topk_v)
    H = torch.scatter(H, 3, topk_idx, topk_v)
    return H


def forward_with_stages(model, sample, target_layer):
    """Run the full forward pass; return (x_at_target_layer_input, stage_attn).

    x_at_target_layer_input has shape (N*M, c, T_layer, V).
    stage_attn has shape (N, K, T_seq) where T_seq is the temporal size after
    the last spatial GCN block.
    """
    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]

    N, C, T, V_in, M = sample.shape
    x = sample.permute(0, 4, 2, 3, 1).contiguous()
    x = model.to_joint_embedding(x)
    x = x + model.pos_embedding[:, :model.num_point]
    x = x + model.hk_encoder().unsqueeze(0).unsqueeze(0)
    x = model.tanh(x)
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(N, M * V_in * model.embedding_channels, T)
    x = model.data_bn(x)
    x = x.view(N, M, V_in, model.embedding_channels, T)
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(N * M, model.embedding_channels, T, V_in)

    captured = {}
    for li, layer in enumerate(layers, start=1):
        if li == target_layer:
            captured['x_in'] = x.clone()
        # mimic ts_hgcn forward residuals
        if li == 3:
            x_prev = captured.get('_x1', x)
            x, _ = layer(x + x_prev)
        elif li == 6:
            x_prev = captured.get('_x4', x)
            x, _ = layer(x + x_prev)
        elif li == 9:
            x_prev = captured.get('_x7', x)
            x, _ = layer(x + x_prev)
        else:
            x, _ = layer(x)
            if li == 1:
                captured['_x1'] = x
            if li == 4:
                captured['_x4'] = x
            if li == 7:
                captured['_x7'] = x

    c_new = x.size(1)
    x_per_person = x.view(N, M, c_new, x.size(2), x.size(3))
    x_seq = x_per_person.mean(dim=1)

    _stage_feats, stage_attn = model.stage_decomposer(x_seq)
    return captured['x_in'], stage_attn  # (N*M, c_l, T_l, V), (N, K, T_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--layer', type=int, default=9,
                    help='extract H from this GCN layer (1-9, default 9 to '
                         'match SPG temporal dim)')
    ap.add_argument('--top-edges', type=int, default=3)
    ap.add_argument('--num-neighbors', type=int, default=6)
    ap.add_argument('--classes', type=int, nargs='+',
                    default=[0, 8, 9, 26, 36, 46])
    ap.add_argument('--scan-batches', type=int, default=20)
    ap.add_argument('--view', type=float, nargs=2, default=(8, -75),
                    metavar=('ELEV', 'AZIM'))
    ap.add_argument('--reset-order-bias', action='store_true',
                    help='at inference, restore stage_decomposer.order_bias to '
                         'its initial linspace(-1, 1, K) so each stage is '
                         'forced onto its intended temporal window.')
    ap.add_argument('--out', default='vis/figs_paper/fig_stage_hyperedge')
    args = ap.parse_args()

    _style.apply()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    K = int(cfg['model_args'].get('num_stages', 4))
    model = build_model(cfg, device)
    load_weights(model, args.ckpt or latest_or_best_ckpt(args.work_dir), device)
    if args.reset_order_bias:
        with torch.no_grad():
            new_bias = torch.linspace(-1.0, 1.0, K, device=device)
            model.stage_decomposer.order_bias.data.copy_(new_bias)
        print(f'[reset] order_bias <- {new_bias.cpu().numpy()}')

    loader = build_test_loader(cfg, batch_size=64, num_workers=0,
                               max_samples=args.scan_batches * 64)
    found = collect_samples_per_class(loader, args.classes)
    missing = [c for c in args.classes if c not in found]
    if missing:
        print(f'[warn] missing samples for classes {missing}; '
              f'increase --scan-batches')

    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]
    target_gcn = layers[args.layer - 1].gcn1

    n_rows = len(args.classes)
    n_cols = K + 1  # K stages + alpha curve column
    fig = plt.figure(figsize=(3.5 * n_cols + 0.4, 3.5 * n_rows))

    for ri, c in enumerate(args.classes):
        if c not in found:
            for kc in range(n_cols):
                ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + kc + 1,
                                     projection='3d' if kc < K else None)
                ax.set_axis_off()
                if kc == 0:
                    ax.set_title(f'class {c}: not found', fontsize=9)
            continue

        sample = found[c].float().to(device)
        coords = extract_coords(sample, frame='mid')
        with torch.no_grad():
            x_in, stage_attn = forward_with_stages(model, sample, args.layer)
            T_layer = x_in.size(2)
            T_seq = stage_attn.size(2)
            if T_layer != T_seq:
                attn = F.interpolate(stage_attn, size=T_layer, mode='linear',
                                     align_corners=False)
            else:
                attn = stage_attn
            attn = attn[0]  # (K, T_layer)

        nrn = model.num_point
        per_stage_joints = []
        for k in range(K):
            w = attn[k:k + 1].repeat(x_in.size(0), 1)  # (N*M, T)
            H = compute_H_weighted(target_gcn, x_in, w)
            H_np = H.mean(dim=(0, 1)).detach().cpu().numpy()[:nrn, :nrn]
            # joint importance = in-degree + out-degree in stage-specific hyper-graph
            scores = H_np.sum(axis=0) + H_np.sum(axis=1)
            # take top-N joints sorted by importance (most important first)
            top_n = max(args.top_edges, 5)
            ranked = np.argsort(-scores)[:top_n]
            per_stage_joints.append([(int(j), float(scores[j])) for j in ranked])

        for k, ranked in enumerate(per_stage_joints):
            ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + k + 1,
                                 projection='3d')
            draw_skeleton(ax, coords, ranked, tuple(args.view))
            cls_name = NTU60_ACTIONS[c] if 0 <= c < len(NTU60_ACTIONS) \
                else f'class {c}'
            if ri == 0:
                ax.set_title(f'Stage {k + 1}', fontsize=18, pad=4,
                             fontweight='bold')
            if k == 0:
                ax.text2D(-0.03, 0.5, cls_name,
                          transform=ax.transAxes,
                          rotation=90, va='center', ha='right',
                          fontsize=18, fontweight='bold')

        ax_curve = fig.add_subplot(n_rows, n_cols, ri * n_cols + K + 1)
        attn_np = stage_attn[0].cpu().numpy()
        cmap = plt.get_cmap('viridis', K)
        T_seq = attn_np.shape[1]
        ts = np.arange(T_seq)
        for k in range(K):
            ax_curve.plot(ts, attn_np[k], color=cmap(k),
                          linewidth=1.8, label=f'k={k + 1}')
        ax_curve.set_xticks([0, T_seq // 2, T_seq - 1])
        ax_curve.set_yticks([0, attn_np.max() * 1.0])
        ax_curve.set_yticklabels([])
        ax_curve.tick_params(labelsize=10)
        if ri == 0:
            ax_curve.set_title(r'$\alpha_k(t)$', fontsize=18, pad=4,
                               fontweight='bold')
            ax_curve.legend(frameon=False, fontsize=10, ncol=K, loc='upper center',
                            bbox_to_anchor=(0.5, -0.05))
        for spine in ('top', 'right'):
            ax_curve.spines[spine].set_visible(False)

    plt.tight_layout()

    out_dir = Path(ensure_outdir(Path(args.out).parent))
    base = Path(args.out).name
    _style.save_dual(fig, str(out_dir / base))
    print('[saved]', out_dir / f'{base}.pdf')


def draw_skeleton(ax, coords, ranked_joints, view):
    """ranked_joints: list of (joint_idx, score), sorted by score desc.
    Top-ranked joint gets the largest red glow."""
    X, Y, Z = coords[:, 0], coords[:, 2], coords[:, 1]
    elev, azim = view
    ax.view_init(elev=elev, azim=azim)

    selected_idx = [j for j, _ in ranked_joints]
    selected_set = set(selected_idx)

    for (i, j) in NTU_BONES:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                color='#333333', linewidth=1.8, alpha=1.0, zorder=1)

    sel_mask = np.array([j in selected_set for j in range(coords.shape[0])])
    if (~sel_mask).any():
        ax.scatter(X[~sel_mask], Y[~sel_mask], Z[~sel_mask],
                   s=14, c='#bdbdbd', edgecolors='none',
                   depthshade=False, zorder=2)

    # rank-based size: rank 0 (most important) -> largest, decreasing for lower ranks
    n_sel = len(selected_idx)
    for rank, j in enumerate(selected_idx):
        # scale: 1.0 for rank 0, down to 0.45 for the lowest rank
        scale = 1.0 - 0.55 * (rank / max(n_sel - 1, 1))
        rings = [(520 * scale, 0.10),
                 (320 * scale, 0.16),
                 (180 * scale, 0.26),
                 (95 * scale, 0.50)]
        for rs, ra in rings:
            ax.scatter([X[j]], [Y[j]], [Z[j]],
                       s=rs, c='#d62728', alpha=ra, edgecolors='none',
                       depthshade=False, zorder=3)
        ax.scatter([X[j]], [Y[j]], [Z[j]],
                   s=42 * scale, c='#d62728',
                   edgecolors='white', linewidths=0.6,
                   depthshade=False, zorder=4)

    rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min(), 1e-6)
    pad = rng * 0.42
    cx = (X.max() + X.min()) / 2
    cy = (Y.max() + Y.min()) / 2
    cz = (Z.max() + Z.min()) / 2
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_zlim(cz - pad, cz + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


if __name__ == '__main__':
    main()
