"""
Skeleton + hyper-edge overlay, paper grade.

Three improvements over the simple version:
  * 2x2 multi-view layout (front, side, top, oblique) for the same skeleton;
  * joint dots colored by *semantic body part*, so the eye groups them;
  * hyper-edges drawn with class-distinct colors and thicker strokes,
    with a small legend block at the bottom.

Usage:
  python -m tools.vis_skeleton_hypergraph \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --layer 5 --sample-idx 0 --top-edges 4 \
      --out vis/figs_paper/fig_hyperedges_l5_s0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from graph.ntu_rgb_d import inward as NTU_BONES
from tools import _style
from tools._io import (build_model, build_test_loader, ensure_outdir,
                       latest_or_best_ckpt, load_config, load_weights)
from tools._ntu_meta import (JOINT_NAMES_25, joint_to_part_color)


def compute_H(gcn, x):
    N, C, T, V = x.size()
    h_x = gcn.hyper_joint
    h_x = h_x.T.unsqueeze(1)
    x_aug = torch.cat([x, h_x.repeat(N, 1, T, 1)], dim=-1)

    t_x = x_aug.mean(2)
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


def draw_skeleton(ax, p, joint_colors, hyper_edges, edge_colors, view):
    elev, azim = view
    ax.view_init(elev=elev, azim=azim)
    for (i, j) in NTU_BONES:
        ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                [p[i, 2], p[j, 2]], color='gray', linewidth=0.8, alpha=0.85)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=28, c=joint_colors,
               edgecolors='k', linewidths=0.4, depthshade=False)
    for (src, nbr), color in zip(hyper_edges, edge_colors):
        for v in nbr:
            ax.plot([p[src, 0], p[v, 0]], [p[src, 1], p[v, 1]],
                    [p[src, 2], p[v, 2]], color=color, linewidth=2.2,
                    alpha=0.7)
        ax.scatter(p[src, 0], p[src, 1], p[src, 2], s=85,
                   c=[color], edgecolors='k', linewidths=0.7,
                   depthshade=False)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--layer', type=int, default=5)
    ap.add_argument('--sample-idx', type=int, default=0)
    ap.add_argument('--top-edges', type=int, default=4)
    ap.add_argument('--num-neighbors', type=int, default=6)
    ap.add_argument('--out', default='vis/figs_paper/fig_hyperedges')
    args = ap.parse_args()

    _style.apply()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    model = build_model(cfg, device)
    load_weights(model,
                 args.ckpt or latest_or_best_ckpt(args.work_dir), device)

    loader = build_test_loader(cfg, batch_size=args.sample_idx + 1,
                               num_workers=0,
                               max_samples=args.sample_idx + 1)
    data, label, _ = next(iter(loader))
    data = data.float().to(device)
    coords = data[args.sample_idx, :, :, :, 0].cpu().numpy().mean(axis=1).T

    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]
    pre_layers = layers[:args.layer - 1]
    layer = layers[args.layer - 1]

    with torch.no_grad():
        N, C, T, V_in, M = data.shape
        x = data.permute(0, 4, 2, 3, 1).contiguous()
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
        for pl in pre_layers:
            x, _ = pl(x)
        H = compute_H(layer.gcn1, x)

    H_np = H[args.sample_idx].mean(dim=0).cpu().numpy()
    nrn = model.num_point
    H_np = H_np[:nrn, :nrn]
    edge_weight = H_np.sum(axis=1)
    centers = np.argsort(-edge_weight)[:args.top_edges]
    hyper_edges = [(int(c),
                    np.argsort(-H_np[c])[:args.num_neighbors].tolist())
                   for c in centers]

    edge_palette = plt.get_cmap('tab10')
    edge_colors = [edge_palette(i) for i in range(len(hyper_edges))]
    joint_colors = joint_to_part_color(_style.SEMANTIC_PARTS)

    out_dir = Path(ensure_outdir(Path(args.out).parent))
    base_name = Path(args.out).name

    fig = plt.figure(figsize=(11, 9.5))
    views = [('front', (15, -90)), ('side', (15, 0)),
             ('top',   (85, -90)), ('oblique', (25, -45))]
    axes = []
    for i, (name, vw) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        draw_skeleton(ax, coords, joint_colors, hyper_edges, edge_colors, vw)
        ax.set_title(f'({chr(97+i)}) {name} view', loc='left')
        axes.append(ax)

    handles = []
    for k, (src, _) in enumerate(hyper_edges):
        handles.append(plt.Line2D([0], [0], color=edge_colors[k], lw=2.4,
                                   marker='o', markeredgecolor='k',
                                   markersize=7,
                                   label=f'hyper-edge {k+1} @ {JOINT_NAMES_25[src]}'))
    for part, c in _style.SEMANTIC_PARTS.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=c, markeredgecolor='k',
                                   markersize=7, label=part))
    fig.legend(handles=handles, loc='lower center', frameon=False, ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(f'Layer {args.layer} hyper-edges on sample {args.sample_idx}'
                 f' (label {int(label[args.sample_idx])})')
    _style.save_dual(fig, str(out_dir / base_name))
    print('[saved]', out_dir / f'{base_name}.pdf')


if __name__ == '__main__':
    main()
