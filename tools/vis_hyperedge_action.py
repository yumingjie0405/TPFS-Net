"""
Static hyper-edge visualization on selected actions.

For each chosen NTU action class, picks one representative test sample,
forwards it to a chosen GCN layer, computes the adaptive hyper-graph H,
selects the top-K most active hyper-edges, and draws the 3D skeleton with
the joints inside any selected hyper-edge highlighted in **red**.

Usage:
  # 一张图 (单个样本, 4 个超边, 一个视角)
  python -m tools.vis_hyperedge_action \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --layer 5 --top-edges 4 --classes 0 \
      --out vis/figs_paper/fig_hyperedge_drink

  # 多类对比 (README 8 个动作的 NTU 类别 id 0/8/9/26/25/36/46/23)
  python -m tools.vis_hyperedge_action \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --layer 5 --top-edges 3 \
      --classes 0 8 9 26 25 36 46 23 \
      --out vis/figs_paper/fig_hyperedge_grid
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
from tools._ntu_meta import JOINT_NAMES_25
from tools.vis_skeleton_hypergraph import compute_H


NTU60_ACTIONS = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop',
    'pick up', 'throw', 'sit down', 'stand up', 'clapping',
    'reading', 'writing', 'tear up paper', 'put on jacket', 'take off jacket',
    'put on shoe', 'take off shoe', 'put on glasses', 'take off glasses',
    'put on a hat', 'take off a hat', 'cheer up', 'hand waving',
    'kicking something', 'reach into pocket', 'hopping', 'jump up',
    'phone call', 'play with phone', 'type on keyboard', 'point to something',
    'taking a selfie', 'check time', 'rub two hands', 'nod head',
    'shake head', 'wipe face', 'salute', 'put palms together', 'cross hands',
    'sneeze/cough', 'staggering', 'falling down', 'headache', 'chest pain',
    'back pain', 'neck pain', 'nausea/vomit', 'fan self', 'punch/slap',
    'kicking', 'pushing', 'pat on back', 'point finger', 'hugging',
    'giving object', 'touch pocket', 'shaking hands', 'walking towards',
    'walking apart',
]


def extract_coords(sample, frame='mid'):
    """sample: (1, C, T, V, M). Return (V, 3) coords for person 0 at one frame.

    NTU sequences are zero-padded to window_size, so we first find the range
    of valid (non-zero) frames, then pick mid / first / last of that range.
    """
    arr = sample[0, :, :, :, 0].cpu().numpy()  # (C, T, V)
    valid = np.abs(arr).sum(axis=(0, 2)) > 1e-6
    if valid.any():
        idxs = np.where(valid)[0]
        if frame == 'mid':
            t = int(idxs[len(idxs) // 2])
        elif frame == 'first':
            t = int(idxs[0])
        elif frame == 'last':
            t = int(idxs[-1])
        else:
            t = int(frame)
    else:
        t = arr.shape[1] // 2
    return arr[:, t, :].T  # (V, 3)


def collect_samples_per_class(loader, target_classes):
    """Return dict {class_id: (data_tensor, sample_index_in_loader)}."""
    found = {}
    target_set = set(target_classes)
    for batch_idx, (data, label, index) in enumerate(loader):
        for i in range(label.size(0)):
            c = int(label[i])
            if c in target_set and c not in found:
                found[c] = data[i:i + 1].clone()
                if len(found) == len(target_set):
                    return found
    return found


def forward_to_layer_H(model, sample, layer_idx):
    """Forward one sample (1,C,T,V,M) up to layer_idx-1 then compute H of that layer."""
    layers = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9]
    target = layers[layer_idx - 1]
    pre = layers[:layer_idx - 1]

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
    for pl in pre:
        x, _ = pl(x)
    H = compute_H(target.gcn1, x)
    return H


def select_hyper_edges(H_np, top_edges, num_neighbors):
    edge_weight = H_np.sum(axis=1)
    centers = np.argsort(-edge_weight)[:top_edges]
    out = []
    for c in centers:
        nbrs = np.argsort(-H_np[c])[:num_neighbors].tolist()
        out.append((int(c), [int(v) for v in nbrs]))
    return out


def draw_one(ax, coords, hyper_edges, view, title):
    """Draw 3D skeleton; joints inside any hyper-edge are red, else light gray.

    NTU axis convention: X = left-right, Y = up, Z = depth (camera).
    Matplotlib 3D has Z up, so we map (NTU_x, NTU_z, NTU_y) -> (mpl_x, mpl_y, mpl_z).
    """
    # remap to mpl convention so the person stands upright
    X = coords[:, 0]
    Y = coords[:, 2]
    Z = coords[:, 1]

    elev, azim = view
    ax.view_init(elev=elev, azim=azim)

    selected = set()
    for src, nbrs in hyper_edges:
        selected.add(src)
        selected.update(nbrs)

    # physical bones — make them clearly visible
    for (i, j) in NTU_BONES:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                color='#555555', linewidth=1.6, alpha=0.95, zorder=1)

    # joints
    sel_mask = np.array([j in selected for j in range(coords.shape[0])])
    if (~sel_mask).any():
        ax.scatter(X[~sel_mask], Y[~sel_mask], Z[~sel_mask],
                   s=26, c='#aaaaaa', edgecolors='k', linewidths=0.3,
                   depthshade=False, zorder=4)
    if sel_mask.any():
        ax.scatter(X[sel_mask], Y[sel_mask], Z[sel_mask],
                   s=95, c='#d62728', edgecolors='k', linewidths=0.7,
                   depthshade=False, zorder=5)

    # equal aspect that preserves human proportions
    rngs = np.array([X.max() - X.min(),
                     Y.max() - Y.min(),
                     Z.max() - Z.min()])
    rng = max(rngs.max(), 1e-6)
    cx, cy, cz = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2, (Z.max() + Z.min()) / 2
    ax.set_xlim(cx - rng / 2, cx + rng / 2)
    ax.set_ylim(cy - rng / 2, cy + rng / 2)
    ax.set_zlim(cz - rng / 2, cz + rng / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--layer', type=int, default=5,
                    help='which GCN layer (1..9) to extract H from')
    ap.add_argument('--top-edges', type=int, default=4,
                    help='how many top hyper-edges to highlight per sample')
    ap.add_argument('--num-neighbors', type=int, default=6,
                    help='how many neighbors per hyper-edge to draw')
    ap.add_argument('--classes', type=int, nargs='+', default=[0],
                    help='NTU60 class ids (0-indexed) to visualize')
    ap.add_argument('--scan-batches', type=int, default=20,
                    help='max test batches to scan when picking one sample per class')
    ap.add_argument('--view', type=float, nargs=2, default=(8, -75),
                    metavar=('ELEV', 'AZIM'),
                    help='3D view (elevation, azimuth) in degrees')
    ap.add_argument('--frame', default='mid',
                    help="which frame: mid|first|last|<int> (default 'mid')")
    ap.add_argument('--out', default='vis/figs_paper/fig_hyperedge_action')
    args = ap.parse_args()

    _style.apply()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    model = build_model(cfg, device)
    load_weights(model, args.ckpt or latest_or_best_ckpt(args.work_dir), device)

    # build a small loader, scan a few batches to grab one sample per class
    loader = build_test_loader(cfg, batch_size=64, num_workers=0,
                               max_samples=args.scan_batches * 64)
    found = collect_samples_per_class(loader, args.classes)
    missing = [c for c in args.classes if c not in found]
    if missing:
        print(f'[warn] could not find samples for classes {missing}; '
              f'increase --scan-batches')

    n = len(args.classes)
    if n == 1:
        cols, rows = 1, 1
        fig_w, fig_h = 5.6, 5.6
    elif n <= 4:
        cols, rows = n, 1
        fig_w, fig_h = 4.5 * cols, 4.7
    else:
        cols = 4
        rows = (n + cols - 1) // cols
        fig_w, fig_h = 3.7 * cols, 3.9 * rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    for idx, c in enumerate(args.classes):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        if c not in found:
            ax.set_axis_off()
            ax.set_title(f'class {c}: not found', fontsize=9)
            continue
        sample = found[c].float().to(device)  # (1,C,T,V,M)
        coords = extract_coords(sample, frame=args.frame)  # (V,3)
        with torch.no_grad():
            H = forward_to_layer_H(model, sample, args.layer)
        H_np = H[0].mean(dim=0).cpu().numpy()
        nrn = model.num_point
        H_np = H_np[:nrn, :nrn]
        edges = select_hyper_edges(H_np, args.top_edges, args.num_neighbors)
        name = NTU60_ACTIONS[c] if 0 <= c < len(NTU60_ACTIONS) else f'class {c}'
        draw_one(ax, coords, edges, tuple(args.view), name)

    fig.suptitle(f'Selected hyper-edges (layer {args.layer}, top-{args.top_edges}); '
                 f'red = joints inside selected hyper-edges',
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path(ensure_outdir(Path(args.out).parent))
    base = Path(args.out).name
    _style.save_dual(fig, str(out_dir / base))
    print('[saved]', out_dir / f'{base}.pdf')


if __name__ == '__main__':
    main()
