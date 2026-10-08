"""
Direction 2: per-stage joint-activation heatmap on the skeleton.

For each chosen NTU action class, picks one representative test sample,
forwards through the model to capture the last GCN layer's features
f[N, C, T, V] and the SPG stage attention alpha_k(t).

For each stage k, computes per-joint activation:
    act[t, v]   = || f[:, t, v] ||_2 over channels
    act_k[v]    = sum_t alpha_k(t) * act[t, v]
and renders a 3D skeleton with each joint colored by act_k[v] (a continuous
heatmap) — no red points, no extra lines.

Layout: rows = action classes, columns = K stages (+ alpha curve column).

Usage:
  python -m tools.vis_stage_heatmap \
      --config config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml \
      --work-dir work_dir/ts_hgcn/ntu60/xsub/joint \
      --classes 0 8 9 26 36 46 \
      --out vis/figs_paper/fig_stage_heatmap
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F

from graph.ntu_rgb_d import inward as NTU_BONES
from tools import _style
from tools._io import (build_model, build_test_loader, ensure_outdir,
                       latest_or_best_ckpt, load_config, load_weights)
from tools.vis_hyperedge_action import (NTU60_ACTIONS, collect_samples_per_class,
                                        extract_coords)


def forward_capture_last(model, sample):
    """Forward to the end of l9; return (last_feats, stage_attn).

    last_feats: (N, c, T_l, V) per-person-averaged
    stage_attn: (N, K, T_l)
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

    skips = {}
    for li, layer in enumerate(layers, start=1):
        if li == 3:
            x_prev = skips.get('_x1', x); x, _ = layer(x + x_prev)
        elif li == 6:
            x_prev = skips.get('_x4', x); x, _ = layer(x + x_prev)
        elif li == 9:
            x_prev = skips.get('_x7', x); x, _ = layer(x + x_prev)
        else:
            x, _ = layer(x)
            if li == 1:
                skips['_x1'] = x
            if li == 4:
                skips['_x4'] = x
            if li == 7:
                skips['_x7'] = x

    c_new = x.size(1)
    x_per_person = x.view(N, M, c_new, x.size(2), x.size(3))
    feats = x_per_person.mean(dim=1)  # (N, c, T_l, V)
    _, stage_attn = model.stage_decomposer(feats)
    return feats, stage_attn


def draw_topn_skeleton(ax, coords, sel_mask, view):
    """Like Figure A's red/gray skeleton but driven by per-stage activation."""
    X, Y, Z = coords[:, 0], coords[:, 2], coords[:, 1]
    ax.view_init(elev=view[0], azim=view[1])

    for (i, j) in NTU_BONES:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                color='#222222', linewidth=1.6, alpha=1.0, zorder=1)

    sel_mask = np.asarray(sel_mask, dtype=bool)
    if (~sel_mask).any():
        ax.scatter(X[~sel_mask], Y[~sel_mask], Z[~sel_mask],
                   s=18, c='#aaaaaa', edgecolors='white', linewidths=0.4,
                   depthshade=False, zorder=2)
    if sel_mask.any():
        ax.scatter(X[sel_mask], Y[sel_mask], Z[sel_mask],
                   s=80, c='#d62728', edgecolors='white', linewidths=0.6,
                   depthshade=False, zorder=3)

    rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min(), 1e-6)
    cx = (X.max() + X.min()) / 2
    cy = (Y.max() + Y.min()) / 2
    cz = (Z.max() + Z.min()) / 2
    ax.set_xlim(cx - rng / 2, cx + rng / 2)
    ax.set_ylim(cy - rng / 2, cy + rng / 2)
    ax.set_zlim(cz - rng / 2, cz + rng / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def draw_heatmap_skeleton(ax, coords, joint_act, vmin, vmax, view, cmap):
    X, Y, Z = coords[:, 0], coords[:, 2], coords[:, 1]
    ax.view_init(elev=view[0], azim=view[1])

    for (i, j) in NTU_BONES:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                color='#222222', linewidth=1.6, alpha=1.0, zorder=1)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(joint_act))
    sizes = 14 + 28 * norm(joint_act)
    ax.scatter(X, Y, Z, s=sizes, c=colors,
               edgecolors='white', linewidths=0.5,
               depthshade=False, zorder=3)

    rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min(), 1e-6)
    cx = (X.max() + X.min()) / 2
    cy = (Y.max() + Y.min()) / 2
    cz = (Z.max() + Z.min()) / 2
    ax.set_xlim(cx - rng / 2, cx + rng / 2)
    ax.set_ylim(cy - rng / 2, cy + rng / 2)
    ax.set_zlim(cz - rng / 2, cz + rng / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--classes', type=int, nargs='+',
                    default=[0, 8, 9, 26, 36, 46])
    ap.add_argument('--scan-batches', type=int, default=20)
    ap.add_argument('--view', type=float, nargs=2, default=(8, -75),
                    metavar=('ELEV', 'AZIM'))
    ap.add_argument('--cmap', default=None,
                    help='matplotlib cmap name; default depends on --mode '
                         "(weighted -> 'YlOrRd', delta -> 'RdBu_r', "
                         "argmax -> 'tab10')")
    ap.add_argument('--mode', default='topn',
                    choices=['weighted', 'delta', 'zscore', 'argmax', 'topn'],
                    help="topn     = highlight top-N joints per stage (default; "
                                    "best for showing which joints each stage focuses on); "
                         "weighted = sum_t alpha_k * ||f||  (raw magnitude); "
                         "delta    = above MINUS row mean; "
                         "zscore   = (above - mean_k) / std_k per joint; "
                         "argmax   = which stage owns each joint")
    ap.add_argument('--top-n', type=int, default=6,
                    help='number of top-activated joints to highlight per '
                         'stage in topn mode (default 6)')
    ap.add_argument('--norm-mode', default='per-row',
                    choices=['per-row', 'per-cell'],
                    help='per-row: same colorbar across stages of one action; '
                         'per-cell: each cell self-normalised')
    ap.add_argument('--reset-order-bias', action='store_true',
                    help='at inference, restore stage_decomposer.order_bias to '
                         'its initial linspace(-1, 1, K) so each stage is '
                         'forced onto its intended temporal window. Use this '
                         'when training collapsed all stages to one position.')
    ap.add_argument('--out', default='vis/figs_paper/fig_stage_heatmap')
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

    n_rows = len(args.classes)
    # Layout columns: K stage skeletons | alpha-curve | colorbar
    # widths chosen so everything has its own slot; matplotlib won't squeeze
    fig_w = 2.7 * K + 2.4 + 0.6
    fig_h = 2.6 * n_rows + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, K + 2,
                          width_ratios=[1.0] * K + [0.85, 0.10],
                          left=0.04, right=0.98,
                          bottom=0.02, top=0.92,
                          wspace=0.08, hspace=0.18)
    if args.cmap is None:
        default_cmap = {'weighted': 'YlOrRd', 'delta': 'RdBu_r',
                        'zscore': 'RdBu_r', 'argmax': 'tab10',
                        'topn': 'Reds'}[args.mode]
    else:
        default_cmap = args.cmap
    cmap = plt.get_cmap(default_cmap)

    # gather per-stage joint activation: norm of the stage-weighted feature
    # (vs the previous "stage-weighted norm" which collapsed direction first)
    rows = []
    nrn = model.num_point
    for c in args.classes:
        if c not in found:
            rows.append(None)
            continue
        sample = found[c].float().to(device)
        coords = extract_coords(sample, frame='mid')
        with torch.no_grad():
            feats, stage_attn = forward_capture_last(model, sample)
        feats = feats[0]              # (c, T_l, V)
        attn = stage_attn[0]          # (K, T_l)

        # subtract temporal mean to focus on stage-specific deviation
        feats_centered = feats - feats.mean(dim=1, keepdim=True)
        # weight features along time per stage; norm over channels
        feat_kcv = torch.einsum('kt,ctv->kcv', attn, feats_centered)
        act_kv = feat_kcv.norm(dim=1).cpu().numpy()  # (K, V)
        act_kv = act_kv[:, :nrn]
        rows.append(dict(coords=coords, act=act_kv,
                         attn=stage_attn[0].cpu().numpy(),
                         cls=c))

    for ri, row in enumerate(rows):
        if row is None:
            for kc in range(K + 1):
                ax = fig.add_subplot(gs[ri, kc],
                                     projection='3d' if kc < K else None)
                ax.set_axis_off()
                if kc == 0:
                    ax.set_title(f'class {args.classes[ri]}: not found',
                                 fontsize=9)
            continue
        coords = row['coords']
        act = row['act']                # (K, V)
        attn = row['attn']              # (K, T_l)

        # post-process per --mode
        if args.mode == 'weighted':
            disp = act
        elif args.mode == 'delta':
            disp = act - act.mean(axis=0, keepdims=True)
        elif args.mode == 'zscore':
            mean_k = act.mean(axis=0, keepdims=True)
            std_k = act.std(axis=0, keepdims=True) + 1e-6
            disp = (act - mean_k) / std_k
        elif args.mode == 'topn':
            # for each stage, mark top-N joints by absolute (centered) act
            # disp[k, v] in {0, 1}; produces N red joints per panel
            ranked = act - act.mean(axis=0, keepdims=True)
            disp = np.zeros_like(act)
            for k in range(act.shape[0]):
                idx = np.argsort(-ranked[k])[:args.top_n]
                disp[k, idx] = 1.0
        else:  # argmax
            owner = act.argmax(axis=0)  # (V,) which stage owns each joint
            disp = np.zeros_like(act)
            for k in range(act.shape[0]):
                disp[k, owner == k] = 1.0

        if args.norm_mode == 'per-row':
            if args.mode in ('delta', 'zscore'):
                vmax = float(np.abs(disp).max() + 1e-6)
                row_min, row_max = -vmax, vmax
            else:
                row_min, row_max = float(disp.min()), float(disp.max())

        for k in range(K):
            ax = fig.add_subplot(gs[ri, k], projection='3d')
            if args.mode == 'topn':
                draw_topn_skeleton(ax, coords, disp[k] > 0.5,
                                   tuple(args.view))
            else:
                if args.norm_mode == 'per-cell':
                    if args.mode in ('delta', 'zscore'):
                        vm = float(np.abs(disp[k]).max() + 1e-6)
                        vmin, vmax = -vm, vm
                    else:
                        vmin, vmax = float(disp[k].min()), float(disp[k].max())
                else:
                    vmin, vmax = row_min, row_max
                draw_heatmap_skeleton(ax, coords, disp[k], vmin, vmax,
                                      tuple(args.view), cmap)
            cls_name = NTU60_ACTIONS[row['cls']] if 0 <= row['cls'] < len(NTU60_ACTIONS) \
                else f"class {row['cls']}"
            if ri == 0:
                ax.set_title(f'stage {k + 1}', fontsize=10, pad=2)
            if k == 0:
                ax.text2D(-0.05, 0.5, cls_name,
                          transform=ax.transAxes,
                          rotation=90, va='center', ha='right', fontsize=10)

        ax_curve = fig.add_subplot(gs[ri, K])
        T_seq = attn.shape[1]
        cmap_k = plt.get_cmap('viridis', K)
        for k in range(K):
            ax_curve.plot(np.arange(T_seq), attn[k], color=cmap_k(k),
                          linewidth=1.5, label=f'k={k + 1}')
            tk = int(np.argmax(attn[k]))
            ax_curve.scatter([tk], [attn[k, tk]], color=cmap_k(k),
                             s=18, zorder=5, edgecolors='k', linewidths=0.4)
        ax_curve.set_xticks([0, T_seq // 2, T_seq - 1])
        ax_curve.set_yticks([])
        ax_curve.tick_params(labelsize=7)
        for spine in ('top', 'right'):
            ax_curve.spines[spine].set_visible(False)
        if ri == 0:
            ax_curve.set_title(r'$\alpha_k(t)$ (peak = $\bullet$)',
                               fontsize=10, pad=2)
        if ri == n_rows - 1:
            ax_curve.set_xlabel('frame index', fontsize=8)

    # global k-color legend at the very top
    legend_handles = [plt.Line2D([0], [0], color=plt.get_cmap('viridis', K)(k),
                                 linewidth=2, label=f'stage {k + 1}')
                      for k in range(K)]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.995), ncol=K, frameon=False,
               fontsize=9)

    title_suffix = {
        'weighted': r'(joint color = $\Vert\sum_t \alpha_k(t)\,\tilde f_{:,t,v}\Vert_2$)',
        'delta':    r'(joint color = stage activation $-$ row mean)',
        'zscore':   r'(joint color = z-score of stage activation across stages)',
        'argmax':   '(joint colored if it has its peak activation in this stage)',
        'topn':     f'(red = top-{args.top_n} joints with the largest stage-weighted feature norm)',
    }[args.mode]
    fig.suptitle(f'TS-HyperGCN per-stage joint focus  {title_suffix}',
                 fontsize=11, y=0.97)

    # colorbar in its OWN gridspec column → no overlap
    if args.norm_mode == 'per-row' and args.mode not in ('argmax', 'topn'):
        if args.mode in ('delta', 'zscore'):
            sm = mpl.cm.ScalarMappable(cmap=cmap,
                                       norm=mpl.colors.Normalize(vmin=-1, vmax=1))
            ticks = [-1, 0, 1]
            ticklabels = ['low', '0', 'high'] if args.mode == 'zscore' \
                else ['below', 'mean', 'above']
            label = ('z-score across stages' if args.mode == 'zscore'
                     else 'relative to row mean')
        else:
            sm = mpl.cm.ScalarMappable(cmap=cmap,
                                       norm=mpl.colors.Normalize(vmin=0, vmax=1))
            ticks = [0, 1]
            ticklabels = ['low', 'high']
            label = 'activation magnitude'
        sm.set_array([])
        # one colorbar across all rows, in the dedicated last column
        cax = fig.add_subplot(gs[:, K + 1])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', ticks=ticks)
        cbar.ax.set_yticklabels(ticklabels, fontsize=7)
        cbar.set_label(label, fontsize=8)

    out_dir = Path(ensure_outdir(Path(args.out).parent))
    base = Path(args.out).name
    _style.save_dual(fig, str(out_dir / base))
    print('[saved]', out_dir / f'{base}.pdf')


if __name__ == '__main__':
    main()
