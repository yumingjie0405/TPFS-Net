"""Visualize every learned stage prototype from a real checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

from tools._io import ensure_outdir, resolve_checkpoint
from tools.vis_config import (COLORS_STAGES, save_figure,
                              setup_journal_style)


def load_checkpoint_prototypes(path):
    """Load the one prototype tensor from a checkpoint; never synthesize it."""
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # older PyTorch
        payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict):
        raise TypeError('checkpoint must contain a state dictionary')
    for wrapper in ('state_dict', 'model_state_dict', 'model'):
        if wrapper in payload and isinstance(payload[wrapper], dict):
            payload = payload[wrapper]
            break
    matches = [(key, value) for key, value in payload.items()
               if key.split('module.')[-1].endswith('stage_proto_head.prototypes')]
    if len(matches) != 1:
        raise RuntimeError(
            'expected exactly one stage_proto_head.prototypes tensor, '
            f'found {[key for key, _ in matches]}'
        )
    key, tensor = matches[0]
    array = tensor.detach().cpu().float().numpy()
    if array.ndim != 3 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError(f'invalid prototype tensor shape/content: {array.shape}')
    return key, array


def normalize_rows(values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError('checkpoint contains a zero-norm prototype')
    return flat / norms


def compute_metrics(normalized, class_ids, stage_ids):
    """Compute metrics in normalized prototype space, not in the embedding."""
    result = {
        'space': 'L2-normalized original checkpoint prototype space',
        'class_silhouette_cosine': float(silhouette_score(
            normalized, class_ids, metric='cosine')),
        'stage_silhouette_cosine': float(silhouette_score(
            normalized, stage_ids, metric='cosine')),
        'class_davies_bouldin_euclidean': float(davies_bouldin_score(
            normalized, class_ids)),
        'stage_davies_bouldin_euclidean': float(davies_bouldin_score(
            normalized, stage_ids)),
    }
    similarity = normalized @ normalized.T
    same_class = class_ids[:, None] == class_ids[None, :]
    same_stage = stage_ids[:, None] == stage_ids[None, :]
    diagonal = np.eye(len(normalized), dtype=bool)
    result['mean_cosine_same_class'] = float(similarity[same_class & ~diagonal].mean())
    result['mean_cosine_different_class'] = float(similarity[~same_class].mean())
    result['mean_cosine_same_stage'] = float(similarity[same_stage & ~diagonal].mean())
    return result


def embed_prototypes(normalized, seed, perplexity):
    if not 1 <= perplexity < len(normalized):
        raise ValueError(f'perplexity must be in [1, {len(normalized) - 1}]')
    return TSNE(n_components=2, perplexity=perplexity, init='pca',
                learning_rate='auto', random_state=seed).fit_transform(normalized)


def write_data(stem, raw, normalized, embedding, class_ids, stage_ids):
    np.savez_compressed(f'{stem}.npz', prototypes=raw, normalized=normalized,
                        embedding=embedding, class_id=class_ids,
                        stage_id=stage_ids)
    with open(f'{stem}.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class_id', 'stage_id', 'embedding_x', 'embedding_y'] +
                        [f'prototype_{i}' for i in range(raw.shape[-1])])
        for i, (class_id, stage_id) in enumerate(zip(class_ids, stage_ids)):
            writer.writerow([int(class_id), int(stage_id), *embedding[i],
                             *raw.reshape(-1, raw.shape[-1])[i]])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='explicit learned checkpoint')
    parser.add_argument('--work-dir', help='directory used to resolve best checkpoint')
    parser.add_argument('--checkpoint-policy', choices=['best', 'latest'], default='best')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--font-path', help='Times New Roman font file; or TPFS_FONT_PATH')
    parser.add_argument('--out', default='vis/figs_paper/fig_proto_tsne')
    args = parser.parse_args(argv)
    if not args.ckpt and not args.work_dir:
        parser.error('one of --ckpt or --work-dir is required')
    return args


def main(argv=None):
    args = parse_args(argv)
    font_path = setup_journal_style(args.font_path)
    checkpoint = resolve_checkpoint(args.work_dir or '.', args.checkpoint_policy,
                                    explicit=args.ckpt)
    key, raw = load_checkpoint_prototypes(checkpoint)
    classes, stages, dimension = raw.shape
    normalized = normalize_rows(raw)
    class_ids = np.repeat(np.arange(classes), stages)
    stage_ids = np.tile(np.arange(stages), classes)
    metrics = compute_metrics(normalized, class_ids, stage_ids)
    embedding = embed_prototypes(normalized, args.seed, args.perplexity)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=class_ids,
                              cmap='turbo', s=28, alpha=.9, edgecolors='none')
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=.02)
    colorbar.set_label(f'Class ID (all {classes} classes)')
    for stage in range(stages):
        mask = stage_ids == stage
        axes[1].scatter(*embedding[mask].T, color=COLORS_STAGES[stage % len(COLORS_STAGES)],
                        s=28, alpha=.9, edgecolors='none')
    for ax, title in zip(axes, ['(a) Colored by class', '(b) Colored by temporal stage']):
        ax.set(title=title, xlabel='t-SNE dimension 1', ylabel='t-SNE dimension 2')
        ax.grid(alpha=.2, linestyle='--')
    handles = [Line2D([], [], marker='o', linestyle='', color=COLORS_STAGES[i % 4],
                      label=f'Stage {i + 1}') for i in range(stages)]
    fig.legend(handles=handles, loc='lower center', ncol=min(stages, 8), frameon=True)
    fig.suptitle('Learned checkpoint prototypes: one shared deterministic embedding')
    fig.tight_layout(rect=(0, .11, 1, .94))

    stem = str(Path(args.out).expanduser())
    ensure_outdir(str(Path(stem).parent))
    write_data(stem, raw, normalized, embedding, class_ids, stage_ids)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    metadata = {'analysis': 'learned checkpoint stage prototypes',
                'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': digest,
                'prototype_key': key, 'shape': [classes, stages, dimension],
                'all_classes_included': True, 'seed': args.seed,
                'tsne': {'perplexity': args.perplexity, 'init': 'pca'},
                'metrics': metrics, 'font_path': font_path}
    Path(f'{stem}.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    save_figure(fig, stem)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
