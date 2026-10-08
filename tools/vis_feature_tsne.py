"""Compare baseline and TPFS-Net final pooled features with reproducible t-SNE.

Both models receive the same batches from one test dataset. Features are captured
by forward pre-hooks on ``model.fc``. Silhouette and Davies-Bouldin are computed
on per-sample L2-normalized high-dimensional features, before PCA or t-SNE.
"""

import argparse
import csv
import json
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tools import _io, vis_config


SEEDS = (0, 1, 2)
FIGURE_COLORS = ('#2a78d6', '#008300', '#e87ba4', '#eda100')
FIGURE_MARKERS = ('o', 's', '^', 'D')
NORMALIZATION = (
    "Per-sample L2 normalization; high-dimensional metrics use Euclidean "
    "distance on these normalized final pooled features, before PCA/t-SNE."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours-config', required=True)
    parser.add_argument('--ours-work-dir', required=True)
    parser.add_argument('--ours-checkpoint', default=None,
                        help='Explicit checkpoint; otherwise logged best is used')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-work-dir', required=True)
    parser.add_argument('--baseline-checkpoint', default=None,
                        help='Explicit checkpoint; otherwise logged best is used')
    classes = parser.add_mutually_exclusive_group()
    classes.add_argument('--class-ids', nargs='+', type=int)
    classes.add_argument('--class-count', '--classes', dest='class_count',
                         type=int)
    parser.add_argument('--max-samples', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--learning-rate', default='auto')
    parser.add_argument('--device', default=None)
    parser.add_argument('--font-path', default=None,
                        help='Times New Roman file; or set TPFS_FONT_PATH')
    parser.add_argument('--out', default='vis/figs_paper/fig_feature_tsne')
    return parser.parse_args(argv)


def require_identical_feeder(ours_cfg, baseline_cfg):
    if ours_cfg.get('feeder') != baseline_cfg.get('feeder'):
        raise ValueError('ours and baseline must use the exact same feeder')
    ours_args = ours_cfg.get('test_feeder_args')
    baseline_args = baseline_cfg.get('test_feeder_args')
    if ours_args != baseline_args:
        raise ValueError(
            'ours and baseline test_feeder_args must be exactly identical:\n'
            f'ours={ours_args!r}\nbaseline={baseline_args!r}'
        )


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_shared_dataset(cfg):
    feeder = _io.import_class(cfg['feeder'])
    return feeder(**dict(cfg['test_feeder_args']))


def _dataset_labels(dataset):
    if not hasattr(dataset, 'label'):
        raise TypeError('test feeder must expose a label sequence for selection')
    labels = np.asarray(dataset.label, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(dataset):
        raise ValueError('feeder label sequence must be one-dimensional and complete')
    return labels


def _balanced_class_indices(labels, selected_classes, max_samples):
    quota, remainder = divmod(max_samples, len(selected_classes))
    chosen = []
    for position, class_id in enumerate(selected_classes):
        class_indices = np.flatnonzero(labels == class_id)
        take = min(len(class_indices), quota + (position < remainder))
        chosen.extend(class_indices[:take].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def select_indices(dataset, class_ids, class_count, max_samples):
    if max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    labels = _dataset_labels(dataset)
    available = np.unique(labels)
    if class_ids is not None:
        selected_classes = np.asarray(class_ids, dtype=np.int64)
        if len(selected_classes) == 0 or len(np.unique(selected_classes)) != len(selected_classes):
            raise ValueError('--class-ids must be non-empty and unique')
    else:
        count = class_count if class_count is not None else min(12, len(available))
        if count < 2:
            raise ValueError('--class-count must be at least 2')
        selected_classes = available[:count]
    missing = np.setdiff1d(selected_classes, available)
    if len(missing):
        raise ValueError(f'requested class IDs are absent: {missing.tolist()}')
    indices = _balanced_class_indices(labels, selected_classes, max_samples)
    chosen_labels = labels[indices]
    if len(indices) < 3 or len(np.unique(chosen_labels)) < 2:
        raise ValueError('selection needs at least three samples from two classes')
    counts = np.unique(chosen_labels, return_counts=True)[1]
    if np.any(counts < 2):
        raise ValueError('each selected class must contribute at least two samples')
    return indices.astype(np.int64), chosen_labels, selected_classes


def _as_index_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def _as_label_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def collect_paired_features(ours, baseline, loader, device):
    captured = {'ours': None, 'baseline': None}

    def make_hook(name):
        def hook(_module, inputs):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f'{name} fc pre-hook did not receive one tensor')
            feature = inputs[0]
            if feature.ndim != 2:
                raise RuntimeError(
                    f'{name} final pooled feature must be 2-D, got {feature.shape}'
                )
            captured[name] = feature.detach().cpu().numpy().copy()
        return hook

    if not hasattr(ours, 'fc') or not hasattr(baseline, 'fc'):
        raise AttributeError('both models must expose the final classifier as .fc')
    handles = [ours.fc.register_forward_pre_hook(make_hook('ours')),
               baseline.fc.register_forward_pre_hook(make_hook('baseline'))]
    features = {'ours': [], 'baseline': []}
    labels_seen, indices_seen = [], []
    try:
        with torch.inference_mode():
            for data, labels, indices in tqdm(loader, desc='final pooled features',
                                               ncols=88):
                data = data.float().to(device)
                batch_labels = _as_label_array(labels)
                batch_indices = _as_index_array(indices)
                captured['ours'] = captured['baseline'] = None
                ours(data)
                ours_feature = captured['ours']
                baseline(data)
                baseline_feature = captured['baseline']
                if ours_feature is None or baseline_feature is None:
                    raise RuntimeError('an fc pre-hook was not invoked by model forward')
                expected = len(batch_labels)
                if ours_feature.shape[0] != expected or baseline_feature.shape[0] != expected:
                    raise RuntimeError('captured feature batch size does not match labels')
                features['ours'].append(ours_feature)
                features['baseline'].append(baseline_feature)
                labels_seen.append(batch_labels)
                indices_seen.append(batch_indices)
    finally:
        for handle in handles:
            handle.remove()
    return (np.concatenate(features['ours']),
            np.concatenate(features['baseline']),
            np.concatenate(labels_seen), np.concatenate(indices_seen))


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError('cannot L2-normalize a zero final pooled feature')
    normalized = features / norms
    if not np.isfinite(normalized).all():
        raise ValueError('features contain non-finite values')
    return normalized


def high_dim_metrics(features, labels):
    return {
        'silhouette': float(silhouette_score(features, labels,
                                             metric='euclidean')),
        'davies_bouldin': float(davies_bouldin_score(features, labels)),
    }


def reduce_and_embed(features, perplexity, iterations, learning_rate):
    n_samples, feature_dim = features.shape
    if not 0 < perplexity < n_samples:
        raise ValueError(
            f'perplexity must be in (0, n_samples={n_samples}), got {perplexity}'
        )
    pca_dim = min(50, n_samples - 1, feature_dim)
    reduced = PCA(n_components=pca_dim, random_state=0).fit_transform(features)
    embeddings = []
    parameters = {
        'n_components': 2, 'perplexity': float(perplexity),
        'early_exaggeration': 12.0, 'learning_rate': learning_rate,
        'n_iter': int(iterations), 'init': 'pca', 'metric': 'euclidean',
        'method': 'barnes_hut', 'angle': 0.5,
    }
    for seed in SEEDS:
        embeddings.append(TSNE(random_state=seed, **parameters).fit_transform(reduced))
    return reduced, np.stack(embeddings), pca_dim, parameters


def embedding_stability(embeddings):
    """Spearman correlations of pairwise distances between t-SNE seeds."""
    distances = [pdist(embedding, metric='euclidean') for embedding in embeddings]
    pairs = {}
    values = []
    for left, right in combinations(range(len(SEEDS)), 2):
        result = spearmanr(distances[left], distances[right])
        correlation = getattr(result, 'statistic', None)
        if correlation is None:
            correlation = result.correlation
        correlation = float(correlation)
        if not np.isfinite(correlation):
            raise ValueError('undefined t-SNE stability correlation')
        key = f'seed_{SEEDS[left]}_vs_{SEEDS[right]}'
        pairs[key] = correlation
        values.append(correlation)
    pairs['mean'] = float(np.mean(values))
    pairs['minimum'] = float(np.min(values))
    return pairs


def verify_selection(expected_indices, expected_labels, indices, labels):
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError('loader indices differ from the selected dataset indices')
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError('loader labels differ from labels at the selected indices')


def _jsonable_tsne(parameters):
    return {key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in parameters.items()}


def save_exports(prefix, labels, indices, class_ids, normalized, reduced,
                 embeddings, metrics, stability, metadata):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix('.npz')
    arrays = {
        'labels': labels, 'indices': indices, 'class_ids': class_ids,
        'seeds': np.asarray(SEEDS, dtype=np.int64),
        'baseline_features_l2': normalized['baseline'],
        'ours_features_l2': normalized['ours'],
        'baseline_pca': reduced['baseline'], 'ours_pca': reduced['ours'],
        'baseline_embeddings': embeddings['baseline'],
        'ours_embeddings': embeddings['ours'],
    }
    for model_name in ('baseline', 'ours'):
        for metric_name, value in metrics[model_name].items():
            arrays[f'{model_name}_high_dim_{metric_name}'] = np.asarray(value)
        for measure, value in stability[model_name].items():
            arrays[f'{model_name}_stability_{measure}'] = np.asarray(value)
    np.savez_compressed(npz_path, **arrays)

    embedding_csv = prefix.parent / f'{prefix.name}_embeddings.csv'
    fields = ['dataset_index', 'label']
    for model_name in ('baseline', 'ours'):
        for seed in SEEDS:
            fields.extend([f'{model_name}_seed{seed}_x',
                           f'{model_name}_seed{seed}_y'])
    with embedding_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, (dataset_index, label) in enumerate(zip(indices, labels)):
            row = {'dataset_index': int(dataset_index), 'label': int(label)}
            for model_name in ('baseline', 'ours'):
                for seed_index, seed in enumerate(SEEDS):
                    point = embeddings[model_name][seed_index, row_index]
                    row[f'{model_name}_seed{seed}_x'] = float(point[0])
                    row[f'{model_name}_seed{seed}_y'] = float(point[1])
            writer.writerow(row)

    metrics_csv = prefix.parent / f'{prefix.name}_metrics.csv'
    with metrics_csv.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'space', 'measure', 'value'])
        for model_name in ('baseline', 'ours'):
            for measure, value in metrics[model_name].items():
                writer.writerow([model_name, 'l2_normalized_high_dim', measure, value])
            for measure, value in stability[model_name].items():
                writer.writerow([model_name, 'tsne_seed_stability', measure, value])

    meta_path = prefix.parent / f'{prefix.name}_meta.json'
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return npz_path, embedding_csv, metrics_csv, meta_path


def make_figure(prefix, baseline_embedding, ours_embedding, labels, class_ids):
    if len(class_ids) > len(FIGURE_COLORS):
        raise ValueError(
            f'the t-SNE figure supports at most {len(FIGURE_COLORS)} sampled '
            'classes so all scatter colors remain distinguishable')
    class_to_code = {int(class_id): code
                     for code, class_id in enumerate(class_ids)}
    codes = np.asarray([class_to_code[int(label)] for label in labels])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=False, sharey=False)
    for ax, embedding, title in zip(
            axes, (baseline_embedding, ours_embedding),
            ('(a) Hyper-GCN', '(b) TPFS-Net')):
        for code, (color, marker) in enumerate(zip(FIGURE_COLORS,
                                                   FIGURE_MARKERS)):
            mask = codes == code
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       color=color, marker=marker, s=14, alpha=0.76,
                       linewidths=0.25, edgecolors='white', rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#c3c2b7')
            spine.set_linewidth(0.7)
    handles = [Line2D([], [], linestyle='none', marker=FIGURE_MARKERS[code],
                      markerfacecolor=FIGURE_COLORS[code],
                      markeredgecolor='white', markeredgewidth=0.4,
                      markersize=5.5, label=f'Class {int(class_id)}')
               for code, class_id in enumerate(class_ids)]
    fig.legend(handles=handles, loc='upper center', ncol=len(class_ids),
               bbox_to_anchor=(0.5, 0.995), frameon=False,
               handletextpad=0.4, columnspacing=1.25)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.82,
                        wspace=0.18)
    vis_config.validate_layout(fig)
    vis_config.save_figure(fig, str(prefix))
    plt.close(fig)


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _parse_learning_rate(value):
    if value == 'auto':
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError('--learning-rate must be "auto" or a positive number') from exc
    if parsed <= 0:
        raise ValueError('--learning-rate must be positive')
    return parsed


def main(argv=None):
    args = parse_args(argv)
    learning_rate = _parse_learning_rate(args.learning_rate)
    if args.iterations <= 0:
        raise ValueError('--iterations must be positive')
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('batch size must be positive and workers non-negative')
    seed_everything(0)
    try:
        vis_config.setup_journal_style(args.font_path)
    except RuntimeError as exc:
        if (args.font_path is not None or
                'Times New Roman is required' not in str(exc)):
            raise
        from matplotlib import rcParams
        rcParams.update({
            'font.family': 'serif', 'font.serif': ['Nimbus Roman'],
            'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
            'axes.titleweight': 'bold', 'xtick.labelsize': 9,
            'ytick.labelsize': 9, 'legend.fontsize': 9,
            'pdf.fonttype': 42, 'ps.fonttype': 42,
        })
        print('[warning] Times New Roman unavailable; using Nimbus Roman')
    device = _resolve_device(args.device)

    ours_config_path = Path(args.ours_config).expanduser().resolve()
    baseline_config_path = Path(args.baseline_config).expanduser().resolve()
    ours_work_dir = Path(args.ours_work_dir).expanduser().resolve()
    baseline_work_dir = Path(args.baseline_work_dir).expanduser().resolve()
    ours_cfg = _io.load_config(ours_config_path)
    baseline_cfg = _io.load_config(baseline_config_path)
    require_identical_feeder(ours_cfg, baseline_cfg)

    ours_checkpoint = _io.resolve_checkpoint(
        ours_work_dir, policy='best', explicit=args.ours_checkpoint)
    baseline_checkpoint = _io.resolve_checkpoint(
        baseline_work_dir, policy='best', explicit=args.baseline_checkpoint)
    ours_checkpoint = str(Path(ours_checkpoint).expanduser().resolve())
    baseline_checkpoint = str(Path(baseline_checkpoint).expanduser().resolve())

    dataset = build_shared_dataset(ours_cfg)
    selected_indices, expected_labels, class_ids = select_indices(
        dataset, args.class_ids, args.class_count, args.max_samples)
    loader = DataLoader(Subset(dataset, selected_indices.tolist()),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    ours = _io.build_model(ours_cfg, device)
    baseline = _io.build_model(baseline_cfg, device)
    _io.load_weights(ours, ours_checkpoint, device)
    _io.load_weights(baseline, baseline_checkpoint, device)
    ours_features, baseline_features, labels, indices = collect_paired_features(
        ours, baseline, loader, device)
    verify_selection(selected_indices, expected_labels, indices, labels)

    normalized = {'ours': l2_normalize(ours_features),
                  'baseline': l2_normalize(baseline_features)}
    metrics = {name: high_dim_metrics(features, labels)
               for name, features in normalized.items()}
    reduced, embeddings, stability, pca_dims, tsne_parameters = {}, {}, {}, {}, {}

    for name in ('baseline', 'ours'):
        reduced[name], embeddings[name], pca_dims[name], params = \
            reduce_and_embed(normalized[name], args.perplexity,
                             args.iterations, learning_rate)
        tsne_parameters[name] = _jsonable_tsne(params)
        stability[name] = embedding_stability(embeddings[name])

    prefix = Path(args.out).expanduser().resolve()
    metadata = {
        'models': {
            'baseline': {'config': str(baseline_config_path),
                         'work_dir': str(baseline_work_dir),
                         'checkpoint': baseline_checkpoint,
                         'feature_dimension': int(baseline_features.shape[1])},
            'ours': {'display_name': 'TPFS-Net', 'config': str(ours_config_path),
                     'work_dir': str(ours_work_dir),
                     'checkpoint': ours_checkpoint,
                     'feature_dimension': int(ours_features.shape[1])},
        },
        'data': {
            'feeder': ours_cfg['feeder'],
            'test_feeder_args': ours_cfg['test_feeder_args'],
            'selection': 'balanced first dataset samples per requested class ID',
            'class_ids': class_ids.tolist(),
            'max_samples': int(args.max_samples),
            'sample_count': int(len(labels)),
            'indices_verified_identical': True,
            'labels_verified_identical': True,
            'same_realized_input_batches_for_both_models': True,
        },
        'features': {
            'capture': 'forward pre-hook on model.fc input',
            'description': 'final pooled high-dimensional feature',
            'normalization': NORMALIZATION,
            'metrics_space': 'L2-normalized high-dimensional feature space',
            'metrics': metrics,
        },
        'dimensionality_reduction': {
            'order': 'L2 normalization, PCA, then independent t-SNE per seed',
            'pca_components': pca_dims,
            'tsne_seeds': list(SEEDS),
            'tsne_parameters': tsne_parameters,
            'stability': {
                'measure': 'Spearman correlation of all pairwise Euclidean distances between t-SNE seeds',
                'values': stability,
            },
        },
        'software': {'numpy': np.__version__, 'torch': torch.__version__,
                     'scikit_learn': sklearn.__version__},
    }
    exported = save_exports(prefix, labels, indices, class_ids, normalized,
                            reduced, embeddings, metrics, stability, metadata)
    make_figure(prefix, embeddings['baseline'][0], embeddings['ours'][0],
                labels, class_ids)
    for name in ('baseline', 'ours'):
        print(f"{name}: high-D silhouette={metrics[name]['silhouette']:.4f}, "
              f"high-D Davies-Bouldin={metrics[name]['davies_bouldin']:.4f}, "
              f"t-SNE stability={stability[name]['mean']:.4f}")
    print('[saved exports] ' + ', '.join(str(path) for path in exported))


if __name__ == '__main__':
    main()
