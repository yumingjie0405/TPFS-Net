"""Measure topology drift from real Baseline and TPFS-Net checkpoints.

Both models receive the same deterministic prefix of the NTU60-XSub test set.
The nine ``gcn1.last_relation`` tensors are reduced to one physical-joint
relation per sample, then compared with the physical skeleton topology.
"""

import argparse
import csv
import inspect
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import torch

from tools._io import (
    ROOT,
    build_model,
    build_test_loader,
    ensure_outdir,
    load_config,
    load_weights,
    resolve_checkpoint,
)
from tools import vis_config as _vis_config


MODEL_NAMES = ('Baseline', 'TPFS-Net')
NUM_LAYERS = 9
NUM_NODES = 25
FORMULAS = {
    'drift': 'd = ||G - A||_F / ||A||_F',
    'leakage': 'e = ||(I - B B^T) G B||_F^2 / r',
    'edge_retention': (
        '|top-|E| directed off-diagonal entries of G intersect E| / |E|'
    ),
    'relation_processing': (
        'crop first 25 nodes; average persons and subsets; symmetrize; '
        'row-normalize'
    ),
    'basis': 'B = r lowest-eigenvalue eigenvectors of the normalized physical Laplacian',
}


def seed_everything(seed):
    """Make sample preprocessing and bootstrap resampling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def row_normalize(matrix, eps=1e-12):
    """Normalize each row by its L1 mass, preserving signed relations."""
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = np.abs(matrix).sum(axis=-1, keepdims=True)
    return np.divide(
        matrix, denominator, out=np.zeros_like(matrix),
        where=denominator > eps,
    )


def process_relations(relation, num_samples, num_nodes=NUM_NODES):
    """Convert captured (N*persons,K,V,V) tensors to (N,V,V)."""
    value = relation.detach().float().cpu().numpy()
    if value.ndim != 4 or value.shape[-2] < num_nodes or value.shape[-1] < num_nodes:
        raise ValueError(
            'last_relation must have shape (N, subset, V, V) with V >= 25; '
            f'got {value.shape}'
        )
    if value.shape[0] % num_samples:
        raise ValueError(
            f'cannot map relation batch {value.shape[0]} to {num_samples} samples'
        )
    persons = value.shape[0] // num_samples
    value = value[:, :, :num_nodes, :num_nodes]
    value = value.reshape(num_samples, persons, value.shape[1], num_nodes, num_nodes)
    value = value.mean(axis=(1, 2))
    value = 0.5 * (value + value.swapaxes(-1, -2))
    return row_normalize(value)


def physical_adjacency(neighbor, num_nodes=NUM_NODES):
    """Build, symmetrize, add self-loops, and row-normalize physical A."""
    adjacency = np.eye(num_nodes, dtype=np.float64)
    for source, target in neighbor:
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
    return row_normalize(0.5 * (adjacency + adjacency.T))


def low_frequency_basis(adjacency, rank):
    """Return the lowest-frequency basis of the normalized Laplacian."""
    if not 1 <= rank <= adjacency.shape[0]:
        raise ValueError(f'rank must be in [1, {adjacency.shape[0]}], got {rank}')
    symmetric = 0.5 * (adjacency + adjacency.T)
    degree = symmetric.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 1e-12
    inv_sqrt[positive] = degree[positive] ** -0.5
    laplacian = np.eye(len(degree)) - (
        inv_sqrt[:, None] * symmetric * inv_sqrt[None, :]
    )
    _, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvectors[:, :rank]


def physical_edges(neighbor, num_nodes=NUM_NODES):
    """Return directed physical off-diagonal entries of symmetric A."""
    edges = set()
    for source, target in neighbor:
        source, target = int(source), int(target)
        if source != target and 0 <= source < num_nodes and 0 <= target < num_nodes:
            edges.add((source, target))
            edges.add((target, source))
    return edges


def edge_retention(matrix, edges):
    """Fraction of physical edges among top-|E| directed off-diagonals."""
    num_nodes = matrix.shape[-1]
    mask_offdiag = ~np.eye(num_nodes, dtype=bool)
    row, column = np.nonzero(mask_offdiag)
    values = matrix[..., row, column]
    count = len(edges)
    if count == 0 or count > values.shape[-1]:
        raise ValueError(f'invalid physical edge count: {count}')
    order = np.argsort(-values, axis=-1, kind='stable')[..., :count]
    edge_mask = np.array(
        [(int(i), int(j)) in edges for i, j in zip(row, column)],
        dtype=np.float64,
    )
    return edge_mask[order].mean(axis=-1)


def compute_metrics(relations, adjacency, basis, edges):
    """Compute per-sample, per-layer drift, leakage, and edge retention."""
    relations = np.asarray(relations, dtype=np.float64)
    if relations.ndim != 4 or relations.shape[-2:] != adjacency.shape:
        raise ValueError(
            f'relations must be (N,L,V,V), got {relations.shape}'
        )
    difference = relations - adjacency[None, None]
    norm_a = np.linalg.norm(adjacency, ord='fro')
    if norm_a <= 0:
        raise ValueError('physical adjacency has zero Frobenius norm')
    drift = np.linalg.norm(difference, axis=(-2, -1)) / norm_a

    projector = np.eye(adjacency.shape[0]) - basis @ basis.T
    projected = np.einsum('uv,nlvw,wr->nlur', projector, relations, basis)
    leakage = np.square(projected).sum(axis=(-2, -1)) / basis.shape[1]
    retention = edge_retention(relations, edges)
    return {
        'drift': drift,
        'leakage': leakage,
        'retention': retention,
    }


def bootstrap_ci(values, rng, resamples=2000, confidence=0.95):
    """Percentile bootstrap confidence interval for the sample mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError('values must be a nonempty (samples, layers) array')
    indices = rng.integers(0, values.shape[0], size=(resamples, values.shape[0]))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return np.quantile(means, [alpha, 1.0 - alpha], axis=0)


def gcn_modules(model):
    """Get the nine ordered spatial GCN modules and enable capture."""
    modules = []
    for layer_index in range(1, NUM_LAYERS + 1):
        layer = getattr(model, f'l{layer_index}', None)
        gcn = getattr(layer, 'gcn1', None)
        if gcn is None or not hasattr(gcn, 'capture_relation'):
            raise AttributeError(f'model layer l{layer_index}.gcn1 cannot capture relations')
        gcn.capture_relation = True
        modules.append(gcn)
    return modules


def extract_batch(model, modules, data, sample_count):
    """Forward one real batch and return all processed captured relations."""
    with torch.inference_mode():
        model(data)
    layer_relations = []
    for layer_index, module in enumerate(modules, start=1):
        if module.last_relation is None:
            raise RuntimeError(f'layer {layer_index} did not capture last_relation')
        layer_relations.append(process_relations(module.last_relation, sample_count))
        module.last_relation = None
    return np.stack(layer_relations, axis=1)


def validate_same_test_set(ours_cfg, base_cfg):
    """Reject comparisons that do not use the same feeder and test samples."""
    if ours_cfg.get('feeder') != base_cfg.get('feeder'):
        raise ValueError('ours and baseline feeder classes differ')
    if ours_cfg.get('test_feeder_args') != base_cfg.get('test_feeder_args'):
        raise ValueError('ours and baseline test_feeder_args must match exactly')
    args = ours_cfg.get('test_feeder_args', {})
    if args.get('split') != 'test' or 'NTU60_CS' not in str(args.get('data_path', '')):
        raise ValueError('the shared loader must be the NTU60-XSub test split')


def collect_relations(model, loader, device, seed):
    """Collect relations and dataset indices in deterministic loader order."""
    seed_everything(seed)
    modules = gcn_modules(model)
    batches, sample_indices = [], []
    for batch in loader:
        data = batch[0].float().to(device, non_blocking=True)
        batches.append(extract_batch(model, modules, data, data.shape[0]))
        if len(batch) >= 3:
            sample_indices.extend(torch.as_tensor(batch[2]).cpu().tolist())
        else:
            start = len(sample_indices)
            sample_indices.extend(range(start, start + data.shape[0]))
    if not batches:
        raise RuntimeError('test loader produced no samples')
    return np.concatenate(batches, axis=0), np.asarray(sample_indices, dtype=np.int64)


def configure_style(font_path=None):
    """Call the shared style API while supporting its current older signature."""
    if font_path:
        font_file = Path(font_path).expanduser().resolve()
        if not font_file.is_file():
            raise FileNotFoundError(f'font file does not exist: {font_file}')
        font_manager.fontManager.addfont(str(font_file))
    setup = _vis_config.setup_journal_style
    if 'font_path' in inspect.signature(setup).parameters:
        setup(font_path=font_path)
    else:
        setup()
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    if 'Times New Roman' not in available:
        raise RuntimeError(
            'Times New Roman is required; install it or pass --font-path '
            'pointing to a Times New Roman font file'
        )
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['mathtext.fontset'] = 'stix'


def validate_layout(fig):
    """Use shared validation when available, otherwise enforce figure bounds."""
    validator = getattr(_vis_config, 'validate_layout', None)
    if validator is not None:
        result = validator(fig)
        if result is False:
            raise RuntimeError('vis_config.validate_layout rejected the figure')
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_window_extent(renderer)
    for artist in fig.findobj():
        if not artist.get_visible() or not hasattr(artist, 'get_window_extent'):
            continue
        try:
            extent = artist.get_window_extent(renderer)
        except (AttributeError, RuntimeError, ValueError):
            continue
        if extent.width <= 0 or extent.height <= 0:
            continue
        tolerance = 2.0
        if (extent.x0 < bounds.x0 - tolerance or extent.y0 < bounds.y0 - tolerance
                or extent.x1 > bounds.x1 + tolerance or extent.y1 > bounds.y1 + tolerance):
            raise RuntimeError(f'figure artist lies outside canvas: {artist!r}')


def save_outputs(fig, output_stem):
    """Use the shared figure writer, allowing APIs with an optional dpi."""
    saver = _vis_config.save_figure
    parameters = inspect.signature(saver).parameters
    if 'dpi' in parameters:
        saver(fig, str(output_stem), dpi=300)
    else:
        saver(fig, str(output_stem))


def summarize(metrics, bootstrap_resamples, seed):
    """Create means and paired bootstrap intervals for all three metrics."""
    summary = {}
    for metric_index, metric_name in enumerate(('drift', 'leakage', 'retention')):
        summary[metric_name] = {}
        for model_index, model_name in enumerate(MODEL_NAMES):
            values = metrics[model_name][metric_name]
            rng = np.random.default_rng(seed + 100 * metric_index + model_index)
            ci = bootstrap_ci(values, rng, resamples=bootstrap_resamples)
            summary[metric_name][model_name] = {
                'mean': values.mean(axis=0),
                'ci_low': ci[0],
                'ci_high': ci[1],
            }
    return summary


def save_csv(path, summary):
    """Write one row per model and layer with mean and 95% CI statistics."""
    fields = ['model', 'layer']
    for metric_name in ('drift', 'leakage', 'retention'):
        fields.extend((f'{metric_name}_mean', f'{metric_name}_ci_low',
                       f'{metric_name}_ci_high'))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in MODEL_NAMES:
            for layer_index in range(NUM_LAYERS):
                row = {'model': model_name, 'layer': layer_index + 1}
                for metric_name in ('drift', 'leakage', 'retention'):
                    stats = summary[metric_name][model_name]
                    row[f'{metric_name}_mean'] = float(stats['mean'][layer_index])
                    row[f'{metric_name}_ci_low'] = float(stats['ci_low'][layer_index])
                    row[f'{metric_name}_ci_high'] = float(stats['ci_high'][layer_index])
                writer.writerow(row)


def save_npz(path, metrics, relations, adjacency, basis, sample_indices):
    """Persist per-sample metrics and layer mean relations."""
    np.savez_compressed(
        path,
        sample_indices=sample_indices,
        physical_adjacency=adjacency,
        low_frequency_basis=basis,
        baseline_drift=metrics['Baseline']['drift'],
        baseline_leakage=metrics['Baseline']['leakage'],
        baseline_retention=metrics['Baseline']['retention'],
        tpfs_net_drift=metrics['TPFS-Net']['drift'],
        tpfs_net_leakage=metrics['TPFS-Net']['leakage'],
        tpfs_net_retention=metrics['TPFS-Net']['retention'],
        baseline_mean_matrices=relations['Baseline'].mean(axis=0),
        tpfs_net_mean_matrices=relations['TPFS-Net'].mean(axis=0),
    )


def plot_curve(ax, summary, metric_name, ylabel, title):
    """Plot both model means and bootstrap bands on one panel."""
    layers = np.arange(1, NUM_LAYERS + 1)
    styles = {
        'Baseline': ('#D55E00', 'o'),
        'TPFS-Net': ('#0072B2', 's'),
    }
    lines = []
    for model_name in MODEL_NAMES:
        color, marker = styles[model_name]
        stats = summary[metric_name][model_name]
        line, = ax.plot(layers, stats['mean'], color=color, marker=marker,
                        label=model_name, linewidth=1.8, markersize=4.5)
        ax.fill_between(layers, stats['ci_low'], stats['ci_high'],
                        color=color, alpha=0.18, linewidth=0)
        lines.append(line)
    ax.set_title(title)
    ax.set_xlabel('GCN layer')
    ax.set_ylabel(ylabel)
    ax.set_xticks(layers)
    ax.set_xlim(0.7, NUM_LAYERS + 0.3)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
    return lines


def make_figure(summary, relations, adjacency):
    """Create the required 2x2 journal figure with a common legend."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    lines = plot_curve(
        axes[0, 0], summary, 'drift', 'Normalized topology drift',
        '(a) Topology drift (mean and 95% CI)',
    )
    plot_curve(
        axes[0, 1], summary, 'leakage', 'Spectral leakage',
        '(b) Low-frequency leakage',
    )
    plot_curve(
        axes[1, 0], summary, 'retention', 'Physical-edge retention',
        '(c) Physical-edge retention',
    )

    panel = axes[1, 1]
    panel.set_title('(d) Final-layer mean absolute deviation')
    panel.set_axis_off()
    deviations = [
        np.abs(relations[name][:, -1].mean(axis=0) - adjacency)
        for name in MODEL_NAMES
    ]
    vmax = max(float(matrix.max()) for matrix in deviations)
    vmax = max(vmax, np.finfo(float).eps)
    heat_axes = [
        panel.inset_axes([0.00, 0.12, 0.42, 0.72]),
        panel.inset_axes([0.50, 0.12, 0.42, 0.72]),
    ]
    image = None
    for heat_ax, model_name, deviation in zip(heat_axes, MODEL_NAMES, deviations):
        image = heat_ax.imshow(deviation, cmap='magma', vmin=0.0, vmax=vmax,
                               interpolation='nearest', aspect='equal')
        heat_ax.set_title(model_name, fontsize=9, fontweight='normal', pad=3)
        heat_ax.set_xlabel('Joint')
        heat_ax.set_ylabel('Joint')
        heat_ax.set_xticks([0, 12, 24], labels=[1, 13, 25])
        heat_ax.set_yticks([0, 12, 24], labels=[1, 13, 25])
    color_ax = panel.inset_axes([0.15, 0.00, 0.62, 0.045])
    fig.colorbar(image, cax=color_ax, orientation='horizontal',
                 label='Absolute deviation')
    fig.legend(lines, MODEL_NAMES, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.90,
                        wspace=0.34, hspace=0.40)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure real checkpoint topology drift on NTU60-XSub.',
    )
    parser.add_argument('--ours-config', required=True)
    parser.add_argument('--ours-work-dir', required=True)
    parser.add_argument('--ours-ckpt', default=None,
                        help='Explicit TPFS-Net checkpoint (default: logged best)')
    parser.add_argument('--base-config', required=True)
    parser.add_argument('--base-work-dir', required=True)
    parser.add_argument('--base-ckpt', default=None,
                        help='Explicit Baseline checkpoint (default: logged best)')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--rank', type=int, default=4,
                        help='Low-frequency basis rank r')
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--font-path', default=None)
    parser.add_argument('--data-only', action='store_true',
                        help='Compute CSV/NPZ/JSON without rendering a figure')
    parser.add_argument('--out', default='vis/figs_paper/fig_topology_drift')
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error('--max-samples must be positive')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    if args.bootstrap_resamples <= 0:
        parser.error('--bootstrap-resamples must be positive')
    return args


def canonical_path(path):
    """Return an absolute path, interpreting relative paths from repo root."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def validate_graphs(ours_model, base_model):
    """Ensure both models expose the same NTU physical topology."""
    ours_neighbor = {tuple(edge) for edge in ours_model.graph.neighbor}
    base_neighbor = {tuple(edge) for edge in base_model.graph.neighbor}
    if ours_neighbor != base_neighbor:
        raise ValueError('ours and baseline graph.neighbor definitions differ')
    if getattr(ours_model, 'num_point', NUM_NODES) != NUM_NODES:
        raise ValueError('ours model is not configured for 25-node NTU data')
    if getattr(base_model, 'num_point', NUM_NODES) != NUM_NODES:
        raise ValueError('baseline model is not configured for 25-node NTU data')
    return list(ours_model.graph.neighbor)


def build_loaded_model(config, checkpoint, device):
    model = build_model(config, device=device)
    load_weights(model, checkpoint, device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    config_paths = {
        'Baseline': canonical_path(args.base_config),
        'TPFS-Net': canonical_path(args.ours_config),
    }
    work_dirs = {
        'Baseline': canonical_path(args.base_work_dir),
        'TPFS-Net': canonical_path(args.ours_work_dir),
    }
    configs = {name: load_config(path) for name, path in config_paths.items()}
    validate_same_test_set(configs['TPFS-Net'], configs['Baseline'])
    ours_explicit = str(canonical_path(args.ours_ckpt)) if args.ours_ckpt else None
    base_explicit = str(canonical_path(args.base_ckpt)) if args.base_ckpt else None
    checkpoints = {
        'Baseline': Path(resolve_checkpoint(
            work_dirs['Baseline'], policy='best', explicit=base_explicit,
        )).resolve(),
        'TPFS-Net': Path(resolve_checkpoint(
            work_dirs['TPFS-Net'], policy='best', explicit=ours_explicit,
        )).resolve(),
    }

    models = {
        name: build_loaded_model(configs[name], checkpoints[name], device)
        for name in MODEL_NAMES
    }
    neighbor = validate_graphs(models['TPFS-Net'], models['Baseline'])
    loader = build_test_loader(
        configs['TPFS-Net'], batch_size=args.batch_size,
        num_workers=args.num_workers, max_samples=args.max_samples,
    )
    relations = {}
    indices = {}
    for name in MODEL_NAMES:
        relations[name], indices[name] = collect_relations(
            models[name], loader, device, args.seed,
        )
    if not np.array_equal(indices['Baseline'], indices['TPFS-Net']):
        raise RuntimeError('models were not evaluated on identical test samples')

    adjacency = physical_adjacency(neighbor)
    basis = low_frequency_basis(adjacency, args.rank)
    edges = physical_edges(neighbor)
    metrics = {
        name: compute_metrics(relations[name], adjacency, basis, edges)
        for name in MODEL_NAMES
    }
    summary = summarize(metrics, args.bootstrap_resamples, args.seed)

    output_stem = canonical_path(args.out)
    ensure_outdir(output_stem.parent)
    save_csv(output_stem.with_suffix('.csv'), summary)
    save_npz(output_stem.with_suffix('.npz'), metrics, relations, adjacency,
             basis, indices['Baseline'])

    configure_style(args.font_path)
    figure = make_figure(summary, relations, adjacency)
    validate_layout(figure)
    save_outputs(figure, output_stem)
    plt.close(figure)

    metadata = {
        'models': list(MODEL_NAMES),
        'checkpoints': {name: str(path) for name, path in checkpoints.items()},
        'configs': {name: str(path) for name, path in config_paths.items()},
        'work_dirs': {name: str(path) for name, path in work_dirs.items()},
        'sample_count': int(len(indices['Baseline'])),
        'sample_indices': indices['Baseline'].tolist(),
        'seed': args.seed,
        'rank': args.rank,
        'bootstrap_resamples': args.bootstrap_resamples,
        'checkpoint_policy': 'explicit when supplied, otherwise best',
        'formulas': FORMULAS,
    }
    output_stem.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8',
    )
    print(f'[saved] {output_stem}.pdf/.png/.csv/.npz/.json')


if __name__ == '__main__':
    main()
