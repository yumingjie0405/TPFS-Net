"""Visualize and export stage attention from a trained best checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tools._io import (build_model, build_test_loader, ensure_outdir,
                       load_config, load_weights, resolve_checkpoint)
from tools.vis_config import (COLORS_STAGES, save_figure,
                              setup_journal_style, validate_layout)


def parse_pairs(values):
    pairs = []
    for value in values or []:
        parts = value.replace(',', ':').split(':')
        if len(parts) != 2:
            raise ValueError(f'invalid direction pair {value!r}; use CLASS:CLASS')
        pair = tuple(map(int, parts))
        if pair[0] == pair[1]:
            raise ValueError(f'direction pair must contain two classes: {value!r}')
        pairs.append(pair)
    return pairs


def dataset_labels(dataset):
    for name in ('label', 'labels', 'targets'):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def select_indices(dataset, classes, limit):
    labels = dataset_labels(dataset)
    if limit < len(classes):
        raise ValueError('num-samples must be at least the number of classes')
    selected = []
    quota, remainder = divmod(limit, len(classes))
    for position, cls in enumerate(classes):
        class_indices = np.flatnonzero(labels == cls)
        take = quota + (1 if position < remainder else 0)
        if len(class_indices) < take:
            raise RuntimeError(
                f'class {cls} has {len(class_indices)} samples, needs {take}')
        selected.extend(class_indices[:take].tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected, labels


def subset_loader(dataset, indices, batch_size, workers):
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=workers, drop_last=False)

def collect_attention(model, loader, device, limit):
    captured = []

    def hook(_module, _inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError('stage_decomposer did not return (features, attention)')
        captured.append(output[1].detach().cpu())

    handle = model.stage_decomposer.register_forward_hook(hook)
    attentions, labels, indices = [], [], []
    try:
        for data, label, index in tqdm(loader, ncols=80, desc='stage attention'):
            captured.clear()
            with torch.inference_mode():
                model(data.float().to(device))
            if len(captured) != 1:
                raise RuntimeError(f'expected one stage capture, got {len(captured)}')
            attentions.append(captured[0].numpy())
            labels.append(np.asarray(label, dtype=np.int64))
            indices.append(np.asarray(index, dtype=np.int64))
    finally:
        handle.remove()
    attention = np.concatenate(attentions, axis=0)[:limit]
    return (attention, np.concatenate(labels)[:limit],
            np.concatenate(indices)[:limit])


def attention_statistics(attention):
    weights = attention / np.maximum(attention.sum(axis=-1, keepdims=True), 1e-12)
    time = np.arange(attention.shape[-1], dtype=np.float64)
    centers = np.sum(weights * time, axis=-1)
    variances = np.sum(weights * (time[None, None] - centers[..., None]) ** 2,
                       axis=-1)
    widths = np.sqrt(np.maximum(variances, 0.0))
    overlap = np.minimum(weights[:, :, None, :],
                         weights[:, None, :, :]).sum(axis=-1)
    return weights, centers, widths, overlap


def sample_names(dataset, indices):
    names = getattr(dataset, 'sample_name', None)
    if names is None:
        return np.asarray([str(i) for i in indices])
    return np.asarray([str(names[int(i)]) for i in indices])


def export_data(base, attention, labels, indices, names, centers, widths,
                overlaps, checkpoint, config, classes, pairs):
    np.savez_compressed(
        f'{base}.npz', attention=attention, labels=labels,
        dataset_indices=indices, sample_names=names, centers=centers,
        widths_std=widths, pairwise_overlaps=overlaps,
        selected_classes=np.asarray(classes, dtype=np.int64),
        direction_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    with open(f'{base}.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset_index', 'sample_name', 'label', 'stage',
                         'expected_center_frame', 'width_std_frames'] +
                        [f'overlap_stage_{k + 1}'
                         for k in range(attention.shape[1])])
        for n, (label, index, name) in enumerate(zip(labels, indices, names)):
            for k in range(attention.shape[1]):
                writer.writerow([int(index), name, int(label), k + 1,
                                 float(centers[n, k]), float(widths[n, k]),
                                 *map(float, overlaps[n, k])])
    metadata = {
        'config': str(Path(config).resolve()),
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_policy': 'explicit' if checkpoint else 'best',
        'sample_selection': 'first matching test-dataset indices; no shuffle',
        'dataset_indices': indices.tolist(), 'labels': labels.tolist(),
        'sample_names': names.tolist(), 'selected_classes': list(classes),
        'direction_pairs': [list(pair) for pair in pairs],
        'center_units': 'output attention frame index',
        'width_definition': 'attention-weighted temporal standard deviation',
        'overlap_definition': 'sum_t min(normalized_attention_i, normalized_attention_j)',
        'model_parameters_modified': False,
    }
    Path(f'{base}_meta.json').write_text(json.dumps(metadata, indent=2))


def class_summary(values, labels, classes):
    means, lows, highs = [], [], []
    for cls in classes:
        group = values[labels == cls]
        if not len(group):
            raise RuntimeError(f'no selected samples for class {cls}')
        means.append(group.mean(axis=0))
        if len(group) == 1:
            lows.append(group[0]); highs.append(group[0])
        else:
            sem = group.std(axis=0, ddof=1) / np.sqrt(len(group))
            lows.append(group.mean(axis=0) - 1.96 * sem)
            highs.append(group.mean(axis=0) + 1.96 * sem)
    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def plot_figure(attention, labels, centers, overlaps, classes):
    stages, frames = attention.shape[1:]
    colors = [COLORS_STAGES[k % len(COLORS_STAGES)] for k in range(stages)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    mean_attention = attention.mean(axis=0)
    image = axes[0, 0].imshow(mean_attention, aspect='auto', cmap='viridis',
                              extent=[0, frames - 1, stages + .5, .5])
    axes[0, 0].set(yticks=np.arange(1, stages + 1), xlabel='Frame',
                   ylabel='Stage', title='(a) Mean stage-attention heatmap')
    fig.colorbar(image, ax=axes[0, 0], fraction=.05, pad=.03,
                 label='Attention')

    time = np.arange(frames)
    handles = []
    for k in range(stages):
        line, = axes[0, 1].plot(time, mean_attention[k], color=colors[k],
                                label=f'Stage {k + 1}')
        handles.append(line)
    axes[0, 1].set(xlabel='Frame', ylabel='Mean attention',
                   title='(b) Temporal attention profiles', xlim=(0, frames - 1))

    means, lows, highs = class_summary(centers, labels, classes)
    x = np.arange(len(classes), dtype=float)
    offsets = np.linspace(-.24, .24, stages)
    for k in range(stages):
        axes[1, 0].errorbar(x + offsets[k], means[:, k],
                            yerr=np.vstack((means[:, k] - lows[:, k],
                                            highs[:, k] - means[:, k])),
                            fmt='o', color=colors[k], capsize=2.5, markersize=3.5)
    axes[1, 0].set(xticks=x, xticklabels=[str(c) for c in classes],
                   xlabel='Class', ylabel='Expected center (frame)',
                   title='(c) Stage centers with 95% CI', ylim=(0, frames - 1))

    mean_overlap = overlaps.mean(axis=0)
    image2 = axes[1, 1].imshow(mean_overlap, vmin=0, vmax=1, cmap='magma')
    axes[1, 1].set(xticks=np.arange(stages), yticks=np.arange(stages),
                   xticklabels=np.arange(1, stages + 1),
                   yticklabels=np.arange(1, stages + 1), xlabel='Stage',
                   ylabel='Stage', title='(d) Mean pairwise overlap')
    fig.colorbar(image2, ax=axes[1, 1], fraction=.05, pad=.03,
                 label='Overlap')
    fig.legend(handles=handles, loc='upper center', ncol=stages,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.09, top=0.88,
                        wspace=0.34, hspace=0.40)
    validate_layout(fig)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--ckpt')
    parser.add_argument('--checkpoint-policy', choices=('best', 'latest'),
                        default='best')
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=6)
    parser.add_argument('--classes', type=int, nargs='+')
    parser.add_argument('--direction-pair', action='append', default=[],
                        metavar='CLASS:CLASS')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--font-path', required=True)
    parser.add_argument('--out', default='vis/figs_paper/fig_stage_attention')
    args = parser.parse_args()

    setup_journal_style(args.font_path)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = parse_pairs(args.direction_pair)
    classes = (args.classes if args.classes is not None else
               list(dict.fromkeys(c for pair in pairs for c in pair)) if pairs
               else list(range(args.num_classes)))
    if len(classes) > 12:
        raise ValueError('at most 12 classes may be plotted without label clutter')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    checkpoint = resolve_checkpoint(args.work_dir, args.checkpoint_policy,
                                    explicit=args.ckpt)
    model = build_model(cfg, device)
    load_weights(model, checkpoint, device)
    model.eval()

    source_loader = build_test_loader(cfg, batch_size=args.batch_size,
                                      num_workers=0)
    dataset = source_loader.dataset
    selected, _ = select_indices(dataset, classes, args.num_samples)
    loader = subset_loader(dataset, selected, args.batch_size, args.num_workers)
    attention, labels, indices = collect_attention(
        model, loader, device, len(selected))
    names = sample_names(dataset, indices)
    attention, centers, widths, overlaps = attention_statistics(attention)

    out = Path(args.out)
    ensure_outdir(out.parent)
    export_data(out, attention, labels, indices, names, centers, widths,
                overlaps, checkpoint, args.config, classes, pairs)
    fig = plot_figure(attention, labels, centers, overlaps, classes)
    save_figure(fig, str(out))
    plt.close(fig)


if __name__ == '__main__':
    main()
