"""NTU skeleton heat-kernel and graph-statistic method analysis."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata

from tools._io import ensure_outdir
from tools.vis_config import save_figure, setup_journal_style

# Canonical NTU RGB+D 25-joint physical skeleton, converted to zero-based IDs.
NTU_EDGES = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
             (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
             (12, 0), (13, 12), (14, 13), (15, 14), (16, 0),
             (17, 16), (18, 17), (19, 18), (21, 7), (22, 7),
             (23, 11), (24, 11)]


def build_adjacency(edges=NTU_EDGES, num_joints=25):
    adjacency = np.zeros((num_joints, num_joints), dtype=float)
    for source, target in edges:
        if source == target or not (0 <= source < num_joints and 0 <= target < num_joints):
            raise ValueError(f'invalid skeleton edge {(source, target)}')
        adjacency[source, target] = adjacency[target, source] = 1.0
    if np.any(adjacency.sum(0) == 0):
        raise ValueError('skeleton contains an isolated joint')
    return adjacency


def heat_descriptors(adjacency, scales):
    degree = adjacency.sum(1)
    inv = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(len(adjacency)) - inv @ adjacency @ inv
    values, vectors = eigh(laplacian)
    kernels = np.stack([(vectors * np.exp(-tau * values)) @ vectors.T
                        for tau in scales])
    return kernels.transpose(1, 0, 2).reshape(len(adjacency), -1), kernels


def graph_statistics(adjacency):
    """Degree, unnormalized betweenness, and mean finite geodesic distance."""
    n = len(adjacency)
    degree = adjacency.sum(1)
    distance = np.full((n, n), np.inf)
    betweenness = np.zeros(n)
    for source in range(n):
        stack, predecessors = [], [[] for _ in range(n)]
        sigma, dist = np.zeros(n), np.full(n, -1, dtype=int)
        sigma[source], dist[source], queue = 1.0, 0, [source]
        for vertex in queue:
            stack.append(vertex)
            for neighbor in np.flatnonzero(adjacency[vertex]):
                if dist[neighbor] < 0:
                    dist[neighbor] = dist[vertex] + 1; queue.append(neighbor)
                if dist[neighbor] == dist[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]; predecessors[neighbor].append(vertex)
        distance[source] = np.where(dist >= 0, dist, np.inf)
        credit = np.zeros(n)
        while stack:
            child = stack.pop()
            for parent in predecessors[child]:
                credit[parent] += sigma[parent] / sigma[child] * (1.0 + credit[child])
            if child != source: betweenness[child] += credit[child]
    betweenness /= 2.0
    mean_geodesic = np.array([np.mean(row[np.isfinite(row) & (row > 0)]) for row in distance])
    return degree, betweenness, mean_geodesic


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0: return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_associations(summary, statistics, seed=0, samples=5000):
    rng, n, result = np.random.default_rng(seed), len(summary), {}
    for name, values in statistics.items():
        estimate = spearman(summary, values)
        draws = []
        for _ in range(samples):
            index = rng.integers(0, n, n)
            value = spearman(summary[index], values[index])
            if np.isfinite(value): draws.append(value)
        if not draws: raise RuntimeError(f'bootstrap failed for {name}')
        result[name] = {'spearman_rho': estimate,
                        'bootstrap_ci_95': [float(v) for v in np.percentile(draws, [2.5, 97.5])],
                        'valid_bootstrap_samples': len(draws)}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='vis/figs_paper/fig_heat_kernel')
    parser.add_argument('--font-path', help='Times New Roman font file; or TPFS_FONT_PATH')
    parser.add_argument('--scales', type=float, nargs='+', default=[0.2, 1.0, 5.0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if any(scale <= 0 for scale in args.scales): raise ValueError('scales must be positive')
    if args.bootstrap_samples < 1: raise ValueError('bootstrap-samples must be positive')
    font_path = setup_journal_style(args.font_path, fontsize=9)
    adjacency = build_adjacency()
    descriptors, kernels = heat_descriptors(adjacency, args.scales)
    degree, betweenness, geodesic = graph_statistics(adjacency)
    summary = np.linalg.norm(descriptors, axis=1)
    stats = {'degree': degree, 'betweenness': betweenness,
             'mean_geodesic_distance': geodesic}
    associations = bootstrap_associations(summary, stats, args.seed, args.bootstrap_samples)

    fig = plt.figure(figsize=(11.5, 6.4))
    grid = fig.add_gridspec(2, 3, left=0.07, right=0.97, bottom=0.10,
                            top=0.90, wspace=0.38, hspace=0.42)
    ax_descriptor = fig.add_subplot(grid[0, :2])
    ax_norm = fig.add_subplot(grid[0, 2])
    stat_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]

    image = ax_descriptor.imshow(descriptors, aspect='auto', cmap='viridis')
    ax_descriptor.set(title='(a) Multi-scale NTU heat descriptors',
                      xlabel='Scale × destination joint', ylabel='Source joint')
    fig.colorbar(image, ax=ax_descriptor, fraction=0.025, pad=0.02,
                 label='Diffusion intensity')
    ax_norm.bar(np.arange(25), summary, color='#4c78a8')
    ax_norm.set(title='(b) Descriptor norm by joint',
                xlabel='NTU joint ID', ylabel=r'$||\phi(v)||_2$')

    tags = ['(c)', '(d)', '(e)']
    for ax, (name, values), tag in zip(stat_axes, stats.items(), tags):
        info = associations[name]
        ax.scatter(values, summary, c=np.arange(25), cmap='viridis', s=30)
        ax.set(title=(f"{tag} {name.replace('_', ' ').title()}\n"
                      f"Spearman ρ={info['spearman_rho']:.2f}"),
               xlabel=name.replace('_', ' ').title(),
               ylabel=r'$||\phi(v)||_2$')
    fig.suptitle('Method analysis: NTU skeleton heat diffusion and graph structure')

    stem = str(Path(args.out).expanduser()); ensure_outdir(str(Path(stem).parent))
    np.savez_compressed(f'{stem}.npz', adjacency=adjacency, descriptors=descriptors,
                        heat_kernels=kernels, scales=args.scales, descriptor_norm=summary,
                        degree=degree, betweenness=betweenness, mean_geodesic_distance=geodesic)
    with open(f'{stem}.csv', 'w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['joint_id', 'descriptor_norm', *stats] + [f'phi_{i}' for i in range(descriptors.shape[1])])
        for joint in range(25): writer.writerow([joint, summary[joint], *[v[joint] for v in stats.values()], *descriptors[joint]])
    metadata = {'analysis': 'method/graph analysis; no learned features or cluster claim',
                'graph': 'canonical NTU RGB+D 25-joint physical skeleton', 'edges': NTU_EDGES,
                'scales': args.scales, 'heat_laplacian': 'symmetric normalized',
                'descriptor_summary': 'L2 norm across all heat-kernel rows and scales',
                'associations': associations, 'bootstrap': {'seed': args.seed, 'samples': args.bootstrap_samples, 'unit': 'joint'},
                'font_path': font_path}
    Path(f'{stem}.meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    save_figure(fig, stem); plt.close(fig)
    print(json.dumps(associations, indent=2))


if __name__ == '__main__': main()
