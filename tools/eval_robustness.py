"""Deterministic, test-time robustness evaluation for NTU skeleton models.

All perturbations operate on feeder output shaped ``(N, C, T, V, M)`` and
require no retraining. Results retain the historical tag-keyed JSON layout,
while adding raw per-seed measurements and auditable metadata.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tools._io import (build_model, build_test_loader, load_config,
                       load_weights, resolve_checkpoint)

# NTU RGB+D uses one-based joint labels in its documentation. Constants below
# are zero-based. Each edge is (child, parent), rooted at SpineShoulder (21).
NTU_PARENT_LINKS = tuple(
    (child - 1, parent - 1) for child, parent in (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
        (18, 17), (19, 18), (20, 19), (22, 8), (23, 8),
        (24, 12), (25, 12)))
NTU_ROOT = 20
UPPER_LIMB_JOINTS = tuple(i - 1 for i in (5, 6, 7, 8, 9, 10, 11, 12,
                                           22, 23, 24, 25))
BOTH_HAND_JOINTS = tuple(i - 1 for i in (8, 12, 22, 23, 24, 25))
LEFT_RIGHT_PAIRS = tuple((left - 1, right - 1) for left, right in (
    (5, 9), (6, 10), (7, 11), (8, 12), (22, 24), (23, 25),
    (13, 17), (14, 18), (15, 19), (16, 20)))


def _require_ntu(x):
    if x.ndim != 5 or x.shape[3] != 25:
        raise ValueError(f'expected NTU tensor (N,C,T,25,M), got {tuple(x.shape)}')

def apply_frame_drop(x, ratio, rng):
    """Randomly retain frames, then restore length by nearest kept frame."""
    n, _, length, _, _ = x.shape
    n_keep = max(int(round((1.0 - ratio) * length)), 1)
    if n_keep >= length:
        return x.clone()
    out = x.clone()
    timeline = np.arange(length)
    for sample in range(n):
        kept = np.sort(rng.choice(length, size=n_keep, replace=False))
        insert = np.searchsorted(kept, timeline)
        right = np.minimum(insert, n_keep - 1)
        left = np.maximum(insert - 1, 0)
        nearest = np.where(timeline - kept[left] <= kept[right] - timeline,
                           kept[left], kept[right])
        nearest = torch.as_tensor(nearest, dtype=torch.long, device=x.device)
        out[sample] = x[sample, :, nearest, :, :]
    return out


def apply_joint_mask(x, ratio, rng):
    """Mask a random fixed set of whole-sequence joints per sample."""
    n, _, _, joints, _ = x.shape
    n_mask = int(round(ratio * joints))
    if n_mask == 0:
        return x.clone()
    mask = torch.ones((n, joints), dtype=x.dtype, device=x.device)
    for sample in range(n):
        selected = rng.choice(joints, size=n_mask, replace=False)
        selected = torch.as_tensor(selected, dtype=torch.long, device=x.device)
        mask[sample, selected] = 0
    return x * mask.view(n, 1, 1, joints, 1)


def apply_temporal_block(x, ratio, rng):
    """Zero one contiguous temporal block, independently per sample."""
    n, _, length, _, _ = x.shape
    block = min(max(int(round(ratio * length)), 1), length)
    out = x.clone()
    for sample in range(n):
        start = int(rng.integers(0, length - block + 1))
        out[sample, :, start:start + block, :, :] = 0
    return out


def apply_fixed_joint_mask(x, joints):
    """Mask named NTU joints for every frame, person, and sample."""
    _require_ntu(x)
    out = x.clone()
    out[:, :, :, list(joints), :] = 0
    return out


def apply_gaussian_noise(x, sigma, rng):
    """Add seeded coordinate noise while preserving zero-padded entries."""
    noise = rng.normal(0.0, sigma, size=tuple(x.shape))
    noise = torch.as_tensor(noise, dtype=x.dtype, device=x.device)
    valid = x.ne(0).any(dim=1, keepdim=True)
    return x + noise * valid

def apply_bone_scaling(x, scale):
    """Scale every NTU parent-child vector, keeping the root fixed.

    The tree is reconstructed parent-first, so descendants follow a scaled
    ancestor rather than being scaled independently in global coordinates.
    """
    _require_ntu(x)
    children = {joint: [] for joint in range(25)}
    for child, parent in NTU_PARENT_LINKS:
        children[parent].append(child)
    out = x.clone()

    def rebuild(parent):
        for child in children[parent]:
            out[:, :, :, child, :] = (
                out[:, :, :, parent, :]
                + scale * (x[:, :, :, child, :] - x[:, :, :, parent, :]))
            rebuild(child)

    rebuild(NTU_ROOT)
    return out


def apply_left_right_swap(x):
    """Swap corresponding left/right NTU joint identities."""
    _require_ntu(x)
    out = x.clone()
    for left, right in LEFT_RIGHT_PAIRS:
        out[:, :, :, left, :] = x[:, :, :, right, :]
        out[:, :, :, right, :] = x[:, :, :, left, :]
    return out


def build_conditions(include_bone_scaling=True):
    """Return condition metadata and callables (x, rng) -> tensor."""
    conditions = [
        ('clean', 'clean', 0.0, False, lambda x, rng: x),
        ('drop25', 'random_frame_drop_nearest', 0.25, True,
         lambda x, rng: apply_frame_drop(x, 0.25, rng)),
        ('drop50', 'random_frame_drop_nearest', 0.50, True,
         lambda x, rng: apply_frame_drop(x, 0.50, rng)),
        ('mask10', 'random_whole_sequence_joint_mask', 0.10, True,
         lambda x, rng: apply_joint_mask(x, 0.10, rng)),
        ('mask30', 'random_whole_sequence_joint_mask', 0.30, True,
         lambda x, rng: apply_joint_mask(x, 0.30, rng)),
        ('block20', 'contiguous_temporal_block_mask', 0.20, True,
         lambda x, rng: apply_temporal_block(x, 0.20, rng)),
        ('block40', 'contiguous_temporal_block_mask', 0.40, True,
         lambda x, rng: apply_temporal_block(x, 0.40, rng)),
        ('upper_limb_mask', 'fixed_upper_limb_mask', None, False,
         lambda x, rng: apply_fixed_joint_mask(x, UPPER_LIMB_JOINTS)),
        ('both_hand_mask', 'fixed_both_hand_mask', None, False,
         lambda x, rng: apply_fixed_joint_mask(x, BOTH_HAND_JOINTS)),
    ]
    for sigma in (0.01, 0.03, 0.05):
        conditions.append((f'noise{sigma:.2f}', 'gaussian_coordinate_noise',
                           sigma, True,
                           lambda x, rng, s=sigma: apply_gaussian_noise(x, s, rng)))
    if include_bone_scaling:
        for delta in (-0.10, -0.05, 0.05, 0.10):
            label = f'bone_scale_{delta:+.0%}'
            conditions.append((label, 'ntu_bone_scaling', delta, False,
                               lambda x, rng, d=delta: apply_bone_scaling(x, 1.0 + d)))
    conditions.append(('left_right_swap', 'left_right_joint_swap', None, False,
                       lambda x, rng: apply_left_right_swap(x)))
    return conditions


@torch.no_grad()
def eval_once(model, loader, device, perturb_fn=None, rng=None):
    correct = 0
    total = 0
    for data, label, _ in tqdm(loader, leave=False):
        data = data.float().to(device)
        label = label.long().to(device)
        if perturb_fn is not None:
            data = perturb_fn(data, rng)
        output = model(data)
        if isinstance(output, tuple):
            output = output[0]
        correct += (output.argmax(dim=1) == label).sum().item()
        total += label.size(0)
    if total == 0:
        raise RuntimeError('test loader is empty')
    return correct / total

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate deterministic test-time skeleton perturbations.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--checkpoint', help='explicit checkpoint path')
    parser.add_argument('--checkpoint-policy', choices=('best', 'latest'),
                        default='best')
    parser.add_argument('--tag', required=True, help='key used in output JSON')
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default='vis/robustness.json')
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error('--num-seeds must be at least 1')

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    device = torch.device(args.device)
    checkpoint = Path(resolve_checkpoint(
        args.work_dir, policy=args.checkpoint_policy,
        explicit=args.checkpoint)).resolve()
    print(f'[checkpoint] {checkpoint}')

    model = build_model(cfg, device)
    load_weights(model, checkpoint, device)
    model.eval()
    loader = build_test_loader(cfg, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    print(f'[loader] {len(loader.dataset)} samples')

    feeder_args = cfg.get('test_feeder_args', {})
    bone_scaling_feasible = not (feeder_args.get('bone', False)
                                 or feeder_args.get('vel', False))
    conditions = build_conditions(bone_scaling_feasible)
    summaries = {}
    condition_meta = []
    for name, kind, severity, stochastic, transform in conditions:
        seeds = list(range(args.num_seeds)) if stochastic else [0]
        raw = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            accuracy = eval_once(model, loader, device, transform, rng)
            elapsed = time.perf_counter() - started
            raw.append({'seed': seed, 'accuracy': float(accuracy),
                        'accuracy_percent': float(accuracy * 100.0),
                        'elapsed_seconds': float(elapsed)})
            print(f'  [{args.tag}/{name}] seed={seed} '
                  f'acc={accuracy * 100:.2f}% ({elapsed:.1f}s)')
        values = np.asarray([item['accuracy_percent'] for item in raw])
        summaries[name] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'n_seeds': len(raw),
            'raw': raw,
        }
        condition_meta.append({
            'name': name, 'transform': kind, 'severity': severity,
            'stochastic': stochastic, 'seeds': seeds,
        })

    if not bone_scaling_feasible:
        condition_meta.extend({
            'name': f'bone_scale_{delta:+.0%}', 'transform': 'ntu_bone_scaling',
            'severity': delta, 'status': 'unavailable',
            'reason': 'requires joint-coordinate input, not bone/motion feeder output',
        } for delta in (-0.10, -0.05, 0.05, 0.10))

    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'tag': args.tag,
        'config_path': str(config_path),
        'config': cfg,
        'work_dir': str(Path(args.work_dir).resolve()),
        'checkpoint': {
            'path': str(checkpoint),
            'sha256': sha256(checkpoint),
            'selection': 'explicit' if args.checkpoint else args.checkpoint_policy,
        },
        'device': str(device),
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'dataset_samples': len(loader.dataset),
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'requested_num_seeds': args.num_seeds,
        'conditions': condition_meta,
        'joint_convention': {
            'dataset': 'NTU RGB+D 25-joint',
            'indexing': 'zero-based in implementation',
            'upper_limb_joints': list(UPPER_LIMB_JOINTS),
            'both_hand_joints': list(BOTH_HAND_JOINTS),
            'left_right_pairs': [list(pair) for pair in LEFT_RIGHT_PAIRS],
            'bone_parent_links': [list(edge) for edge in NTU_PARENT_LINKS],
            'bone_scaling_feasible': bone_scaling_feasible,
        },
    }

    # Keep condition summaries directly under the tag for compatibility with
    # callers expecting payload[tag][condition]['mean'].
    payload = dict(summaries)
    payload['_metadata'] = metadata
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing[args.tag] = payload
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + '\n')
    print(f'\n[saved] {out_path}')
    for name, result in summaries.items():
        print(f'  {name:<20s} {result["mean"]:.2f} ± '
              f'{result["std"]:.2f} (n={result["n_seeds"]})')


if __name__ == '__main__':
    main()
