"""
Shared loaders for the visualization tools.
"""

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def import_class(path):
    mod_str, _, cls_str = path.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], cls_str)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg, device='cuda:0'):
    Model = import_class(cfg['model'])
    model = Model(**cfg['model_args']).to(device)
    model.eval()
    return model


def load_weights(model, ckpt_path, device='cuda:0'):
    state = torch.load(ckpt_path, map_location=device)
    state = OrderedDict(
        (k.split('module.')[-1], v) for k, v in state.items()
    )
    model.load_state_dict(state, strict=False)
    return model


def _epoch_from_ckpt(path):
    match = re.search(r'runs-(\d+)-', Path(path).name)
    if not match:
        raise ValueError(f'cannot parse epoch from checkpoint name: {path}')
    return int(match.group(1))


def best_epoch_from_log(work_dir):
    """Return the best epoch recorded in ``log.txt``.

    The training logs end with ``Best accuracy`` followed by ``Epoch number``.
    Returning the last such pair also handles resumed runs.
    """
    log_path = Path(work_dir) / 'log.txt'
    if not log_path.exists():
        return None
    best_epoch = None
    for line in log_path.read_text(errors='ignore').splitlines():
        match = re.search(r'Epoch number:\s*(\d+)', line)
        if match:
            best_epoch = int(match.group(1))
    return best_epoch


def resolve_checkpoint(work_dir, policy='best', explicit=None):
    """Resolve a checkpoint with an auditable policy.

    ``best`` follows the best epoch recorded by training. ``latest`` selects
    the highest epoch. ``explicit`` requires a path. No policy silently falls
    back from best to latest because that changes reported accuracy.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f'explicit checkpoint does not exist: {path}')
        return str(path)

    work = Path(work_dir)
    candidates = sorted(work.glob('runs-*.pt'), key=_epoch_from_ckpt)
    if not candidates:
        raise FileNotFoundError(f'no runs-*.pt under {work_dir}')
    if policy == 'latest':
        return str(candidates[-1])
    if policy != 'best':
        raise ValueError(f'unsupported checkpoint policy: {policy}')

    best_epoch = best_epoch_from_log(work)
    if best_epoch is None:
        raise RuntimeError(
            f'log.txt under {work_dir} has no recorded best epoch; '
            'pass an explicit checkpoint or use policy="latest"'
        )
    matches = [p for p in candidates if _epoch_from_ckpt(p) == best_epoch]
    if len(matches) != 1:
        raise RuntimeError(
            f'expected one checkpoint for best epoch {best_epoch} under '
            f'{work_dir}, found {len(matches)}'
        )
    return str(matches[0])


def latest_or_best_ckpt(work_dir):
    """Backward-compatible alias that now correctly selects the logged best."""
    return resolve_checkpoint(work_dir, policy='best')


def build_test_loader(cfg, batch_size=32, num_workers=2, max_samples=None):
    Feeder = import_class(cfg['feeder'])
    args = dict(cfg['test_feeder_args'])
    ds = Feeder(**args)
    if max_samples is not None:
        idx = np.arange(min(max_samples, len(ds)))
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )


def ensure_outdir(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    return out
