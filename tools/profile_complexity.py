"""Profile model size and inference cost without requiring optional packages."""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tools._io import build_model, load_config, load_weights, resolve_checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def forward(model, sample):
    output = model(sample)
    return output[0] if isinstance(output, tuple) else output


def optional_ops(model, sample):
    """Use THOP only when already installed; never estimate operations."""
    if importlib.util.find_spec('thop') is None:
        return {
            'status': 'unavailable', 'tool': None,
            'reason': 'optional package thop is not installed',
            'convention': None, 'macs': None, 'flops': None,
        }
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return {
            'status': 'available', 'tool': 'thop',
            'convention': ('THOP MAC count; FLOPs reported as 2 * MACs '
                           '(one multiply-accumulate equals two FLOPs).'),
            'macs': int(macs), 'flops': int(2 * macs),
        }
    except Exception as exc:
        return {
            'status': 'unavailable', 'tool': 'thop',
            'reason': f'THOP profiling failed: {type(exc).__name__}: {exc}',
            'convention': None, 'macs': None, 'flops': None,
        }

def measure_latency(model, sample, device, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            forward(model, sample)
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            forward(model, sample)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    mean_ms = statistics.mean(timings)
    sorted_ms = sorted(timings)
    p50_ms = statistics.median(sorted_ms)
    p95_index = max(0, min(len(sorted_ms) - 1,
                           int(0.95 * len(sorted_ms) + 0.999999) - 1))
    result = {
        'warmup_iterations': warmup,
        'measured_iterations': repeats,
        'per_batch_ms': {
            'mean': mean_ms,
            'median': p50_ms,
            'p95': sorted_ms[p95_index],
            'min': min(timings),
            'max': max(timings),
        },
        'throughput_samples_per_second': sample.shape[0] * 1000.0 / mean_ms,
        'timing_method': 'perf_counter with device synchronization',
    }
    if device.type == 'cuda':
        result['cuda_peak_memory_bytes'] = int(
            torch.cuda.max_memory_allocated(device))
        result['cuda_peak_memory_reserved_bytes'] = int(
            torch.cuda.max_memory_reserved(device))
    else:
        result['cuda_peak_memory_bytes'] = None
        result['cuda_peak_memory_reserved_bytes'] = None
    return result


def write_csv(path, report):
    flat = {
        'created_utc': report['metadata']['created_utc'],
        'config_path': report['metadata']['config_path'],
        'checkpoint_path': report['metadata']['checkpoint']['path'],
        'checkpoint_sha256': report['metadata']['checkpoint']['sha256'],
        'device': report['metadata']['device'],
        'device_name': report['metadata']['device_name'],
        'dtype': report['metadata']['dtype'],
        'model_class': report['metadata']['model_class'],
        'config_json': json.dumps(report['metadata']['config'], sort_keys=True),
        'input_shape': 'x'.join(map(str, report['input_shape'])),
        'trainable_parameters': report['parameters']['trainable'],
        'total_parameters': report['parameters']['total'],
        'mac_status': report['operations']['status'],
        'mac_tool': report['operations'].get('tool'),
        'mac_reason': report['operations'].get('reason'),
        'macs': report['operations'].get('macs'),
        'flops': report['operations'].get('flops'),
        'flop_convention': report['operations'].get('convention'),
        'latency_mean_ms': report['latency']['per_batch_ms']['mean'],
        'latency_median_ms': report['latency']['per_batch_ms']['median'],
        'latency_p95_ms': report['latency']['per_batch_ms']['p95'],
        'latency_min_ms': report['latency']['per_batch_ms']['min'],
        'latency_max_ms': report['latency']['per_batch_ms']['max'],
        'throughput_samples_per_second': report['latency'][
            'throughput_samples_per_second'],
        'cuda_peak_memory_bytes': report['latency'][
            'cuda_peak_memory_bytes'],
        'cuda_peak_memory_reserved_bytes': report['latency'][
            'cuda_peak_memory_reserved_bytes'],
        'warmup_iterations': report['latency']['warmup_iterations'],
        'measured_iterations': report['latency']['measured_iterations'],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile parameters, optional operations, and inference.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--checkpoint', help='explicit checkpoint path')
    parser.add_argument('--checkpoint-policy', choices=('best', 'latest'),
                        default='best')
    parser.add_argument('--device', default='cuda:0',
                        help='torch device, including cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--joints', type=int, default=25)
    parser.add_argument('--persons', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--out-json', default='vis/complexity.json')
    parser.add_argument('--out-csv', default='vis/complexity.csv')
    args = parser.parse_args()
    if min(args.batch_size, args.channels, args.frames,
           args.joints, args.persons, args.repeats) < 1:
        parser.error('input dimensions and --repeats must be positive')
    if args.warmup < 0:
        parser.error('--warmup must be non-negative')
    return args


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested ({device}) but is not available')
    checkpoint = Path(resolve_checkpoint(
        args.work_dir, policy=args.checkpoint_policy,
        explicit=args.checkpoint)).resolve()
    model = build_model(cfg, device)
    load_weights(model, checkpoint, device)
    model.eval()
    shape = (args.batch_size, args.channels, args.frames,
             args.joints, args.persons)
    sample = torch.randn(shape, dtype=torch.float32, device=device)

    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    operations = optional_ops(model, sample)
    latency = measure_latency(model, sample, device, args.warmup, args.repeats)
    report = {
        'metadata': {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'config_path': str(config_path),
            'config': cfg,
            'work_dir': str(Path(args.work_dir).resolve()),
            'checkpoint': {
                'path': str(checkpoint),
                'sha256': file_sha256(checkpoint),
                'selection': ('explicit' if args.checkpoint
                              else args.checkpoint_policy),
            },
            'device': str(device),
            'device_name': (torch.cuda.get_device_name(device)
                            if device.type == 'cuda'
                            else platform.processor() or 'CPU'),
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'dtype': str(sample.dtype),
            'model_class': cfg.get('model'),
        },
        'input_shape': list(shape),
        'parameters': {
            'trainable': int(trainable),
            'total': int(total),
            'definition': 'exact scalar parameter count from model.parameters()',
        },
        'operations': operations,
        'latency': latency,
    }
    json_path = Path(args.out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    csv_path = Path(args.out_csv)
    write_csv(csv_path, report)

    print(f'trainable parameters: {trainable:,}')
    print(f'total parameters:     {total:,}')
    if operations['status'] == 'available':
        print(f'MACs: {operations["macs"]:,}')
        print(f'FLOPs: {operations["flops"]:,}')
        print(operations['convention'])
    else:
        print(f'MACs/FLOPs: unavailable ({operations["reason"]})')
    print(f'latency: {latency["per_batch_ms"]["mean"]:.3f} ms/batch')
    print('throughput: '
          f'{latency["throughput_samples_per_second"]:.3f} samples/s')
    if latency['cuda_peak_memory_bytes'] is not None:
        print(f'CUDA peak allocated: {latency["cuda_peak_memory_bytes"]:,} bytes')
    print(f'saved: {json_path}, {csv_path}')


if __name__ == '__main__':
    main()
