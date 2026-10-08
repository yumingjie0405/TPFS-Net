"""Build an auditable registry of training runs under work_dir."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from tools._io import best_epoch_from_log, resolve_checkpoint


MODEL_NAMES = {
    'baseline': 'Hyper-GCN baseline',
    'ts_hgcn': 'TS-HyperGCN v1',
    'ts_hgcn_v2': 'TS-HyperGCN v2',
    'ts_hgcn_v3': 'TPFS-Net v3',
    'ts_hgcn_v3p1': 'TPFS-Net v3.1',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path):
    result = {'best_accuracy': None, 'best_epoch': None, 'params': None}
    for line in path.read_text(errors='ignore').splitlines():
        match = re.search(r'Best accuracy:\s*([0-9.]+)', line)
        if match:
            result['best_accuracy'] = float(match.group(1))
        match = re.search(r'Epoch number:\s*(\d+)', line)
        if match:
            result['best_epoch'] = int(match.group(1))
        match = re.search(r'Model total number of params:\s*(\d+)', line)
        if match:
            result['params'] = int(match.group(1))
    return result


def parse_scope(work_dir, run_dir):
    parts = run_dir.relative_to(work_dir).parts
    if len(parts) < 4:
        return None
    model, dataset, protocol, modality = parts[:4]
    return {
        'model_key': model,
        'model_name': MODEL_NAMES.get(model, model),
        'dataset': dataset,
        'protocol': protocol,
        'modality': modality,
    }


def build_registry(work_dir):
    work_dir = Path(work_dir).resolve()
    records = []
    for log_path in sorted(work_dir.glob('*/*/*/*/log.txt')):
        run_dir = log_path.parent
        scope = parse_scope(work_dir, run_dir)
        if scope is None:
            continue
        parsed = parse_log(log_path)
        checkpoints = sorted(run_dir.glob('runs-*.pt'))
        best_ckpt = None
        status = 'incomplete'
        if parsed['best_epoch'] is not None and checkpoints:
            try:
                best_ckpt = Path(resolve_checkpoint(run_dir, policy='best'))
                status = 'complete'
            except (RuntimeError, FileNotFoundError):
                status = 'checkpoint_mismatch'
        score = run_dir / 'epoch1_test_score.pkl'
        rights = sorted(run_dir.glob('*_right.txt'))
        record = {
            **scope,
            'run_dir': str(run_dir),
            'status': status,
            'best_accuracy': parsed['best_accuracy'],
            'best_accuracy_pct': (
                parsed['best_accuracy'] * 100
                if parsed['best_accuracy'] is not None else None
            ),
            'best_epoch': parsed['best_epoch'],
            'params': parsed['params'],
            'best_checkpoint': str(best_ckpt) if best_ckpt else None,
            'best_checkpoint_sha256': sha256(best_ckpt) if best_ckpt else None,
            'score_file': str(score) if score.exists() else None,
            'score_provenance': (
                'generic filename; may have been overwritten by manual test'
                if score.exists() else None
            ),
            'right_file': str(rights[-1]) if rights else None,
            'log_file': str(log_path),
        }
        records.append(record)
    return records


def write_registry(records, out_base):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix('.json')
    csv_path = out_base.with_suffix('.csv')
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    fields = list(records[0]) if records else []
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='work_dir')
    parser.add_argument('--out', default='artifacts/experiment_manifest')
    args = parser.parse_args()
    records = build_registry(args.work_dir)
    json_path, csv_path = write_registry(records, args.out)
    complete = sum(r['status'] == 'complete' for r in records)
    print(f'[registry] {len(records)} runs, {complete} complete')
    print(f'[saved] {json_path}')
    print(f'[saved] {csv_path}')


if __name__ == '__main__':
    main()
