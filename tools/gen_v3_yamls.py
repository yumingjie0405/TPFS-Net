#!/usr/bin/env python3
"""
Generate 16 v3 yamls (4 protocols x 4 modalities) from the existing v1 yamls.
Drop-in: invokes model.ts_hgcn_v3.Model and adds the new hyperparameters.

Usage:
  python tools/gen_v3_yamls.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'config' / 'ts_hgcn'
DST = ROOT / 'config' / 'ts_hgcn_v3p1'

PROTO_DIRS = [
    'nturgbd-cross-subject',
    'nturgbd-cross-view',
    'nturgbd120-cross-subject',
    'nturgbd120-cross-set',
]
MODALITIES = ['joint', 'bone', 'joint_motion', 'bone_motion']

# protocol-dir → work_dir slug
PROTO_TO_WD = {
    'nturgbd-cross-subject': 'ntu60/xsub',
    'nturgbd-cross-view': 'ntu60/xview',
    'nturgbd120-cross-subject': 'ntu120/xsub',
    'nturgbd120-cross-set': 'ntu120/xset',
}


def patch(yaml_text: str, wd_slug: str) -> str:
    """Convert v1 yaml text → v3.1 yaml text."""
    # 1. work_dir → ts_hgcn_v3.1 to avoid collision with finished v3 runs
    yaml_text = re.sub(
        r'^work_dir:\s*\./work_dir/ts_hgcn/.*$',
        f'work_dir: ./work_dir/ts_hgcn_v3p1/{wd_slug}',
        yaml_text, count=1, flags=re.M)
    # 2. model class
    yaml_text = yaml_text.replace('model: model.ts_hgcn.Model',
                                  'model: model.ts_hgcn_v3.Model')
    # 3. λ_topo: 0.1 → 0.2 (relaxed from v3's 0.5 to avoid hurting bone stream)
    yaml_text = re.sub(r'lambda_topo:\s*0\.1\b', 'lambda_topo: 0.2', yaml_text)
    # 4. add new args after lambda_ce_stage line.
    #    ramp_start=5 (was 30): aux losses start kicking in early so SPG learns
    #    alongside main backbone instead of being shoved in late.
    #    ramp_end=60 (was 100): full strength reached by ⅓ way through training,
    #    leaving 80 epochs for joint refinement.
    new_block = (
        '  lambda_ce_stage: 0.3\n'
        '  lambda_div: 0.1\n'
        '  stage_sharpness: 40.0\n'
        '  ramp_start: 5\n'
        '  ramp_end: 60\n'
    )
    yaml_text = re.sub(r'  lambda_ce_stage:\s*0\.3\n',
                       new_block, yaml_text, count=1)
    return yaml_text


def main():
    written = 0
    for proto in PROTO_DIRS:
        wd_slug = PROTO_TO_WD[proto]
        for m in MODALITIES:
            src = SRC / proto / f'ts_hgcn_{m}.yaml'
            assert src.exists(), f'missing v1 yaml: {src}'
            dst_dir = DST / proto
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f'ts_hgcn_v3_{m}.yaml'

            with src.open() as f:
                text = f.read()
            new_text = patch(text, f'{wd_slug}/{m}')
            with dst.open('w') as f:
                f.write(new_text)
            written += 1
            print(f'  wrote {dst.relative_to(ROOT)}')
    print(f'\ndone: {written} yamls generated')


if __name__ == '__main__':
    main()
