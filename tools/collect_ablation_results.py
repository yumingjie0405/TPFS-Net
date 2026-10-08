#!/usr/bin/env python3
"""Collect observed NTU60-XSub joint ablation artifacts into CSV and JSON."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config/ablation/ntu60_xsub_joint"
DEFAULT_OUT = ROOT / "work_dir/ablation/ntu60_xsub_joint"
VARIANTS = (
    "no_hk", "no_topology_loss", "tpfm_only", "no_proto_loss",
    "no_stage_ce", "no_diversity", "full", "k2", "k6", "k8",
    "tau_005", "tau_02", "tau_05",
)
PATTERNS = {
    "best_accuracy": re.compile(r"Best accuracy:\s*([0-9.eE+-]+)"),
    "best_epoch": re.compile(r"Epoch number:\s*(\d+)"),
    "params_final": re.compile(r"Model total number of params:\s*(\d+)"),
    "params_initial": re.compile(r"# Parameters:\s*(\d+)"),
}
FIELDS = (
    "experiment", "is_baseline_reference", "status", "config", "work_dir",
    "log", "best_accuracy", "best_epoch", "params", "best_checkpoint",
)


def last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def checkpoint_for(work_dir: Path, epoch: int | None) -> Path | None:
    if epoch is None:
        return None
    candidates = list(work_dir.glob(f"runs-{epoch}-*.pt"))
    if not candidates:
        return None

    def step(path: Path) -> int:
        match = re.fullmatch(r"runs-\d+-(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=lambda path: (step(path), path.name)).resolve()


def result_row(name: str, config: Path, work_dir: Path,
               baseline: bool = False) -> dict[str, Any]:
    log = work_dir / "log.txt"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    accuracy_raw = last_match(PATTERNS["best_accuracy"], text)
    epoch_raw = last_match(PATTERNS["best_epoch"], text)
    params_raw = last_match(PATTERNS["params_final"], text)
    if params_raw is None:
        params_raw = last_match(PATTERNS["params_initial"], text)
    accuracy = float(accuracy_raw) if accuracy_raw is not None else None
    epoch = int(epoch_raw) if epoch_raw is not None else None
    params = int(params_raw) if params_raw is not None else None
    checkpoint = checkpoint_for(work_dir, epoch)
    complete = all(value is not None for value in (accuracy, epoch, params, checkpoint))
    return {
        "experiment": name,
        "is_baseline_reference": baseline,
        "status": "complete" if complete else "pending",
        "config": str(config.resolve()),
        "work_dir": str(work_dir.resolve()),
        "log": str(log.resolve()),
        "best_accuracy": accuracy,
        "best_epoch": epoch,
        "params": params,
        "best_checkpoint": str(checkpoint) if checkpoint else None,
    }


def load_work_dir(config: Path) -> Path:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    path = Path(data["work_dir"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=DEFAULT_OUT / "results.csv")
    parser.add_argument("--json", type=Path,
                        default=DEFAULT_OUT / "results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_config = (
        ROOT / "config/ts_hgcn_v3p1/nturgbd-cross-subject/ts_hgcn_v3_joint.yaml"
    )
    rows = [result_row("baseline_reference", baseline_config,
                       load_work_dir(baseline_config), baseline=True)]
    for name in VARIANTS:
        config = CONFIG_DIR / f"{name}.yaml"
        rows.append(result_row(name, config, load_work_dir(config)))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("complete", "pending")}
    print(f"Wrote {len(rows)} rows to {args.csv} and {args.json} "
          f"(complete={counts['complete']}, pending={counts['pending']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
