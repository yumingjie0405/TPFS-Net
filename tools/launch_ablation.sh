#!/usr/bin/env bash
# Sequential NTU60-XSub joint-stream ablation launcher.
# Dry-run is the default; pass --run to train. Existing completed runs are
# skipped unless --force is also supplied.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEVICE=${DEVICE:-0}
MODE=dry-run
FORCE=false

usage() {
  cat <<'EOF'
Usage: tools/launch_ablation.sh [--run] [--force]

  --run    Execute training (default: print commands only)
  --force  Remove an existing run directory before training; requires --run

Set DEVICE to select the CUDA device (default: 0).
EOF
}

while (($#)); do
  case "$1" in
    --run) MODE=run ;;
    --force) FORCE=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ "$FORCE" == true && "$MODE" != run ]]; then
  printf 'error: --force requires --run\n' >&2
  exit 2
fi

CONFIG_DIR="$ROOT/config/ablation/ntu60_xsub_joint"
RUN_ROOT="$ROOT/work_dir/ablation/ntu60_xsub_joint"
BASELINE="$ROOT/work_dir/ts_hgcn_v3p1/ntu60/xsub/joint"
VARIANTS=(
  no_hk no_topology_loss tpfm_only no_proto_loss no_stage_ce no_diversity
  full k2 k6 k8 tau_005 tau_02 tau_05
)

is_complete() {
  local wd=$1 epoch checkpoint
  [[ -f "$wd/log.txt" ]] || return 1
  epoch=$(grep -E 'Epoch number: [0-9]+' "$wd/log.txt" | tail -n 1 | sed -E 's/.*Epoch number: ([0-9]+).*/\1/')
  [[ -n "$epoch" ]] || return 1
  compgen -G "$wd/runs-${epoch}-*.pt" >/dev/null
}

printf 'Baseline reference only (never launched): %s\n' "$BASELINE"
printf 'Mode: %s; device: %s; variants: %d\n' "$MODE" "$DEVICE" "${#VARIANTS[@]}"

for name in "${VARIANTS[@]}"; do
  cfg="$CONFIG_DIR/$name.yaml"
  wd="$RUN_ROOT/$name"
  [[ -f "$cfg" ]] || { printf 'error: missing config: %s\n' "$cfg" >&2; exit 1; }

  if is_complete "$wd" && [[ "$FORCE" != true ]]; then
    printf 'SKIP complete: %s\n' "$name"
    continue
  fi

  cmd=(python -u "$ROOT/main.py" --config "$cfg" --work-dir "$wd" --device "$DEVICE")
  if [[ "$MODE" == dry-run ]]; then
    if [[ -e "$wd" ]]; then state='existing artifacts'; else state='pending'; fi
    printf 'DRY-RUN %-18s (%s):' "$name" "$state"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  if [[ -e "$wd" ]]; then
    if [[ "$FORCE" == true ]]; then
      case "$wd" in
        "$RUN_ROOT"/*) rm -rf -- "$wd" ;;
        *) printf 'error: refusing to remove unexpected path: %s\n' "$wd" >&2; exit 1 ;;
      esac
    else
      printf 'error: %s has incomplete/existing artifacts; inspect it or rerun with --force\n' "$wd" >&2
      exit 1
    fi
  fi

  mkdir -p "$wd"
  log="$wd/_ablation.log"
  printf '[%s] START %s\n' "$(date -Is)" "$name" | tee "$log"
  if "${cmd[@]}" >>"$log" 2>&1; then
    printf '[%s] COMPLETE %s\n' "$(date -Is)" "$name" | tee -a "$log"
  else
    rc=$?
    printf '[%s] FAILED(%d) %s\n' "$(date -Is)" "$rc" "$name" | tee -a "$log" >&2
    exit "$rc"
  fi
done
