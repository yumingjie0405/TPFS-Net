#!/usr/bin/env bash
# Run --phase test on every work_dir that has a checkpoint but no *_right.txt yet.
#
# Usage:
#   bash tools/test_all.sh                    # auto-detect, run sequentially
#   FORCE=1 bash tools/test_all.sh            # re-test even if right.txt already exists
#   DEVICE=0 bash tools/test_all.sh           # pin to GPU 0
#
# Reads each directory's training yaml from runs-*-ckpt's sibling files: we
# infer (model, dataset, protocol, modality) from the work_dir layout and map
# back to config/<model>/<protocol>/<modality>.yaml. The mapping below covers
# every directory created by tools/launch_remaining.sh.
#
# Output per directory (created by main.py with --save-score True):
#   runs-<best>-<step>_right.txt
#   runs-<best>-<step>_wrong.txt
#   epoch1_test_score.pkl

set -u
DEVICE=${DEVICE:-0}
FORCE=${FORCE:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# work_dir layout: work_dir/<model>/<dataset>/<protocol>/<modality>
# config layout:   config/{base|ts_hgcn}/<protocol_dir>/<prefix>_<modality>.yaml
# protocol_dir / prefix differ by dataset+protocol — encode the mapping here.
cfg_for() {
  local model=$1 dataset=$2 protocol=$3 modality=$4
  local cfg_root cfg_proto prefix
  case "$model" in
    baseline)  cfg_root=config/base;     prefix=hyper    ;;
    ts_hgcn)   cfg_root=config/ts_hgcn;  prefix=ts_hgcn  ;;
    *) echo ""; return ;;
  esac
  case "$dataset/$protocol" in
    ntu60/xsub)   cfg_proto=nturgbd-cross-subject     ;;
    ntu60/xview)  cfg_proto=nturgbd-cross-view        ;;
    ntu120/xsub)  cfg_proto=nturgbd120-cross-subject  ;;
    ntu120/xset)  cfg_proto=nturgbd120-cross-set      ;;
    *) echo ""; return ;;
  esac
  echo "$cfg_root/$cfg_proto/${prefix}_${modality}.yaml"
}

todo=()
skip=()
nockpt=()

for wd in work_dir/baseline/*/*/* work_dir/ts_hgcn/*/*/*; do
  [ -d "$wd" ] || continue
  # parse path: work_dir/<model>/<dataset>/<protocol>/<modality>
  IFS='/' read -r _ model dataset protocol modality <<<"$wd"
  ckpt=$(ls -1 "$wd"/runs-*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$ckpt" ]; then
    nockpt+=("$wd")
    continue
  fi
  right=$(ls -1 "$wd"/runs-*_right.txt 2>/dev/null | head -1)
  if [ -n "$right" ] && [ "$FORCE" != "1" ]; then
    skip+=("$wd  (right.txt 已存在)")
    continue
  fi
  cfg=$(cfg_for "$model" "$dataset" "$protocol" "$modality")
  if [ -z "$cfg" ] || [ ! -f "$cfg" ]; then
    skip+=("$wd  (找不到 config: $cfg)")
    continue
  fi
  todo+=("$wd|$cfg|$ckpt")
done

echo "==== test_all.sh dry-run summary ===="
echo "需要 test: ${#todo[@]} 个"
for t in "${todo[@]}"; do
  IFS='|' read -r wd cfg ckpt <<<"$t"
  printf "  %-50s  ckpt=%s\n" "$wd" "$(basename "$ckpt")"
done
echo
echo "跳过(已有 right.txt 或无 cfg): ${#skip[@]} 个"
for s in "${skip[@]}"; do echo "  $s"; done
echo
echo "无 ckpt(还没训完): ${#nockpt[@]} 个"
for n in "${nockpt[@]}"; do echo "  $n"; done
echo

if [ "${1:-}" = "--dry" ]; then exit 0; fi
if [ ${#todo[@]} -eq 0 ]; then echo "无任务,退出"; exit 0; fi

read -r -p "回车开始,Ctrl-C 取消: " _ || true

ok=0; fail=0
for t in "${todo[@]}"; do
  IFS='|' read -r wd cfg ckpt <<<"$t"
  log="$wd/_test.log"
  echo "[$(date)] test $wd  (ckpt=$(basename "$ckpt"))" | tee -a "$log"
  python -u main.py \
      --config "$cfg" \
      --weights "$ckpt" \
      --work-dir "$wd" \
      --phase test \
      --save-score True \
      --device "$DEVICE" \
      >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    ok=$((ok+1))
    echo "[$(date)] OK  $wd" | tee -a "$log"
  else
    fail=$((fail+1))
    echo "[$(date)] FAIL($rc) $wd  -> 看 $log" | tee -a "$log"
  fi
done

echo
echo "==== done ====  ok=$ok  fail=$fail"
