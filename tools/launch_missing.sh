#!/usr/bin/env bash
# Re-run the 15 jobs that crashed with EOFError on the previous attempts.
# Difference from launch_remaining.sh:
#   * feeds TWO inputs (printf 'y\n\n') so main.py's delete + tensorboard
#     prompts don't EOF-crash the process
#   * skips directories that already have a runs-140-*.pt (full training)
#   * pre-cleans any partial runs/ directory left from an aborted attempt
#
# Usage:
#   nohup bash tools/launch_missing.sh > work_dir/launch_missing.log 2>&1 &
#   tail -f work_dir/launch_missing.log

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DEVICE=${DEVICE:-0}

JOBS=(
  # baseline NTU60-CSub  (joint already done by orphan)
  "config/base/nturgbd-cross-subject/hyper_bone.yaml          work_dir/baseline/ntu60/xsub/bone"
  "config/base/nturgbd-cross-subject/hyper_joint_motion.yaml  work_dir/baseline/ntu60/xsub/joint_motion"
  "config/base/nturgbd-cross-subject/hyper_bone_motion.yaml   work_dir/baseline/ntu60/xsub/bone_motion"
  # baseline NTU60-CV
  "config/base/nturgbd-cross-view/hyper_joint.yaml         work_dir/baseline/ntu60/xview/joint"
  "config/base/nturgbd-cross-view/hyper_bone.yaml          work_dir/baseline/ntu60/xview/bone"
  "config/base/nturgbd-cross-view/hyper_joint_motion.yaml  work_dir/baseline/ntu60/xview/joint_motion"
  "config/base/nturgbd-cross-view/hyper_bone_motion.yaml   work_dir/baseline/ntu60/xview/bone_motion"
  # baseline NTU120-CSet
  "config/base/nturgbd120-cross-set/hyper_joint.yaml         work_dir/baseline/ntu120/xset/joint"
  "config/base/nturgbd120-cross-set/hyper_bone.yaml          work_dir/baseline/ntu120/xset/bone"
  "config/base/nturgbd120-cross-set/hyper_joint_motion.yaml  work_dir/baseline/ntu120/xset/joint_motion"
  "config/base/nturgbd120-cross-set/hyper_bone_motion.yaml   work_dir/baseline/ntu120/xset/bone_motion"
  # ts_hgcn NTU60-CV
  "config/ts_hgcn/nturgbd-cross-view/ts_hgcn_joint.yaml         work_dir/ts_hgcn/ntu60/xview/joint"
  "config/ts_hgcn/nturgbd-cross-view/ts_hgcn_bone.yaml          work_dir/ts_hgcn/ntu60/xview/bone"
  "config/ts_hgcn/nturgbd-cross-view/ts_hgcn_joint_motion.yaml  work_dir/ts_hgcn/ntu60/xview/joint_motion"
  "config/ts_hgcn/nturgbd-cross-view/ts_hgcn_bone_motion.yaml   work_dir/ts_hgcn/ntu60/xview/bone_motion"
)

echo "[$(date)] launch_missing: ${#JOBS[@]} jobs on device $DEVICE"

ok=0; fail=0; skipped=0
for entry in "${JOBS[@]}"; do
  read -r cfg wd <<<"$entry"

  # skip if already trained (any runs-140-*.pt exists)
  if ls "$wd"/runs-140-*.pt >/dev/null 2>&1; then
    echo "[$(date)] SKIP $wd (already has runs-140-*.pt)"
    skipped=$((skipped+1))
    continue
  fi

  # clean any partial runs/ dir from previous aborted attempts
  if [ -d "$wd/runs" ]; then
    echo "[$(date)] cleaning partial $wd/runs"
    rm -rf "$wd/runs"
  fi

  mkdir -p "$wd"
  log="$wd/_relaunch.log"
  echo "[$(date)] start $wd"  | tee -a "$log"

  # printf 'y\n\n' feeds BOTH input() prompts (delete? + refresh tensorboard)
  printf 'y\n\n' | python -u main.py \
      --config "$cfg" --work-dir "$wd" --device "$DEVICE" \
      >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    ok=$((ok+1))
    echo "[$(date)] OK   $wd" | tee -a "$log"
  else
    fail=$((fail+1))
    echo "[$(date)] FAIL($rc) $wd" | tee -a "$log"
  fi
done

echo
echo "==== launch_missing done ====  ok=$ok  fail=$fail  skipped=$skipped"
