#!/usr/bin/env bash
# Launch all 16 ts_hgcn_v3 trainings (4 protocols × 4 modalities) with
# bounded concurrency. Default 2 parallel; bump to 3 if GPU has headroom.
#
# Each training takes ~10 GB on RTX A6000. With 47 GB free, 3 concurrent
# fits with ~15 GB headroom. 4 is risky.
#
# Usage:
#   nohup bash tools/launch_v3_all.sh > work_dir/launch_v3_all.log 2>&1 &
#   PARALLEL=3 nohup bash tools/launch_v3_all.sh > work_dir/launch_v3_all.log 2>&1 &
#   tail -f work_dir/launch_v3_all.log
#
# Skipping: any directory that already has runs-140-*.pt is auto-skipped,
# so this script is safe to re-run after a partial completion.

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DEVICE=${DEVICE:-0}
PARALLEL=${PARALLEL:-2}

# (config_yaml, work_dir) — order matters for prioritising the joint flow first
JOBS=(
  # NTU60-CSub  (4)
  "config/ts_hgcn_v3/nturgbd-cross-subject/ts_hgcn_v3_joint.yaml         work_dir/ts_hgcn_v3/ntu60/xsub/joint"
  "config/ts_hgcn_v3/nturgbd-cross-subject/ts_hgcn_v3_bone.yaml          work_dir/ts_hgcn_v3/ntu60/xsub/bone"
  "config/ts_hgcn_v3/nturgbd-cross-subject/ts_hgcn_v3_joint_motion.yaml  work_dir/ts_hgcn_v3/ntu60/xsub/joint_motion"
  "config/ts_hgcn_v3/nturgbd-cross-subject/ts_hgcn_v3_bone_motion.yaml   work_dir/ts_hgcn_v3/ntu60/xsub/bone_motion"
  # NTU60-CV    (4)
  "config/ts_hgcn_v3/nturgbd-cross-view/ts_hgcn_v3_joint.yaml         work_dir/ts_hgcn_v3/ntu60/xview/joint"
  "config/ts_hgcn_v3/nturgbd-cross-view/ts_hgcn_v3_bone.yaml          work_dir/ts_hgcn_v3/ntu60/xview/bone"
  "config/ts_hgcn_v3/nturgbd-cross-view/ts_hgcn_v3_joint_motion.yaml  work_dir/ts_hgcn_v3/ntu60/xview/joint_motion"
  "config/ts_hgcn_v3/nturgbd-cross-view/ts_hgcn_v3_bone_motion.yaml   work_dir/ts_hgcn_v3/ntu60/xview/bone_motion"
  # NTU120-CSub (4)
  "config/ts_hgcn_v3/nturgbd120-cross-subject/ts_hgcn_v3_joint.yaml         work_dir/ts_hgcn_v3/ntu120/xsub/joint"
  "config/ts_hgcn_v3/nturgbd120-cross-subject/ts_hgcn_v3_bone.yaml          work_dir/ts_hgcn_v3/ntu120/xsub/bone"
  "config/ts_hgcn_v3/nturgbd120-cross-subject/ts_hgcn_v3_joint_motion.yaml  work_dir/ts_hgcn_v3/ntu120/xsub/joint_motion"
  "config/ts_hgcn_v3/nturgbd120-cross-subject/ts_hgcn_v3_bone_motion.yaml   work_dir/ts_hgcn_v3/ntu120/xsub/bone_motion"
  # NTU120-CSet (4)
  "config/ts_hgcn_v3/nturgbd120-cross-set/ts_hgcn_v3_joint.yaml         work_dir/ts_hgcn_v3/ntu120/xset/joint"
  "config/ts_hgcn_v3/nturgbd120-cross-set/ts_hgcn_v3_bone.yaml          work_dir/ts_hgcn_v3/ntu120/xset/bone"
  "config/ts_hgcn_v3/nturgbd120-cross-set/ts_hgcn_v3_joint_motion.yaml  work_dir/ts_hgcn_v3/ntu120/xset/joint_motion"
  "config/ts_hgcn_v3/nturgbd120-cross-set/ts_hgcn_v3_bone_motion.yaml   work_dir/ts_hgcn_v3/ntu120/xset/bone_motion"
)

echo "[$(date)] launch_v3_all: ${#JOBS[@]} jobs, parallel=${PARALLEL}, device=${DEVICE}"

# Build skip / queue lists upfront so we can report progress
queue=()
skipped=0
for entry in "${JOBS[@]}"; do
  read -r cfg wd <<<"$entry"
  if ls "$wd"/runs-140-*.pt >/dev/null 2>&1; then
    echo "  SKIP $wd (already has runs-140-*.pt)"
    skipped=$((skipped + 1))
    continue
  fi
  queue+=("$entry")
done
echo "[$(date)] $((${#JOBS[@]} - skipped)) jobs to run, $skipped already done"
echo

# launcher
run_one() {
  local cfg=$1 wd=$2
  mkdir -p "$wd"
  rm -rf "$wd/runs"  # clean stale runs/ so main.py won't double-prompt
  local log="$wd/_v3.log"
  echo "[$(date)] START $wd" | tee -a "$log"
  printf 'y\n\n' | python -u main.py \
      --config "$cfg" --work-dir "$wd" --device "$DEVICE" \
      >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date)] OK    $wd" | tee -a "$log"
  else
    echo "[$(date)] FAIL($rc) $wd" | tee -a "$log"
  fi
}

# semaphore-style: keep at most PARALLEL background jobs alive
ok=0; fail=0
running=()
for entry in "${queue[@]}"; do
  read -r cfg wd <<<"$entry"

  # block until a slot frees
  while [ ${#running[@]} -ge $PARALLEL ]; do
    new_running=()
    for pid in "${running[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_running+=("$pid")
      else
        wait "$pid"
        if [ $? -eq 0 ]; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
      fi
    done
    running=("${new_running[@]}")
    [ ${#running[@]} -ge $PARALLEL ] && sleep 30
  done

  run_one "$cfg" "$wd" &
  pid=$!
  running+=("$pid")
  echo "[$(date)] launched PID=$pid for $wd  (running=${#running[@]}/$PARALLEL)"
  sleep 5  # tiny stagger so two jobs don't initialize CUDA at the exact same instant
done

# drain
echo "[$(date)] all queued, waiting for the last batch to finish ..."
for pid in "${running[@]}"; do
  wait "$pid"
  if [ $? -eq 0 ]; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
done

echo
echo "==== launch_v3_all done ====  ok=$ok  fail=$fail  skipped=$skipped"
