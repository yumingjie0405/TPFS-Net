#!/usr/bin/env bash
# Launch the remaining experiments as background nohup jobs.
#
# Already done:
#   * ts_hgcn @ NTU60-CSub all 4 modalities (work_dir/ts_hgcn/ntu60/xsub/*)
#   * baseline @ NTU120-CSub all 4 modalities (work_dir/baseline/ntu120/xsub/*)
#
# This script launches the 24 remaining runs across:
#   baseline x { NTU60-CSub, NTU60-CV, NTU120-CSet }            x 4 modalities
#   ts_hgcn  x { NTU60-CV, NTU120-CSub, NTU120-CSet }           x 4 modalities
#
# Tune CARDS to spread across GPUs. Each card runs jobs sequentially.
# Defaults: serialize on single card 0. Override via env, e.g.:
#   CARDS="0 1 2 3" bash tools/launch_remaining.sh
#
# Logs go to work_dir/<model>/<dataset>/<protocol>/<modality>/_launch.log
# To monitor: tail -f work_dir/.../_launch.log
# To kill all: pkill -f "main.py --config"

set -e
mkdir -p work_dir/.locks
CARDS=${CARDS:-0}
read -ra CARD_ARR <<<"$CARDS"
N_CARDS=${#CARD_ARR[@]}

# ----- jobs as (label, config, work_dir) -----
JOBS=(
  # baseline NTU60-CSub
  "base_n60cs_j  config/base/nturgbd-cross-subject/hyper_joint.yaml         work_dir/baseline/ntu60/xsub/joint"
  "base_n60cs_b  config/base/nturgbd-cross-subject/hyper_bone.yaml          work_dir/baseline/ntu60/xsub/bone"
  "base_n60cs_jm config/base/nturgbd-cross-subject/hyper_joint_motion.yaml  work_dir/baseline/ntu60/xsub/joint_motion"
  "base_n60cs_bm config/base/nturgbd-cross-subject/hyper_bone_motion.yaml   work_dir/baseline/ntu60/xsub/bone_motion"
  # baseline NTU60-CV
  "base_n60cv_j  config/base/nturgbd-cross-view/hyper_joint.yaml         work_dir/baseline/ntu60/xview/joint"
  "base_n60cv_b  config/base/nturgbd-cross-view/hyper_bone.yaml          work_dir/baseline/ntu60/xview/bone"
  "base_n60cv_jm config/base/nturgbd-cross-view/hyper_joint_motion.yaml  work_dir/baseline/ntu60/xview/joint_motion"
  "base_n60cv_bm config/base/nturgbd-cross-view/hyper_bone_motion.yaml   work_dir/baseline/ntu60/xview/bone_motion"
  # baseline NTU120-CSet
  "base_n120cset_j  config/base/nturgbd120-cross-set/hyper_joint.yaml         work_dir/baseline/ntu120/xset/joint"
  "base_n120cset_b  config/base/nturgbd120-cross-set/hyper_bone.yaml          work_dir/baseline/ntu120/xset/bone"
  "base_n120cset_jm config/base/nturgbd120-cross-set/hyper_joint_motion.yaml  work_dir/baseline/ntu120/xset/joint_motion"
  "base_n120cset_bm config/base/nturgbd120-cross-set/hyper_bone_motion.yaml   work_dir/baseline/ntu120/xset/bone_motion"
  # ts_hgcn NTU60-CV
  "ts_n60cv_j  config/ts_hgcn/nturgbd-cross-view/ts_hgcn_joint.yaml         work_dir/ts_hgcn/ntu60/xview/joint"
  "ts_n60cv_b  config/ts_hgcn/nturgbd-cross-view/ts_hgcn_bone.yaml          work_dir/ts_hgcn/ntu60/xview/bone"
  "ts_n60cv_jm config/ts_hgcn/nturgbd-cross-view/ts_hgcn_joint_motion.yaml  work_dir/ts_hgcn/ntu60/xview/joint_motion"
  "ts_n60cv_bm config/ts_hgcn/nturgbd-cross-view/ts_hgcn_bone_motion.yaml   work_dir/ts_hgcn/ntu60/xview/bone_motion"
  # ts_hgcn NTU120-CSub
  "ts_n120cs_j  config/ts_hgcn/nturgbd120-cross-subject/ts_hgcn_joint.yaml         work_dir/ts_hgcn/ntu120/xsub/joint"
  "ts_n120cs_b  config/ts_hgcn/nturgbd120-cross-subject/ts_hgcn_bone.yaml          work_dir/ts_hgcn/ntu120/xsub/bone"
  "ts_n120cs_jm config/ts_hgcn/nturgbd120-cross-subject/ts_hgcn_joint_motion.yaml  work_dir/ts_hgcn/ntu120/xsub/joint_motion"
  "ts_n120cs_bm config/ts_hgcn/nturgbd120-cross-subject/ts_hgcn_bone_motion.yaml   work_dir/ts_hgcn/ntu120/xsub/bone_motion"
  # ts_hgcn NTU120-CSet
  "ts_n120cset_j  config/ts_hgcn/nturgbd120-cross-set/ts_hgcn_joint.yaml         work_dir/ts_hgcn/ntu120/xset/joint"
  "ts_n120cset_b  config/ts_hgcn/nturgbd120-cross-set/ts_hgcn_bone.yaml          work_dir/ts_hgcn/ntu120/xset/bone"
  "ts_n120cset_jm config/ts_hgcn/nturgbd120-cross-set/ts_hgcn_joint_motion.yaml  work_dir/ts_hgcn/ntu120/xset/joint_motion"
  "ts_n120cset_bm config/ts_hgcn/nturgbd120-cross-set/ts_hgcn_bone_motion.yaml   work_dir/ts_hgcn/ntu120/xset/bone_motion"
)

# Round-robin distribute jobs to cards. Each card runs its jobs sequentially.
declare -A CARD_QUEUE
for i in "${!JOBS[@]}"; do
  card_idx=$((i % N_CARDS))
  card=${CARD_ARR[$card_idx]}
  CARD_QUEUE[$card]+="${JOBS[$i]}|"
done

for card in "${CARD_ARR[@]}"; do
  queue=${CARD_QUEUE[$card]}
  IFS='|' read -ra job_lines <<<"$queue"
  echo "card $card -> ${#job_lines[@]} jobs"
  (
    for line in "${job_lines[@]}"; do
      [ -z "$line" ] && continue
      read -r label cfg wd <<<"$line"
      mkdir -p "$wd"
      log="$wd/_launch.log"
      echo "[$(date)] start $label on card $card -> $wd" | tee -a "$log"
      # echo y | feeds the existing-runs prompt; -u for unbuffered logs
      echo y | python -u main.py --config "$cfg" --work-dir "$wd" --device "$card" \
          >> "$log" 2>&1 || echo "[$(date)] $label FAILED" | tee -a "$log"
    done
  ) > "work_dir/.locks/card${card}.log" 2>&1 &
  echo "  -> queue PID $!"
done

echo
echo "All queues launched. To monitor a single job: tail -f work_dir/<...>/_launch.log"
echo "To list active runs: pgrep -af 'main.py --config'"
echo "To stop everything : pkill -f 'main.py --config'"
