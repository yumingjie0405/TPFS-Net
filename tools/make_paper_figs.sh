#!/usr/bin/env bash
# Generate the full set of paper figures from existing checkpoints.
#
# Usage:
#   bash tools/make_paper_figs.sh                  # ours-only figures
#   OURS_WD=...  OURS_CFG=...  BASE_WD=...  BASE_CFG=...  \
#       bash tools/make_paper_figs.sh              # ours + baseline panels
#
# Required envs (with sensible defaults):
OURS_WD=${OURS_WD:-work_dir/ts_hgcn/ntu60/xsub/joint}
OURS_CFG=${OURS_CFG:-config/ts_hgcn/nturgbd-cross-subject/ts_hgcn_joint.yaml}
NUM_CLASS=${NUM_CLASS:-60}
OUT=${OUT:-vis/figs_paper}

mkdir -p "$OUT"
set -e

echo "[1/9] adjacency"
if [ -n "$BASE_WD" ] && [ -n "$BASE_CFG" ]; then
  python -m tools.vis_adjacency \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --baseline-config "$BASE_CFG" --baseline-work-dir "$BASE_WD" \
      --out "$OUT/fig_adjacency"
else
  python -m tools.vis_adjacency \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --out "$OUT/fig_adjacency"
fi

echo "[2/9] stage attention"
python -m tools.vis_stage_attention \
    --config "$OURS_CFG" --work-dir "$OURS_WD" \
    --num-samples 32 --num-classes 6 \
    --out "$OUT/fig_stage_attention"

echo "[3/9] proto tsne"
python -m tools.vis_prototype_tsne \
    --config "$OURS_CFG" --work-dir "$OURS_WD" \
    --classes 12 --out "$OUT/fig_proto_tsne"

echo "[4/9] feature tsne"
if [ -n "$BASE_WD" ] && [ -n "$BASE_CFG" ]; then
  python -m tools.vis_feature_tsne \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --baseline-config "$BASE_CFG" --baseline-work-dir "$BASE_WD" \
      --classes 12 --max-samples 1500 \
      --out "$OUT/fig_feature_tsne"
else
  python -m tools.vis_feature_tsne \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --classes 12 --max-samples 1500 \
      --out "$OUT/fig_feature_tsne"
fi

echo "[5/9] skeleton hyper-edges (3 representative samples)"
for s in 0 8 32; do
  python -m tools.vis_skeleton_hypergraph \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --layer 5 --sample-idx $s --top-edges 4 \
      --out "$OUT/fig_hyperedges_l5_s${s}"
done

echo "[6/9] spectral energy"
if [ -n "$BASE_WD" ] && [ -n "$BASE_CFG" ]; then
  python -m tools.vis_spectral_energy \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --baseline-config "$BASE_CFG" --baseline-work-dir "$BASE_WD" \
      --out "$OUT/fig_spectral_energy"
else
  python -m tools.vis_spectral_energy \
      --config "$OURS_CFG" --work-dir "$OURS_WD" \
      --out "$OUT/fig_spectral_energy"
fi

if [ -n "$OURS_RIGHT" ] && [ -n "$BASE_RIGHT" ]; then
  echo "[7/9] confusion delta"
  python -m tools.vis_confusion_delta \
      --ours "$OURS_RIGHT" --base "$BASE_RIGHT" \
      --num-class "$NUM_CLASS" --top 30 \
      --out "$OUT/fig_confusion_delta"

  echo "[8/9] per-class delta"
  python -m tools.vis_per_class_delta \
      --ours "$OURS_RIGHT" --base "$BASE_RIGHT" \
      --num-class "$NUM_CLASS" --top 15 \
      --out "$OUT/fig_per_class_delta"
else
  echo "[7-8/9] skipped (set OURS_RIGHT and BASE_RIGHT to *_right.txt)"
fi

echo "[9/9] done. Output at $OUT/"
