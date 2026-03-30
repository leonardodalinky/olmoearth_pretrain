#!/usr/bin/env bash
# =============================================================================
# marsh_sft_eval_launch.sh — launch script for Marsh SFT evaluation
#
# Usage:
#   bash marsh_sft_eval_launch.sh
#
# Override any parameter via env vars, e.g.:
#   CHECKPOINT=/path/to/final_model.safetensors \
#   PREDICT_MODE=slide \
#   SLIDE_WINDOW_STRIDE=64 \
#   OUTPUT_DIR=/tmp/predicted_annotation \
#   bash marsh_sft_eval_launch.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$0")"
SCRIPT="${SCRIPT_DIR}/marsh_sft_eval.py"

# One or more MOSE dataset roots (space-separated in this array)
DATASET_DIRS=(
    "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/MOSE-format"
    # "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/MOSE-format"
)

# Trained checkpoint path (.pt/.pth/.safetensors)
CHECKPOINT="${CHECKPOINT:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/oe_training_log_base_both/final_model.safetensors}"

# Output root for predicted annotations
OUTPUT_DIR="${OUTPUT_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/predicted_annotation}"
# OUTPUT_DIR="${OUTPUT_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/predicted_annotation}"

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------
MODEL_SIZE="${MODEL_SIZE:-base}"
PREDICT_MODE="${PREDICT_MODE:-resize}"
SLIDE_WINDOW_STRIDE="${SLIDE_WINDOW_STRIDE:-${SLIDE_WINDOW_SIZE:-64}}"
PATCH_SIZE="${PATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=(
    --dataset_dir ${DATASET_DIRS[@]}
    --checkpoint "${CHECKPOINT}"
    --output_dir "${OUTPUT_DIR}"
    --model_size "${MODEL_SIZE}"
    --predict_mode "${PREDICT_MODE}"
    --slide_window_stride "${SLIDE_WINDOW_STRIDE}"
    --patch_size "${PATCH_SIZE}"
    --device "${DEVICE}"
)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Marsh SFT Eval"
echo "----------------------------------------------------------------------"
echo "  Dataset(s): ${DATASET_DIRS[*]}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Model     : ${MODEL_SIZE}"
echo "  Inference : mode=${PREDICT_MODE}, slide_stride=${SLIDE_WINDOW_STRIDE}, patch=${PATCH_SIZE}, device=${DEVICE}"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}"
python "${SCRIPT}" "${ARGS[@]}"
