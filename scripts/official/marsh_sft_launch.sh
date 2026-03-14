#!/usr/bin/env bash
# =============================================================================
# base_marsh_sft_launch.sh — launch script for Marsh SFT training
#
# Usage:
#   # Single-GPU (no accelerate config needed)
#   bash base_marsh_sft_launch.sh
#
#   # Multi-GPU (configure accelerate first: accelerate config)
#   NPROC=4 bash base_marsh_sft_launch.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=checkpoints/checkpoint-epoch-0049 bash base_marsh_sft_launch.sh
#
# All parameters can be overridden via environment variables before calling
# this script, e.g.:
#   EPOCHS=200 BATCH_SIZE=8 bash base_marsh_sft_launch.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname $0)"
SCRIPT="${SCRIPT_DIR}/marsh_sft.py"

# Root of the MOSE-format dataset (required — set via env or edit here)
# DATASET_DIR="${DATASET_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/MOSE-format-filtered}"
DATASET_DIR1="/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/MOSE-format-filtered"
DATASET_DIR2="/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/MOSE-format-filtered"

# Where to write checkpoints and best_model.pt
SAVE_DIR="${SAVE_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/oe_training_log}"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_SIZE="${MODEL_SIZE:-base}"
PRETRAINED_REPO="${PRETRAINED_REPO:-allenai/OlmoEarth-v1-Base}"

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"   # first 20 % of epochs: decoder only
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"   # cosine decay floor (10 % of initial LR)

# ---------------------------------------------------------------------------
# Checkpoints / logging
# ---------------------------------------------------------------------------
SAVE_EVERY="${SAVE_EVERY:-25}"         # save a full checkpoint every N epochs
LOG_INTERVAL="${LOG_INTERVAL:-10}"    # print/log loss every N steps
RESUME_FROM="${RESUME_FROM:-}"        # set to a checkpoint dir to resume

# ---------------------------------------------------------------------------
# Wandb (falls back to WANDB_PROJECT / WANDB_NAME env vars in the script)
# ---------------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_NAME="${WANDB_NAME:-marsh_sft_$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-$MODEL_SIZE}"
# export WANDB_MODE="offline"
export WANDB_MODE="online"

# ---------------------------------------------------------------------------
# Multi-GPU: number of processes (1 = single GPU)
# ---------------------------------------------------------------------------
NPROC="${NPROC:-1}"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=(
    --dataset_dir   "${DATASET_DIR1}" "${DATASET_DIR2}"
    --model_size    "${MODEL_SIZE}"
    --pretrained_repo "${PRETRAINED_REPO}"
    --epochs        "${EPOCHS}"
    --batch_size    "${BATCH_SIZE}"
    --lr            "${LR}"
    --encoder_lr    "${ENCODER_LR}"
    --weight_decay  "${WEIGHT_DECAY}"
    --num_workers   "${NUM_WORKERS}"
    --seed          "${SEED}"
    --warmup_ratio  "${WARMUP_RATIO}"
    --lr_min_ratio  "${LR_MIN_RATIO}"
    --save_dir      "${SAVE_DIR}"
    --save_every    "${SAVE_EVERY}"
    --log_interval  "${LOG_INTERVAL}"
)

# Optional: resume from checkpoint
if [[ -n "${RESUME_FROM}" ]]; then
    ARGS+=(--resume_from "${RESUME_FROM}")
fi

# Optional: wandb
if [[ -n "${WANDB_PROJECT}" ]]; then
    ARGS+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_NAME}" ]]; then
    ARGS+=(--wandb_run_name "${WANDB_NAME}")
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Marsh SFT Training"
echo "----------------------------------------------------------------------"
echo "  Dataset  : ${DATASET_DIR1} ${DATASET_DIR2}"
echo "  Model    : ${MODEL_SIZE}  (${PRETRAINED_REPO})"
echo "  Epochs   : ${EPOCHS}  |  Batch: ${BATCH_SIZE}  |  GPUs: ${NPROC}"
echo "  LR       : decoder=${LR}  encoder=${ENCODER_LR}"
echo "  Schedule : warmup=${WARMUP_RATIO}  cosine_min=${LR_MIN_RATIO}"
echo "  Save dir : ${SAVE_DIR}  (every ${SAVE_EVERY} epochs)"
if [[ -n "${RESUME_FROM}" ]]; then
    echo "  Resume   : ${RESUME_FROM}"
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
    echo "  Wandb    : ${WANDB_PROJECT} / ${WANDB_NAME}"
fi
echo "======================================================================"

mkdir -p "${SAVE_DIR}"

if [[ "${NPROC}" -gt 1 ]]; then
    accelerate launch --num_processes "${NPROC}" "${SCRIPT}" "${ARGS[@]}"
else
    python "${SCRIPT}" "${ARGS[@]}"
fi
