#!/usr/bin/env bash
# =============================================================================
# marsh_sft_eval_stat.sh — launch script for Marsh eval statistics
#
# This script compares MOSE golden masks and predicted masks, then:
# 1) outputs overlay JPGs (golden=orange, predicted=red)
# 2) computes global acc / precision / recall / f1
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$0")"
SCRIPT="${SCRIPT_DIR}/marsh_sft_eval_stat.py"

# One or more MOSE roots (golden annotations + JPEGImages)
DATASET_DIRS=(
    "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/MOSE-format"
    # "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/MOSE-format"
)

# One or more predicted roots.
# You can provide:
#  - one root for all datasets, or
#  - one root per dataset.
PREDICTED_DIRS=(
    "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/predicted_annotation"
    # "/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/predicted_annotation"
)

# Output root for overlays + metrics_summary.json
OUTPUT_DIR="${OUTPUT_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2018_Virginia_10262025/marsh_eval_stat}"
# OUTPUT_DIR="${OUTPUT_DIR:-/sciclone/proj-ds/ai4scientist/kelin/Wetland_Mapping/NAIP_2021_Virginia_01182026/marsh_eval_stat}"

# Overlay alpha (0~1)
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.45}"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=(
    --dataset_dir "${DATASET_DIRS[@]}"
    --predicted_dir "${PREDICTED_DIRS[@]}"
    --output_dir "${OUTPUT_DIR}"
    --overlay_alpha "${OVERLAY_ALPHA}"
)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Marsh SFT Eval Stat"
echo "----------------------------------------------------------------------"
echo "  Dataset(s)   : ${DATASET_DIRS[*]}"
echo "  Predicted(s) : ${PREDICTED_DIRS[*]}"
echo "  Output       : ${OUTPUT_DIR}"
echo "  Overlay alpha: ${OVERLAY_ALPHA}"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}"
python "${SCRIPT}" "${ARGS[@]}"
