"""Compute Marsh eval statistics and export overlay visualizations.

Given one or more MOSE dataset roots and corresponding predicted-annotation roots,
this script:

1. Reads eval IDs from ``VAL_IDS_FILE`` (fallback: all IDs under ``Annotations``).
2. Loads original JPEG, golden mask, and predicted mask.
3. Saves two overlay JPGs for visualization:
   - golden mask overlay: translucent orange
   - predicted mask overlay: translucent red
   - diff overlay: FP/FN union in translucent yellow-green
   - diff colormap PNG: TP/FP/FN color-separated map
4. Computes global pixel-wise metrics:
   - accuracy, precision, recall, f1

Mask convention:
- marsh: palette index < 128 (typically 1)
- non-marsh: palette index >= 128 (typically 255)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scripts.official.marsh_sft import VAL_IDS_FILE
except ModuleNotFoundError:
    from marsh_sft import VAL_IDS_FILE

logger = logging.getLogger(__name__)


def build_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Marsh eval statistics and overlays")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more MOSE dataset roots (golden labels + JPEGImages).",
    )
    parser.add_argument(
        "--predicted_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more predicted-annotation roots. Must match --dataset_dir count, or be a single root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="marsh_eval_stat",
        help="Output root for overlays and metric report.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0, 1].",
    )
    return parser.parse_args()


def _read_eval_ids(dataset_root: Path) -> list[str]:
    """Read eval IDs from VAL_IDS_FILE; fallback to all IDs in Annotations."""
    val_ids_path = dataset_root / VAL_IDS_FILE
    if val_ids_path.exists():
        ids = [line.strip() for line in val_ids_path.read_text().splitlines() if line.strip()]
        if ids:
            return ids

    ann_root = dataset_root / "Annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"No eval IDs file and no Annotations directory: {dataset_root}")
    return sorted(p.name for p in ann_root.iterdir() if p.is_dir())


def _resolve_pred_root(predicted_root: Path, dataset_root: Path) -> Path:
    """Resolve root that contains Annotations/<id>/00000.png for a dataset.

    Supported forms:
    - predicted_root/Annotations/...
    - predicted_root/<dataset_name>/Annotations/...
    """
    direct = predicted_root / "Annotations"
    if direct.is_dir():
        return predicted_root

    nested = predicted_root / dataset_root.name / "Annotations"
    if nested.is_dir():
        return predicted_root / dataset_root.name

    raise FileNotFoundError(
        "Predicted root missing Annotations directory in either " f"{direct} or {nested}"
    )


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load palette mask and convert to binary marsh mask (True=marsh)."""
    with Image.open(mask_path) as mask_img:
        mask_raw = np.array(mask_img, dtype=np.uint8)
    # Marsh labels are stored as low indices (typically 1), background is 255.
    return mask_raw < 128


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Apply translucent color on masked pixels."""
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")

    out = image_rgb.astype(np.float32).copy()
    color_f = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_f
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_diff_colormap(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Build a color-separated diff map.

    Colors:
    - TP (pred=1, gt=1): green
    - FP (pred=1, gt=0): red
    - FN (pred=0, gt=1): blue
    - TN (pred=0, gt=0): black
    """
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)

    h, w = gt_mask.shape
    diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    diff_rgb[tp] = (0, 200, 0)
    diff_rgb[fp] = (255, 0, 0)
    diff_rgb[fn] = (0, 102, 255)
    return diff_rgb


def _dataset_output_root(output_root: Path, dataset_root: Path, total_datasets: int) -> Path:
    """Choose output root for one dataset."""
    if total_datasets > 1:
        return output_root / dataset_root.name
    return output_root


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def main() -> None:
    args = build_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be within [0, 1]")

    dataset_roots = [Path(p) for p in args.dataset_dir]
    pred_roots_raw = [Path(p) for p in args.predicted_dir]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if len(pred_roots_raw) not in (1, len(dataset_roots)):
        raise ValueError(
            "--predicted_dir must provide either one root for all datasets or one root per dataset"
        )

    if len(pred_roots_raw) == 1 and len(dataset_roots) > 1:
        pred_roots = pred_roots_raw * len(dataset_roots)
    else:
        pred_roots = pred_roots_raw

    # global confusion counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    total_requested = 0
    total_compared = 0
    total_missing_pred = 0
    total_missing_golden = 0
    total_missing_jpeg = 0

    total_datasets = len(dataset_roots)

    for ds_root, pred_root_raw in zip(dataset_roots, pred_roots):
        pred_root = _resolve_pred_root(pred_root_raw, ds_root)
        eval_ids = _read_eval_ids(ds_root)

        ds_requested = len(eval_ids)
        ds_compared = 0
        ds_missing_pred = 0
        ds_missing_golden = 0
        ds_missing_jpeg = 0

        ds_tp = 0
        ds_fp = 0
        ds_fn = 0
        ds_tn = 0

        ds_out_root = _dataset_output_root(output_root, ds_root, total_datasets)
        logger.info("Dataset=%s predicted_root=%s ids=%d", ds_root, pred_root, ds_requested)

        for sample_id in eval_ids:
            jpg_path = ds_root / "JPEGImages" / sample_id / "00000.jpg"
            gt_mask_path = ds_root / "Annotations" / sample_id / "00000.png"
            pred_mask_path = pred_root / "Annotations" / sample_id / "00000.png"

            if not jpg_path.exists():
                ds_missing_jpeg += 1
                total_missing_jpeg += 1
                continue
            if not gt_mask_path.exists():
                ds_missing_golden += 1
                total_missing_golden += 1
                continue
            if not pred_mask_path.exists():
                ds_missing_pred += 1
                total_missing_pred += 1
                continue

            with Image.open(jpg_path) as jpg:
                image_rgb = np.array(jpg.convert("RGB"), dtype=np.uint8)

            gt_mask = _load_binary_mask(gt_mask_path)
            pred_mask = _load_binary_mask(pred_mask_path)

            if gt_mask.shape != pred_mask.shape:
                logger.warning(
                    "Shape mismatch id=%s gt=%s pred=%s (skip)",
                    sample_id,
                    gt_mask.shape,
                    pred_mask.shape,
                )
                continue

            if image_rgb.shape[:2] != gt_mask.shape:
                logger.warning(
                    "Image/mask mismatch id=%s image=%s mask=%s (skip)",
                    sample_id,
                    image_rgb.shape[:2],
                    gt_mask.shape,
                )
                continue

            tp = int(np.logical_and(pred_mask, gt_mask).sum())
            fp = int(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
            fn = int(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())
            tn = int(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum())

            ds_tp += tp
            ds_fp += fp
            ds_fn += fn
            ds_tn += tn

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            golden_overlay = _overlay_mask(
                image_rgb=image_rgb,
                mask=gt_mask,
                color=(255, 165, 0),  # orange
                alpha=args.overlay_alpha,
            )
            predicted_overlay = _overlay_mask(
                image_rgb=image_rgb,
                mask=pred_mask,
                color=(255, 0, 0),  # red
                alpha=args.overlay_alpha,
            )
            diff_error_mask = np.logical_xor(pred_mask, gt_mask)
            diff_overlay = _overlay_mask(
                image_rgb=image_rgb,
                mask=diff_error_mask,
                color=(154, 205, 50),  # yellow-green
                alpha=args.overlay_alpha,
            )
            diff_colormap = _build_diff_colormap(gt_mask=gt_mask, pred_mask=pred_mask)

            golden_out = ds_out_root / "overlay_golden" / sample_id / "00000.jpg"
            pred_out = ds_out_root / "overlay_predicted" / sample_id / "00000.jpg"
            diff_overlay_out = ds_out_root / "overlay_diff" / sample_id / "00000.jpg"
            diff_colormap_out = ds_out_root / "diff_colormap" / sample_id / "00000.png"
            golden_out.parent.mkdir(parents=True, exist_ok=True)
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            diff_overlay_out.parent.mkdir(parents=True, exist_ok=True)
            diff_colormap_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(golden_overlay).save(golden_out, quality=95)
            Image.fromarray(predicted_overlay).save(pred_out, quality=95)
            Image.fromarray(diff_overlay).save(diff_overlay_out, quality=95)
            Image.fromarray(diff_colormap).save(diff_colormap_out)

            ds_compared += 1
            total_compared += 1

        total_requested += ds_requested
        ds_total = ds_tp + ds_fp + ds_fn + ds_tn
        ds_acc = _safe_div(ds_tp + ds_tn, ds_total)
        ds_precision = _safe_div(ds_tp, ds_tp + ds_fp)
        ds_recall = _safe_div(ds_tp, ds_tp + ds_fn)
        ds_f1 = _safe_div(2.0 * ds_precision * ds_recall, ds_precision + ds_recall)

        logger.info(
            "Done dataset=%s requested=%d compared=%d missing_pred=%d missing_golden=%d "
            "missing_jpeg=%d acc=%.4f precision=%.4f recall=%.4f f1=%.4f out_root=%s",
            ds_root,
            ds_requested,
            ds_compared,
            ds_missing_pred,
            ds_missing_golden,
            ds_missing_jpeg,
            ds_acc,
            ds_precision,
            ds_recall,
            ds_f1,
            ds_out_root,
        )

    total_pixels = total_tp + total_fp + total_fn + total_tn
    accuracy = _safe_div(total_tp + total_tn, total_pixels)
    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    summary = {
        "requested_samples": total_requested,
        "compared_samples": total_compared,
        "missing_predicted_mask": total_missing_pred,
        "missing_golden_mask": total_missing_golden,
        "missing_jpeg": total_missing_jpeg,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    summary_path = output_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("==============================================================")
    logger.info(
        "Global metrics: acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
        accuracy,
        precision,
        recall,
        f1,
    )
    logger.info(
        "Counts: requested=%d compared=%d missing_pred=%d missing_golden=%d missing_jpeg=%d",
        total_requested,
        total_compared,
        total_missing_pred,
        total_missing_golden,
        total_missing_jpeg,
    )
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
