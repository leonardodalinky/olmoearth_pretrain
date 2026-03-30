"""Run Marsh segmentation inference on MOSE eval images and save predicted annotations.

This script reads IDs from ``VAL_IDS_FILE`` for each dataset root, loads the specified
trained checkpoint, runs ``MarshModel.predict_image`` on each JPEG image, and writes
palette PNG masks under a MOSE-style ``Annotations/<id>/00000.png`` structure.

Prediction label mapping:
- marsh (foreground): 1
- non-marsh (background): 255
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors

try:
    from scripts.official.marsh_sft import (
        MAX_PATCH_SIZE,
        RGBtoBGRTransform,
        VAL_IDS_FILE,
        build_model,
    )
except ModuleNotFoundError:
    from marsh_sft import MAX_PATCH_SIZE, RGBtoBGRTransform, VAL_IDS_FILE, build_model

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS

logger = logging.getLogger(__name__)


def build_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Marsh SFT model on MOSE eval split")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more MOSE dataset roots.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt/.pth/.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predicted_annotation",
        help="Output root for predicted annotations.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=list(MODEL_SIZE_ARGS.keys()),
        help="Encoder size used by the checkpoint.",
    )
    parser.add_argument(
        "--predict_mode",
        type=str,
        default="resize",
        choices=["resize", "slide"],
        help="Prediction mode for predict_image.",
    )
    parser.add_argument(
        "--slide_window_stride",
        "--slide_window_size",
        dest="slide_window_stride",
        type=int,
        default=32,
        help="Sliding-window stride for predict_image (used when predict_mode='slide').",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=MAX_PATCH_SIZE,
        help="Patch size used for model inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (e.g., cuda, cpu).",
    )
    return parser.parse_args()


def _read_ids(ids_file: Path) -> list[str]:
    """Read non-empty IDs from text file."""
    return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]


def _extract_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load checkpoint and return a clean model state_dict."""
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(checkpoint_path))
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint structure at {checkpoint_path}")

    cleaned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = val
    return cleaned


def _find_palette(data_root: Path, ids: Iterable[str]) -> list[int]:
    """Read palette from one annotation sample in the dataset."""
    for sample_id in ids:
        sample_png = data_root / "Annotations" / sample_id / "00000.png"
        if not sample_png.exists():
            continue
        with Image.open(sample_png) as ref:
            palette = ref.getpalette()
            if palette is not None:
                return palette
    raise FileNotFoundError(f"No palette PNG found under {data_root / 'Annotations'} for eval IDs")


def _dataset_output_root(output_root: Path, dataset_root: Path, total_datasets: int) -> Path:
    """Choose output root for one dataset."""
    if total_datasets > 1:
        return output_root / dataset_root.name
    return output_root


def _save_predicted_mask(mask_label: np.ndarray, palette: list[int], out_path: Path) -> None:
    """Save predicted label map as palette PNG."""
    pred_index = np.where(mask_label == 1, 1, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_img = Image.fromarray(pred_index, mode="P")
    pred_img.putpalette(palette)
    pred_img.save(out_path)


def main() -> None:
    """Main inference entrypoint."""
    args = build_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_root = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    common = CommonComponents(
        training_modalities=["marsh"],
        run_name="marsh_sft_eval",
        save_folder=str(output_root),
    )
    model = build_model(common, model=args.model_size)
    state_dict = _extract_state_dict(ckpt_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning("Missing model keys: %s", load_result.missing_keys[:8])
    if load_result.unexpected_keys:
        logger.warning("Unexpected model keys: %s", load_result.unexpected_keys[:8])

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    transform = RGBtoBGRTransform()

    dataset_roots = [Path(p) for p in args.dataset_dir]
    total_written = 0
    total_requested = 0
    total_missing_jpeg = 0
    total_failed = 0
    total_pred_seconds = 0.0
    total_pred_pixels = 0
    total_marsh_pixels = 0
    total_datasets = len(dataset_roots)

    for ds_root in dataset_roots:
        val_ids_path = ds_root / VAL_IDS_FILE
        if not val_ids_path.exists():
            logger.warning("Skip dataset (missing eval ID file): %s", val_ids_path)
            continue

        eval_ids = _read_ids(val_ids_path)
        if not eval_ids:
            logger.warning("Skip dataset (empty eval ID file): %s", val_ids_path)
            continue

        ds_requested = len(eval_ids)
        ds_written = 0
        ds_missing_jpeg = 0
        ds_failed = 0
        ds_pred_seconds = 0.0
        ds_pred_pixels = 0
        ds_marsh_pixels = 0
        total_requested += ds_requested

        palette = _find_palette(ds_root, eval_ids)
        ds_out_root = _dataset_output_root(output_root, ds_root, total_datasets)
        logger.info("Evaluating dataset=%s ids=%d", ds_root, ds_requested)

        for sample_id in eval_ids:
            jpg_path = ds_root / "JPEGImages" / sample_id / "00000.jpg"
            if not jpg_path.exists():
                logger.warning("Missing JPEG, skip: %s", jpg_path)
                ds_missing_jpeg += 1
                total_missing_jpeg += 1
                continue

            try:
                with Image.open(jpg_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    t0 = time.perf_counter()
                    pred = model.predict_image(
                        image=rgb_img,
                        predict_mode=args.predict_mode,
                        slide_window_stride=args.slide_window_stride,
                        transform=transform,
                        patch_size=args.patch_size,
                    )
                    ds_pred_seconds += time.perf_counter() - t0
            except Exception:
                logger.exception("Failed prediction for sample_id=%s image=%s", sample_id, jpg_path)
                ds_failed += 1
                total_failed += 1
                continue

            pred_np = pred.detach().cpu().numpy().astype(np.uint8)
            out_png = ds_out_root / "Annotations" / sample_id / "00000.png"
            _save_predicted_mask(pred_np, palette, out_png)
            marsh_pixels = int((pred_np == 1).sum())
            pixels = int(pred_np.size)

            ds_marsh_pixels += marsh_pixels
            ds_pred_pixels += pixels
            total_marsh_pixels += marsh_pixels
            total_pred_pixels += pixels
            ds_written += 1
            total_written += 1

        avg_sec = ds_pred_seconds / ds_written if ds_written > 0 else 0.0
        marsh_ratio = ds_marsh_pixels / ds_pred_pixels if ds_pred_pixels > 0 else 0.0
        logger.info(
            "Done dataset=%s requested=%d written=%d missing_jpeg=%d failed=%d "
            "avg_pred_sec=%.3f marsh_ratio=%.4f out_root=%s",
            ds_root,
            ds_requested,
            ds_written,
            ds_missing_jpeg,
            ds_failed,
            avg_sec,
            marsh_ratio,
            ds_out_root,
        )
        total_pred_seconds += ds_pred_seconds

    overall_avg_sec = total_pred_seconds / total_written if total_written > 0 else 0.0
    overall_marsh_ratio = total_marsh_pixels / total_pred_pixels if total_pred_pixels > 0 else 0.0
    logger.info(
        "All done. requested=%d written=%d missing_jpeg=%d failed=%d "
        "avg_pred_sec=%.3f marsh_ratio=%.4f output_root=%s",
        total_requested,
        total_written,
        total_missing_jpeg,
        total_failed,
        overall_avg_sec,
        overall_marsh_ratio,
        output_root,
    )


if __name__ == "__main__":
    main()
