"""Trying to supervised-finetuning for marsh project."""

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy warning triggered by olmoearth internals
warnings.filterwarnings("ignore", message=".*pkg_resources.*", module="class_registry.*")
warnings.filterwarnings("ignore", message=".*frozen.*", module="pydantic.*")
warnings.filterwarnings("ignore", message=".*repr.*", module="pydantic.*")


import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from olmoearth_pretrain.data.concat import OlmoEarthConcatDataset
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE
from olmoearth_pretrain.data.marsh_dataset import (
    MarshDataLoader,
    MarshDataset,
    MarshDatasetConfig,
)
from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    MaskedOlmoEarthSample,
    TokensAndMasks,
)
from olmoearth_pretrain.unet.unet_parts import DoubleConv, OutConv

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

MARSH_IMAGE_SIZE = 1024

TRAIN_IDS_FILE = "train_list_goodwin.txt"
VAL_IDS_FILE = "val_list_goodwin.txt"


# =====================================================================
# Loss functions for imbalanced segmentation
# =====================================================================


class DiceLoss(nn.Module):
    """Symmetric soft Dice loss averaged over all classes.

    Computes Dice for **every** class and averages, so images that contain
    only one class (all-background or all-marsh) still produce stable,
    meaningful gradients:

    - All-background image, model predicts all-background:
      background Dice ≈ 1, marsh Dice = smooth/(0+smooth) ≈ 1  → loss ≈ 0 ✓
    - All-background image, model falsely predicts marsh:
      marsh Dice penalises the false positives correctly ✓
    - Mixed image: both terms contribute normally ✓

    ``loss = 1/C · Σ_c [1 - (2·|P_c∩T_c| + ε) / (|P_c| + |T_c| + ε)]``
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W)   targets : (B, H, W) long
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        n_classes = probs.shape[1]
        total = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        for c in range(n_classes):
            prob_c = probs[:, c]  # (B, H, W)
            target_c = (targets == c).float()  # (B, H, W)
            intersection = (prob_c * target_c).sum(dim=(1, 2))
            cardinality = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            total += 1.0 - dice_c.mean()
        return total / n_classes


class DiceCELoss(nn.Module):
    """Dice + class-weighted CrossEntropy for imbalanced binary segmentation.

    Args:
        marsh_weight: Weight applied to the marsh class (label=1) in CE.
            A value of ``N_background / N_marsh`` is a good starting point;
            typical range 3–10 for sparse foreground.
        dice_weight: Scalar multiplier on the Dice term (default 1.0).
    """

    def __init__(self, marsh_weight: float = 3.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.marsh_weight = marsh_weight
        self._dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = torch.tensor([1.0, self.marsh_weight], device=logits.device, dtype=logits.dtype)
        ce = nn.functional.cross_entropy(logits, targets, weight=weight)
        dice = self._dice(logits, targets)
        return ce + self.dice_weight * dice


class RGBtoBGRTransform(Transform):
    """Swap R and B channels: [R, G, B, IR] → [B, G, R, IR].

    S2 L2A pretrained weights expect BGR band order in the first 3 bands.
    Marsh data is stored as RGB.  This transform swaps channels 0 and 2
    on the ``marsh`` modality (last axis of the ``(B, H, W, T, C)`` tensor).
    """

    def apply(self, batch: OlmoEarthSample) -> OlmoEarthSample:
        marsh_data = batch.marsh
        if marsh_data is None:
            return batch
        # Swap channel 0 (R) ↔ channel 2 (B) — last dim is the band axis
        idx = list(range(marsh_data.shape[-1]))  # [0, 1, 2, 3]
        idx[0], idx[2] = idx[2], idx[0]  # [2, 1, 0, 3]
        swapped = marsh_data[..., idx]
        return batch._replace(marsh=swapped)


def build_args() -> argparse.Namespace:
    """Parse command-line arguments for Marsh SFT training."""
    parser = argparse.ArgumentParser(description="Marsh SFT Training")

    # Data
    data_g = parser.add_argument_group("data")
    data_g.add_argument(
        "--dataset_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more MOSE-format dataset roots. Example: --dataset_dir /data/a /data/b",
    )

    # Model
    model_g = parser.add_argument_group("model")
    model_g.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=list(MODEL_SIZE_ARGS.keys()),
        help="Encoder size (default: base).",
    )
    model_g.add_argument(
        "--pretrained_repo",
        type=str,
        default="allenai/OlmoEarth-v1-Base",
        help="HuggingFace repo for pretrained weights.",
    )

    # Training
    train_g = parser.add_argument_group("training")
    train_g.add_argument("--epochs", type=int, default=100)
    train_g.add_argument("--batch_size", type=int, default=16)
    train_g.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate.")
    train_g.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-5,
        help="Encoder learning rate (fine-tuning).",
    )
    train_g.add_argument("--weight_decay", type=float, default=1e-4)
    train_g.add_argument("--num_workers", type=int, default=4)
    train_g.add_argument("--seed", type=int, default=42)

    # Schedule
    sched_g = parser.add_argument_group("schedule")
    sched_g.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        help="Fraction of epochs for decoder-only warmup (default: 0.2).",
    )
    sched_g.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="Cosine decay minimum as fraction of initial LR (default: 0.1).",
    )

    # Checkpoints
    ckpt_g = parser.add_argument_group("checkpoints")
    ckpt_g.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint and best-model saves.",
    )
    ckpt_g.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs.")
    ckpt_g.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training.",
    )

    # Loss
    loss_g = parser.add_argument_group("loss")
    loss_g.add_argument(
        "--marsh_weight",
        type=float,
        default=3.0,
        help="Class weight for the marsh foreground in CE loss (default: 3.0). "
        "Increase when marsh pixels are very sparse.",
    )
    loss_g.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Multiplier on the Dice term in DiceCE loss (default: 1.0).",
    )

    # Logging
    log_g = parser.add_argument_group("logging")
    log_g.add_argument("--log_interval", type=int, default=10, help="Log loss every N steps.")
    log_g.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project (default: from WANDB_PROJECT env var).",
    )
    log_g.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: from WANDB_NAME env var).",
    )

    return parser.parse_args()


class MarshModel(nn.Module):
    """Encoder + UNet-style progressive-upsample decoder for marsh segmentation.

    The encoder produces patch tokens of shape ``(B, P_H, P_W, D)`` where
    """

    N_CLASSES = 2  # binary segmentation (background=0, marsh=1)

    def __init__(
        self,
        marsh_image_size: int,
        encoder: Encoder,
        n_classes: int = 2,
    ):
        super().__init__()
        self.marsh_image_size = marsh_image_size
        self.n_classes = n_classes
        self.encoder = encoder

        D = encoder.embedding_size  # e.g. 768

        # Transposed-conv decoder: 3 stages of ×2 upsample → 32 → 256
        # Modality.MARSH has image_tile_size_factor=1, patch_size=8, so:
        #   token grid = target_hw / (1×8) = 32  (for 256px input)
        #   decoder needs ×8 = 2³ → 3 upsampling stages
        # Each stage: ConvTranspose2d(stride=2) doubles spatial dims, then
        # DoubleConv (Conv-BN-ReLU ×2) refines features.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(D, 256, kernel_size=2, stride=2),  # 32  → 64
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 64  → 128
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 128 → 256
            DoubleConv(64, 64),
            OutConv(64, n_classes),  # (B, 2, 256, 256)
        )

    def forward(self, x: MaskedOlmoEarthSample, patch_size: int) -> torch.Tensor:
        """Forward pass: encoder → decoder → per-pixel logits.

        Returns:
            logits of shape ``(B, n_classes, H, W)``
        """
        # --- Encoder ---
        d = self.encoder(x, patch_size)
        tokens_and_masks: TokensAndMasks = d["tokens_and_masks"]
        marsh_tokens = tokens_and_masks.marsh  # (B, P_H, P_W, T=1, BandSets=1, D)

        # Squeeze singleton T & BandSets dims, then to channel-first for Conv2d
        features = rearrange(marsh_tokens, "b ph pw 1 1 d -> b d ph pw")  # (B, D, P_H, P_W)

        # --- Decoder ---
        logits = self.decoder(features)  # (B, n_classes, H, W)
        return logits

    @torch.no_grad()
    def predict(self, x: MaskedOlmoEarthSample, patch_size: int) -> torch.Tensor:
        """Predict per-pixel classes from marsh tokens."""
        logits = self(x, patch_size)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_image(
        self,
        image: Image,
        predict_mode: str = "resize",
        slide_window_stride: int = 32,
        transform: Transform | None = None,
        patch_size: int = MAX_PATCH_SIZE,
    ) -> torch.Tensor:
        """Predict per-pixel classes from an image.

        Args:
            image: PIL image to predict on.
            predict_mode: Prediction mode. One of {"resize", "slide"}.
            slide_window_stride: Step size for sliding-window mode (in pixels).
            transform: Optional transform to apply to the sample
            patch_size: Patch size for the encoder

        Returns:
            Predicted class labels as torch.Tensor of shape (H, W)
        """
        import numpy as np
        import torch.nn.functional as F

        from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
        from olmoearth_pretrain.data.marsh_dataset import collate_marsh_batched

        device = next(self.parameters()).device
        normalizer_computed = Normalizer(Strategy.COMPUTED)

        def _build_masked_sample(tile: np.ndarray) -> MaskedOlmoEarthSample:
            h, w, c = tile.shape
            if c == 3:
                ir = np.full((h, w, 1), MISSING_VALUE, dtype=tile.dtype)
                tile = np.concatenate([tile, ir], axis=2)
            elif c != 4:
                raise ValueError(f"Expected 3 or 4 channels, got {c} channels.")

            tile = tile.astype(np.float32)

            # normalize (use NAIP stats — marsh shares the same bands)
            missing_mask = tile == MISSING_VALUE
            tile = normalizer_computed.normalize(Modality.NAIP, tile)
            tile = np.where(missing_mask, 0, tile)

            marsh_data = tile.astype(np.float32)
            marsh_data = np.expand_dims(marsh_data, axis=2)  # (H, W, 1, 4)
            timestamps = np.array([[1, 0, 2023]], dtype=np.int64)
            sample = OlmoEarthSample(
                marsh=marsh_data,
                timestamps=timestamps,
            )
            dummy_labels = np.zeros((IMAGE_TILE_SIZE, IMAGE_TILE_SIZE), dtype=np.int64)
            masked_sample = collate_marsh_batched(
                [
                    (
                        patch_size,
                        sample,
                        dummy_labels,
                    )
                ],
                transform=transform,
            )[1]
            return masked_sample

        # Convert PIL Image to numpy array
        img_array = np.array(image)  # (H, W, C)
        orig_h, orig_w = img_array.shape[:2]

        if predict_mode not in {"resize", "slide"}:
            raise ValueError(
                f"Invalid predict_mode={predict_mode!r}, expected 'resize' or 'slide'."
            )

        if predict_mode == "resize":
            img_resized = image.resize((IMAGE_TILE_SIZE, IMAGE_TILE_SIZE), Image.BILINEAR)
            img_array_resized = np.array(img_resized)  # (256, 256, C)
            masked_sample = _build_masked_sample(img_array_resized).to_device(device)

            logits = self(masked_sample, patch_size)  # (1, n_classes, 256, 256)
            preds = torch.argmax(logits, dim=1).squeeze(0)  # (256, 256)

            preds_resized = (
                F.interpolate(
                    preds.unsqueeze(0).unsqueeze(0).float(),
                    size=(orig_h, orig_w),
                    mode="nearest",
                )
                .squeeze()
                .long()
            )
            return preds_resized

        if slide_window_stride <= 0:
            raise ValueError(
                f"slide_window_stride must be > 0 for slide mode, got {slide_window_stride}."
            )

        # Initialize probability accumulator and count map
        prob_sum = torch.zeros((self.n_classes, orig_h, orig_w), device=device, dtype=torch.float32)
        count_map = torch.zeros((orig_h, orig_w), device=device, dtype=torch.float32)

        # Sliding window iteration
        for y in range(0, orig_h, slide_window_stride):
            for x in range(0, orig_w, slide_window_stride):
                # Calculate window boundaries
                y_end = min(y + IMAGE_TILE_SIZE, orig_h)
                x_end = min(x + IMAGE_TILE_SIZE, orig_w)

                # Extract window
                window = img_array[y:y_end, x:x_end]  # (h, w, C)
                window_h, window_w = window.shape[:2]

                # Pad to IMAGE_TILE_SIZE if needed
                if window_h < IMAGE_TILE_SIZE or window_w < IMAGE_TILE_SIZE:
                    padded = np.zeros(
                        (IMAGE_TILE_SIZE, IMAGE_TILE_SIZE, window.shape[2]),
                        dtype=window.dtype,
                    )
                    padded[:window_h, :window_w] = window
                    window = padded

                masked_sample = _build_masked_sample(window).to_device(device)

                # Predict
                logits = self(masked_sample, patch_size)  # (1, n_classes, 256, 256)
                probs = torch.softmax(logits, dim=1).squeeze(0)  # (n_classes, 256, 256)

                # Extract valid region (remove padding if any)
                valid_probs = probs[:, :window_h, :window_w]  # (n_classes, h, w)

                # Accumulate probabilities
                prob_sum[:, y:y_end, x:x_end] += valid_probs
                count_map[y:y_end, x:x_end] += 1.0

        # Average probabilities
        prob_avg = prob_sum / count_map.unsqueeze(0).clamp(min=1.0)

        # Get final predictions via argmax
        final_preds = torch.argmax(prob_avg, dim=0)  # (H, W)

        return final_preds


def build_model(common: CommonComponents, model: str = "base") -> MarshModel:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS[model]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_linear_patch_embed=False,
    )
    encoder = encoder_config.build()
    return MarshModel(marsh_image_size=MARSH_IMAGE_SIZE, encoder=encoder)


def build_dataset(args: argparse.Namespace) -> MarshDataset:
    """Build the dataset for an experiment."""
    return MarshDatasetConfig(data_dir=args.dataset_dir).build()


def load_pretrained_encoder_weights(model: MarshModel, weights_path: str) -> list[str]:
    """Load S2 L2A pretrained encoder weights into *model*, renaming to ``marsh``.

    Returns:
        List of missing keys from ``load_state_dict(strict=False)``.
    """
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_weights = {
        k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")
    }
    # sentinel2_l2a → marsh
    for k in list(encoder_weights.keys()):
        if "sentinel2_l2a" in k:
            encoder_weights[k.replace("sentinel2_l2a", "marsh")] = encoder_weights.pop(k)
    # Remove extra BandSets that marsh doesn't have
    for k in list(encoder_weights.keys()):
        if "marsh__1" in k or "marsh__2" in k:
            encoder_weights.pop(k)
    # Keep only the first BandSet channel embedding
    chan_key = "composite_encodings.per_modality_channel_embeddings.marsh"
    if chan_key in encoder_weights:
        encoder_weights[chan_key] = encoder_weights[chan_key][:1]

    result = model.encoder.load_state_dict(encoder_weights, strict=False)
    return result.missing_keys


def train(
    model: MarshModel,
    train_loader: MarshDataLoader,
    *,
    eval_loader: MarshDataLoader | None = None,
    epochs: int = 100,
    lr: float = 1e-4,
    encoder_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    log_interval: int = 10,
    save_dir: str = "checkpoints",
    save_every: int = 5,
    resume_from: str | None = None,
    warmup_ratio: float = 0.2,
    lr_min_ratio: float = 0.1,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
    marsh_weight: float = 5.0,
    dice_weight: float = 1.0,
) -> MarshModel:
    """Train MarshModel with HuggingFace Accelerate.

    Features
    --------
    - **Accelerate** for device placement, mixed-precision, multi-GPU.
    - **wandb** logging (project defaults to ``WANDB_PROJECT`` env var).
    - **Two-phase schedule**: first ``warmup_ratio`` of epochs trains the
      decoder only (encoder frozen); the remaining epochs train the full
      model.
    - **Cosine LR decay** starting after the warmup phase, decaying to
      ``lr_min_ratio × lr``.
    - **Periodic checkpoints** every ``save_every`` epochs.
    - **Resume** from a previous checkpoint via ``resume_from``.

    Args:
        model: MarshModel (encoder + decoder).
        train_loader: MarshDataLoader.
        eval_loader: Optional MarshDataLoader for evaluation after each epoch.
        epochs: Total training epochs.
        lr: Decoder learning rate.
        encoder_lr: Encoder learning rate (fine-tuning).
        weight_decay: AdamW weight decay.
        log_interval: Log metrics every N optimiser steps.
        save_dir: Root directory for checkpoints.
        save_every: Save a full checkpoint every N epochs.
        resume_from: Checkpoint directory to resume from.
        warmup_ratio: Fraction of epochs for decoder-only warmup.
        lr_min_ratio: Cosine-decay lower bound (fraction of initial LR).
        wandb_project: Wandb project name (``None`` → disabled).
        wandb_run_name: Wandb run name.
        seed: Random seed.

    Returns:
        The trained model (unwrapped).
    """
    # ------------------------------------------------------------------
    # 1. Accelerator
    # ------------------------------------------------------------------
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
    )

    if wandb_project:
        tracker_init_kwargs: dict = {}
        if wandb_run_name:
            tracker_init_kwargs["wandb"] = {"name": wandb_run_name}
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "epochs": epochs,
                "lr": lr,
                "encoder_lr": encoder_lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "lr_min_ratio": lr_min_ratio,
                "batch_size": train_loader.batch_size,
                "seed": seed,
                "marsh_weight": marsh_weight,
                "dice_weight": dice_weight,
            },
            init_kwargs=tracker_init_kwargs,
        )

    # ------------------------------------------------------------------
    # 2. Optimiser  (two param-groups: encoder + decoder)
    # ------------------------------------------------------------------
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},  # group 0
            {"params": decoder_params, "lr": lr},  # group 1
        ],
        weight_decay=weight_decay,
    )

    # ------------------------------------------------------------------
    # 3. LR schedule  (per optimiser-step)
    #    Phase 1 (warmup): encoder LR = 0, decoder LR = constant
    #    Phase 2 (full):   cosine decay from 1.0 → lr_min_ratio for both
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def _encoder_lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _decoder_lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=[_encoder_lr_lambda, _decoder_lr_lambda])

    # ------------------------------------------------------------------
    # 4. Accelerate-prepare  (model, optimiser, scheduler)
    #    DataLoader is *not* prepared — it uses a custom IterableDataset
    #    that handles batching internally.
    # ------------------------------------------------------------------
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    criterion = DiceCELoss(marsh_weight=marsh_weight, dice_weight=dice_weight)
    accelerator.print(f"Loss: DiceCELoss(marsh_weight={marsh_weight}, dice_weight={dice_weight})")

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isdir(resume_from):
        accelerator.load_state(resume_from)
        meta_path = os.path.join(resume_from, "training_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            start_epoch = meta["epoch"] + 1
            global_step = meta["global_step"]
        accelerator.print(f"Resumed from {resume_from}, starting epoch {start_epoch}")

    # ------------------------------------------------------------------
    # 6. Phase tracking — freeze encoder during warmup
    # ------------------------------------------------------------------
    warmup_epoch_end = int(epochs * warmup_ratio)
    raw_model: MarshModel = accelerator.unwrap_model(model)
    in_warmup = start_epoch < warmup_epoch_end

    if in_warmup:
        for p in raw_model.encoder.parameters():
            p.requires_grad = False
        accelerator.print(f"Phase 1 (decoder-only): epochs 0 – {warmup_epoch_end - 1}")

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # --- Phase transition: unfreeze encoder ---
        if in_warmup and epoch >= warmup_epoch_end:
            for p in raw_model.encoder.parameters():
                p.requires_grad = True
            in_warmup = False
            accelerator.print(f"Phase 2 (full model): epoch {epoch} — encoder unfrozen")

        train_loader.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            patch_size, masked_sample, labels = batch
            masked_sample = masked_sample.to_device(accelerator.device)
            labels = labels.to(accelerator.device)

            logits = model(masked_sample, patch_size)
            loss = criterion(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                enc_lr = scheduler.get_last_lr()[0]
                dec_lr = scheduler.get_last_lr()[1]
                log_dict = {
                    "train/loss": loss.item(),
                    "train/encoder_lr": enc_lr,
                    "train/decoder_lr": dec_lr,
                    "train/epoch": epoch,
                }
                if wandb_project:
                    accelerator.log(log_dict, step=global_step)
                accelerator.print(
                    f"  [step {global_step}] loss={loss.item():.4f}  "
                    f"enc_lr={enc_lr:.2e}  dec_lr={dec_lr:.2e}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        accelerator.print(
            f"Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}  steps={num_batches}"
        )
        if wandb_project:
            accelerator.log({"train/epoch_avg_loss": avg_loss}, step=global_step)

        # --- Evaluation ---
        if eval_loader is not None:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            total_tp = 0  # true positives  (pred=1, label=1)
            total_fp = 0  # false positives (pred=1, label=0)
            total_fn = 0  # false negatives (pred=0, label=1)
            total_correct = 0
            total_pixels = 0

            with torch.no_grad():
                for eval_batch in eval_loader:
                    ps, ms, lbl = eval_batch
                    ms = ms.to_device(accelerator.device)
                    lbl = lbl.to(accelerator.device)

                    logits = model(ms, ps)
                    loss_val = criterion(logits, lbl)
                    eval_loss += loss_val.item()
                    eval_batches += 1

                    preds = logits.argmax(dim=1)  # (B, H, W)
                    total_correct += (preds == lbl).sum().item()
                    total_pixels += lbl.numel()
                    total_tp += ((preds == 1) & (lbl == 1)).sum().item()
                    total_fp += ((preds == 1) & (lbl == 0)).sum().item()
                    total_fn += ((preds == 0) & (lbl == 1)).sum().item()

            eval_avg_loss = eval_loss / max(eval_batches, 1)
            accuracy = total_correct / max(total_pixels, 1)
            precision = total_tp / max(total_tp + total_fp, 1)
            recall = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            accelerator.print(
                f"  Eval: loss={eval_avg_loss:.4f}  acc={accuracy:.4f}  "
                f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}"
            )
            if wandb_project:
                accelerator.log(
                    {
                        "eval/loss": eval_avg_loss,
                        "eval/accuracy": accuracy,
                        "eval/precision": precision,
                        "eval/recall": recall,
                        "eval/f1": f1,
                    },
                    step=global_step,
                )

        # --- Periodic checkpoint ---
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_dir = os.path.join(save_dir, f"checkpoint-epoch-{epoch:04d}")
            accelerator.save_state(ckpt_dir)
            if accelerator.is_main_process:
                torch.save(
                    {"epoch": epoch, "global_step": global_step, "avg_loss": avg_loss},
                    os.path.join(ckpt_dir, "training_meta.pt"),
                )
            accelerator.print(f"  Saved checkpoint → {ckpt_dir}")

        # --- Best model ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            if accelerator.is_main_process:
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save(raw_model.state_dict(), best_path)
                accelerator.print(f"  New best (loss={avg_loss:.4f}) → {best_path}")

    if wandb_project:
        accelerator.end_training()

    return raw_model


# =====================================================================
# Full training entry point
# =====================================================================


def main() -> None:
    """Complete Marsh SFT training pipeline.

    Usage::

        # Single GPU
        python base_marsh_sft.py --dataset_dir /data/marsh --epochs 100

        # Multi-GPU via accelerate
        accelerate launch base_marsh_sft.py --dataset_dir /data/marsh --epochs 100

        # Resume from checkpoint
        python base_marsh_sft.py --dataset_dir /data/marsh --resume_from checkpoints/checkpoint-epoch-0049
    """
    args = build_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(f"Args: {vars(args)}")

    # --- Wandb (env vars as fallback) ---
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    wandb_run_name = args.wandb_run_name or os.environ.get("WANDB_NAME")

    # --- Training Dataset (possibly multiple roots, then concatenated) ---
    train_datasets: list[MarshDataset] = []
    eval_datasets: list[MarshDataset] = []
    target_hw: int = IMAGE_TILE_SIZE

    for dataset_dir in args.dataset_dir:
        dataset_root = Path(dataset_dir)

        train_ids_path = dataset_root / TRAIN_IDS_FILE
        if train_ids_path.exists():
            logger.info(f"Using training IDs from {train_ids_path}")
            include_ids = [
                line.strip() for line in train_ids_path.read_text().splitlines() if line.strip()
            ]
        else:
            include_ids = None

        ds_config = MarshDatasetConfig(
            data_dir=dataset_dir,
            split="train",
            normalize=True,
            training_modalities=["marsh"],
            seed=args.seed,
            include_ids=include_ids,
        )
        target_hw = ds_config.target_hw
        train_ds = ds_config.build()
        train_ds.prepare()
        train_datasets.append(train_ds)
        logger.info(f"Train dataset ready: root={dataset_root} samples={len(train_ds)}")

        val_ids_path = dataset_root / VAL_IDS_FILE
        if val_ids_path.exists():
            logger.info(f"Using evaluation IDs from {val_ids_path}")
            eval_include_ids = [
                line.strip() for line in val_ids_path.read_text().splitlines() if line.strip()
            ]
            eval_ds_config = MarshDatasetConfig(
                data_dir=dataset_dir,
                split="eval",
                normalize=True,
                training_modalities=["marsh"],
                seed=args.seed,
                include_ids=eval_include_ids,
            )
            eval_ds = eval_ds_config.build()
            eval_ds.prepare()
            eval_datasets.append(eval_ds)
            logger.info(f"Eval dataset ready: root={dataset_root} samples={len(eval_ds)}")
        else:
            logger.warning(f"No evaluation IDs file found at {val_ids_path}, skipping this root.")

    if not train_datasets:
        raise ValueError("No training datasets were built from --dataset_dir")

    if len(train_datasets) == 1:
        dataset: MarshDataset | OlmoEarthConcatDataset = train_datasets[0]
    else:
        dataset = OlmoEarthConcatDataset(train_datasets)
        dataset.prepare()
    logger.info(
        f"Merged training dataset ready: roots={len(train_datasets)} samples={len(dataset)}"
    )

    if len(eval_datasets) == 0:
        eval_dataset: MarshDataset | OlmoEarthConcatDataset | None = None
        logger.warning("No evaluation datasets were built; evaluation will be skipped.")
    elif len(eval_datasets) == 1:
        eval_dataset = eval_datasets[0]
    else:
        eval_dataset = OlmoEarthConcatDataset(eval_datasets)
        eval_dataset.prepare()
    if eval_dataset is not None:
        logger.info(
            f"Merged evaluation dataset ready: roots={len(eval_datasets)} samples={len(eval_dataset)}"
        )

    # --- Model + pretrained weights ---
    common = CommonComponents(
        training_modalities=["marsh"],
        run_name=wandb_run_name or "marsh_sft",
        save_folder=args.save_dir,
    )
    model = build_model(common, model=args.model_size)

    weights_path = hf_hub_download(repo_id=args.pretrained_repo, filename="weights.pth")
    missing = load_pretrained_encoder_weights(model, weights_path)
    if missing:
        logger.warning(f"Missing encoder keys ({len(missing)}): {missing[:5]} ...")

    # --- DataLoader ---
    transform = RGBtoBGRTransform()
    # sampled_hw_p: token-grid side length = target_hw / patch_size
    # MARSH has image_tile_size_factor=1, so token_grid = target_hw / patch_size
    train_loader = MarshDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        patch_size=MAX_PATCH_SIZE,
        sampled_hw_p=target_hw // MAX_PATCH_SIZE,
        token_budget=None,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=True,
        transform=transform,
        drop_last=True,
    )
    if eval_dataset is not None:
        eval_loader = MarshDataLoader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            patch_size=MAX_PATCH_SIZE,
            sampled_hw_p=target_hw // MAX_PATCH_SIZE,
            token_budget=None,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=True,
            transform=transform,
            drop_last=False,
        )
    else:
        eval_loader = None

    # --- Train ---
    model = train(
        model,
        train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        warmup_ratio=args.warmup_ratio,
        lr_min_ratio=args.lr_min_ratio,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        seed=args.seed,
        marsh_weight=args.marsh_weight,
        dice_weight=args.dice_weight,
    )
    logger.info("Training complete.")

    # --- Save final model (unwrapped) ---
    final_path = Path(args.save_dir) / "final_model.safetensors"
    save_safetensors(model.state_dict(), str(final_path))
    logger.info(f"Final model saved → {final_path}")


if __name__ == "__main__":
    # First, donwload checkpoint from huggingface using huggingface_hub library
    # TODO: download
    # weights_path = hf_hub_download(repo_id="allenai/OlmoEarth-v1-Base", filename="weights.pth")
    # ---- Quick smoke test: run with  python base_marsh_sft.py --smoke ----
    if "--smoke" in sys.argv:
        logging.basicConfig(level=logging.INFO)
        logger.info("=== Smoke test: mock dataset + dataloader ===")

        # Build a mock dataset (no real data needed)
        mock_ds = MarshDatasetConfig(
            mock=True,
            mock_hw=IMAGE_TILE_SIZE,
            mock_n=32,
            normalize=True,
            training_modalities=["marsh"],
        ).build()
        mock_ds.prepare()
        logger.info(f"Dataset length: {len(mock_ds)}")

        # Build dataloader
        dl = MarshDataLoader(
            dataset=mock_ds,
            batch_size=4,
            patch_size=MAX_PATCH_SIZE,
            sampled_hw_p=IMAGE_TILE_SIZE // MAX_PATCH_SIZE,  # full image
            token_budget=None,
            shuffle=True,
            num_workers=0,  # keep it simple for smoke test
            seed=0,
            pin_memory=False,
        )

        logger.info(f"DataLoader batches: {len(dl)}")
        for i, batch in enumerate(dl):
            patch_size, sample, labels = batch
            logger.info(
                f"Batch {i}: patch_size={patch_size}, "
                f"marsh shape={sample.marsh.shape}, "
                f"timestamps shape={sample.timestamps.shape}, "
                f"labels shape={labels.shape}"
            )
            if i >= 2:
                break
        logger.info("=== Smoke test passed ===")

        logger.info("=== Smoke test: build model and load weights ===")
        model = build_model(
            CommonComponents(
                training_modalities=["marsh"],
                run_name="smoke",
                save_folder="/tmp/smoke",
            ),
            model="nano",
        )
        logger.info(f"Model total params: {sum(p.numel() for p in model.parameters())}")
        weights_path = hf_hub_download(repo_id="allenai/OlmoEarth-v1-Nano", filename="weights.pth")
        missing = load_pretrained_encoder_weights(model, weights_path)
        if missing:
            logger.info(f"Missing keys ({len(missing)}): {missing[:5]} ...")
        model.eval()

        # Grab first batch for forward pass
        dl2 = MarshDataLoader(
            dataset=mock_ds,
            batch_size=4,
            patch_size=MAX_PATCH_SIZE,
            sampled_hw_p=IMAGE_TILE_SIZE // MAX_PATCH_SIZE,
            token_budget=None,
            shuffle=False,
            num_workers=0,
            seed=0,
            pin_memory=False,
        )
        batch = next(iter(dl2))
        patch_size_val, masked_sample, labels = batch

        logger.info(f"\n=== Forward pass (patch_size={patch_size_val}) ===")
        logger.info(f"Input marsh shape: {masked_sample.marsh.shape}")
        logger.info(f"Input marsh_mask shape: {masked_sample.marsh_mask.shape}")
        logger.info(f"Input timestamps shape: {masked_sample.timestamps.shape}")
        logger.info(f"Labels shape: {labels.shape}")

        with torch.no_grad():
            logits = model.forward(masked_sample, patch_size_val)

        logger.info("\n--- Model output ---")
        logger.info(f"  logits shape: {logits.shape}")  # (B, n_classes, H, W)
        logger.info(f"  logits dtype: {logits.dtype}")
        preds = logits.argmax(dim=1)  # (B, H, W)
        logger.info(f"  preds  shape: {preds.shape}")
        logger.info(f"  preds unique: {preds.unique().tolist()}")

        logger.info("=== Forward smoke test passed ===")

        # --- Quick training loop on mock data (CPU, 2 epochs) ---
        logger.info("=== Smoke test: train() on mock data ===")
        model.train()
        train_dl = MarshDataLoader(
            dataset=mock_ds,
            batch_size=4,
            patch_size=MAX_PATCH_SIZE,
            sampled_hw_p=IMAGE_TILE_SIZE // MAX_PATCH_SIZE,
            token_budget=None,
            shuffle=True,
            num_workers=0,
            seed=42,
            pin_memory=False,
        )
        model = train(
            model,
            train_dl,
            epochs=100,
            lr=1e-4,
            encoder_lr=1e-5,
            log_interval=1,
            save_dir="/tmp/smoke_checkpoints",
            save_every=25,
        )
        logger.info("=== Train smoke test passed ===")
    else:
        main()
