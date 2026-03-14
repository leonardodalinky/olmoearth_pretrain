"""Marsh training dataset for SFT (binary classification, RGB only, NAIP modality).

This dataset is designed to be swappable with ``OlmoEarthDataset`` in the training
pipeline.  It conforms to the same interface: ``__getitem__(GetItemArgs)`` returning
``(patch_size, OlmoEarthSample)``, along with ``training_modalities``, ``fingerprint``,
``fingerprint_version``, ``prepare()``, and ``__len__()``.

The RGB channels are mapped into the NAIP modality whose band order is
[R, G, B, IR].  The first 3 bands are populated with the Marsh RGB data
and the IR band is filled with ``MISSING_VALUE``.  NAIP is
``is_multitemporal=False`` (static), so ``T=1``.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image as PILImage
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.collate import collate_olmoearth_pretrain
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE, MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.transform import Transform, TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.train.masking import MaskingConfig, MaskingStrategy

logger = logging.getLogger(__name__)

# NAIP band order: [R, G, B, IR] — 4 bands total
NAIP_NUM_BANDS = Modality.NAIP.num_bands  # 4

# A default dummy timestamp (day=1, month=0 (zero-indexed January), year=2023)
DEFAULT_DAY_MONTH_YEAR = [1, 0, 2023]


def _get_marsh_ids(data_dir: Path) -> list[str]:
    """Return sorted list of sample IDs from ``data_dir/JPEGImages/``.

    Each sub-directory under ``JPEGImages/`` is treated as one sample,
    e.g. ``JPEGImages/0_0_0/00000.jpg``.
    """
    jpeg_dir = data_dir / "JPEGImages"
    if not jpeg_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages directory not found: {jpeg_dir}")
    ids = sorted(d.name for d in jpeg_dir.iterdir() if d.is_dir())
    if not ids:
        raise FileNotFoundError(f"No sample directories found under {jpeg_dir}")
    return ids


class MarshDataset(Dataset):
    """Marsh binary classification training dataset (RGB → NAIP modality).

    This dataset is a drop-in replacement for ``OlmoEarthDataset``.  It is
    compatible with ``OlmoEarthDataLoader`` and the training pipeline in
    ``experiment.py``.

    Each sample returns ``(patch_size, OlmoEarthSample)`` where the 3 RGB
    channels are placed into the NAIP modality bands ``[R, G, B]`` and the IR
    band is set to ``MISSING_VALUE``.  A dummy timestamp is provided (``T=1``).

    The output ``naip`` field has shape ``(H, W, 1, 4)`` — matching
    OlmoEarth's ``[H, W, T, C]`` layout for static spatial modalities.

    Args:
        data_dir: Root directory of the Marsh dataset.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        training_modalities: Modalities reported to the data loader (default ``["naip"]``).
        dtype: Numpy dtype for output tensors.
        normalize: If True, normalize using pretraining stats.
        dataset_percentage: Fraction of the dataset to use (``0.0``–``1.0``).
        seed: Random seed for dataset-percentage subsampling.
    """

    default_day_month_year = DEFAULT_DAY_MONTH_YEAR

    # Default mock image size and number of samples
    MOCK_HW = 256
    MOCK_N = 64

    def __init__(
        self,
        data_dir: UPath | Path | str,
        split: str = "train",
        training_modalities: list[str] | None = None,
        dtype: np.dtype = np.float32,
        normalize: bool = True,
        dataset_percentage: float = 1.0,
        seed: int = 0,
        mock: bool = False,
        mock_hw: int | None = None,
        mock_n: int | None = None,
        include_ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        target_hw: int = IMAGE_TILE_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.training_modalities: list[str] = training_modalities or ["marsh"]
        self.dtype = dtype
        self.normalize = normalize
        self.dataset_percentage = dataset_percentage
        self.seed = seed
        self.mock = mock
        self.mock_hw = mock_hw or self.MOCK_HW
        self.mock_n = mock_n or self.MOCK_N
        self.include_ids = include_ids
        self.exclude_ids = exclude_ids
        self.target_hw = target_hw

        if self.normalize:
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        # Populated by prepare()
        self.ids: list[str] | None = None  # sample IDs (real data only)
        self.images: np.ndarray | None = None  # bulk storage (mock only)
        self.labels: np.ndarray | None = None  # bulk storage (mock only)
        self.sample_indices: np.ndarray | None = None
        self.latlon_distribution: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties expected by OlmoEarthDataLoader / OlmoEarthConcatDataset
    # ------------------------------------------------------------------

    @property
    def fingerprint_version(self) -> str:
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"marsh,data_dir={self.data_dir},split={self.split},"
            f"dtype={self.dtype},dataset_percentage={self.dataset_percentage}".encode()
        )
        return sha256_hash.hexdigest()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Collect sample IDs and set ``sample_indices`` (called before training)."""
        if self.sample_indices is not None:
            logger.info("MarshDataset is already prepared")
            return

        if self.mock:
            rng = np.random.default_rng(self.seed)
            hw, n = self.mock_hw, self.mock_n
            self.images = rng.random((n, hw, hw, 3), dtype=np.float32) * 255
            self.labels = rng.integers(0, 2, size=(n, hw, hw), dtype=np.int64)
            num_samples = n
            logger.info(f"MarshDataset MOCK mode: {n} samples of {hw}x{hw}")
        else:
            all_ids = _get_marsh_ids(self.data_dir)
            # Apply include / exclude filters
            if self.include_ids is not None:
                include_set = set(self.include_ids)
                all_ids = [sid for sid in all_ids if sid in include_set]
            if self.exclude_ids is not None:
                exclude_set = set(self.exclude_ids)
                all_ids = [sid for sid in all_ids if sid not in exclude_set]
            self.ids = all_ids
            num_samples = len(self.ids)
            logger.info(
                f"MarshDataset found {num_samples} samples "
                f"(include={len(self.include_ids) if self.include_ids else 'all'}, "
                f"exclude={len(self.exclude_ids) if self.exclude_ids else 'none'})"
            )
        logger.info(f"MarshDataset loaded: split={self.split}, samples={num_samples}")

        self.sample_indices = np.arange(num_samples)

        # Apply dataset_percentage sub-sampling
        if self.dataset_percentage < 1.0:
            rng = np.random.default_rng(self.seed)
            self.sample_indices = rng.choice(
                self.sample_indices,
                size=int(num_samples * self.dataset_percentage),
                replace=False,
            )
            logger.info(
                f"Sub-sampled to {len(self.sample_indices)} samples "
                f"({self.dataset_percentage * 100:.1f}%)"
            )

        # Dummy latlon distribution (zeros) — required by some callbacks
        self.latlon_distribution = np.zeros((len(self.sample_indices), 2), dtype=np.float32)

    def __len__(self) -> int:
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared — call prepare() first")
        return len(self.sample_indices)

    # ------------------------------------------------------------------
    # __getitem__  — matches OlmoEarthDataset signature
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Single-sample I/O helpers
    # ------------------------------------------------------------------

    def _load_single(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load one (image, label) pair from disk.

        Returns:
            image: ``(H, W, 3)`` float32, RGB.
            label: ``(H, W)`` int64, 1 = marsh, 0 = background.
        """
        img_path = self.data_dir / "JPEGImages" / sample_id / "00000.jpg"
        lbl_path = self.data_dir / "Annotations" / sample_id / "00000.png"

        with PILImage.open(img_path) as img_file, PILImage.open(lbl_path) as lbl_file:
            assert lbl_file.mode == "P", f"Expected palette PNG for label, got mode {lbl_file.mode}"
            image = np.array(img_file.convert("RGB"), dtype=np.float32)
            # Palette PNG: pixel value 1 → marsh (1), 255 → background (0)
            raw_label = np.array(lbl_file, dtype=np.uint8)
            label = (raw_label < 128).astype(np.int64)
        return image, label

    @staticmethod
    def _ensure_target_size(
        image: np.ndarray,
        label: np.ndarray,
        target_hw: int,
        fix_center: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random-crop (if large enough) or resize to ``target_hw × target_hw``."""
        h, w = image.shape[:2]

        if h >= target_hw and w >= target_hw:
            # Random crop
            if not fix_center:
                start_h = np.random.randint(0, h - target_hw + 1)
                start_w = np.random.randint(0, w - target_hw + 1)
            else:
                start_h = (h - target_hw) // 2
                start_w = (w - target_hw) // 2
            image = image[start_h : start_h + target_hw, start_w : start_w + target_hw]
            label = label[start_h : start_h + target_hw, start_w : start_w + target_hw]
        else:
            # Resize to target_hw × target_hw
            img_pil = PILImage.fromarray(image.astype(np.uint8)).resize(
                (target_hw, target_hw), PILImage.BILINEAR
            )
            image = np.array(img_pil, dtype=np.float32)
            lbl_pil = PILImage.fromarray(label.astype(np.uint8)).resize(
                (target_hw, target_hw), PILImage.NEAREST
            )
            label = np.array(lbl_pil, dtype=np.int64)
        return image, label

    # ------------------------------------------------------------------
    # __getitem__  — matches OlmoEarthDataset signature
    # ------------------------------------------------------------------

    def __getitem__(self, args: GetItemArgs) -> tuple[int, OlmoEarthSample, np.ndarray]:
        """Return ``(patch_size, OlmoEarthSample, label)`` for one Marsh sample.

        Images are lazily loaded from disk (MOSE directory layout) and
        cropped / resized to ``target_hw × target_hw`` (default 256).

        The RGB image is placed into a MARSH tensor of shape
        ``(target_hw, target_hw, 1, 4)`` with IR set to ``MISSING_VALUE``.
        The label is ``(target_hw, target_hw)`` int64.
        """
        assert self.sample_indices is not None, "Dataset is not prepared — call prepare() first"

        real_idx = self.sample_indices[args.idx]  # type: ignore[index]

        # --- Load image & label ---
        if self.mock:
            assert self.images is not None and self.labels is not None
            image = self.images[real_idx]  # (H, W, 3)
            label = self.labels[real_idx]  # (H, W)
        else:
            assert self.ids is not None
            image, label = self._load_single(self.ids[real_idx])

        # --- Crop / resize to target_hw × target_hw ---
        image, label = self._ensure_target_size(
            image, label, self.target_hw, fix_center=self.split != "train"
        )
        # image, label = self._ensure_target_size(image, label, self.target_hw, fix_center=False)
        h, w = image.shape[:2]  # now target_hw × target_hw

        # Build MARSH tensor: (H, W, 4) with IR = MISSING_VALUE
        marsh_data = np.full((h, w, NAIP_NUM_BANDS), MISSING_VALUE, dtype=self.dtype)
        marsh_data[:, :, :3] = image.astype(self.dtype)

        # Add time dimension: (H, W, C) → (H, W, T=1, C)
        marsh_data = np.expand_dims(marsh_data, axis=2)  # (H, W, 1, 4)

        # Optionally normalize (use NAIP stats — marsh shares the same bands)
        if self.normalize:
            missing_mask = marsh_data == MISSING_VALUE
            marsh_data = self.normalizer_computed.normalize(Modality.NAIP, marsh_data).astype(
                self.dtype
            )
            marsh_data = np.where(missing_mask, 0, marsh_data).astype(self.dtype)

        # Dummy timestamp: (T=1, 3)
        timestamps = np.array([self.default_day_month_year], dtype=np.int64)  # (1, 3)

        sample = OlmoEarthSample(
            marsh=marsh_data,
            timestamps=timestamps,
        )

        return args.patch_size, sample, label


# =============================================================================
# Config — mirrors OlmoEarthDatasetConfig so it can be swapped in experiments
# =============================================================================


@dataclass
class MarshDatasetConfig(Config):
    """Configuration for ``MarshDataset``.

    Drop-in replacement for ``OlmoEarthDatasetConfig``, usable in
    ``OlmoEarthExperimentConfig.dataset``.
    """

    data_dir: str = ""
    split: str = "train"
    training_modalities: list[str] | None = None
    dtype: str = "float32"
    normalize: bool = True
    dataset_percentage: float = 1.0
    seed: int = 0
    mock: bool = False
    mock_hw: int | None = None
    mock_n: int | None = None
    include_ids: list[str] | None = None
    exclude_ids: list[str] | None = None
    target_hw: int = IMAGE_TILE_SIZE

    def _get_numpy_dtype(self) -> np.dtype:
        if self.dtype == "float16":
            return np.float16
        elif self.dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def validate(self) -> None:
        if not self.mock and not self.data_dir:
            raise ValueError("data_dir must be set (or enable mock=True)")

    def build(self) -> MarshDataset:
        """Build and return a ``MarshDataset``."""
        self.validate()
        return MarshDataset(
            data_dir=UPath(self.data_dir) if self.data_dir else Path("."),
            split=self.split,
            training_modalities=self.training_modalities,
            dtype=self._get_numpy_dtype(),
            normalize=self.normalize,
            dataset_percentage=self.dataset_percentage,
            seed=self.seed,
            mock=self.mock,
            mock_hw=self.mock_hw,
            mock_n=self.mock_n,
            include_ids=self.include_ids,
            exclude_ids=self.exclude_ids,
            target_hw=self.target_hw,
        )


# =============================================================================
# Collator — extends collate_single_masked_batched with label support
# =============================================================================


def collate_marsh_batched(
    batch: list[tuple[int, OlmoEarthSample, np.ndarray]],
    transform: Transform | None = None,
    masking_strategy: MaskingStrategy | None = None,
) -> tuple[int, MaskedOlmoEarthSample, torch.Tensor]:
    """Collate a Marsh batch: stack samples + labels, apply transform & masking.

    This extends ``collate_single_masked_batched`` to also handle the
    per-pixel segmentation labels returned by ``MarshDataset.__getitem__``.

    Args:
        batch: List of ``(patch_size, OlmoEarthSample, labels)`` tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: Optional masking strategy.  When ``None``, the raw
            ``OlmoEarthSample`` is returned instead of ``MaskedOlmoEarthSample``.

    Returns:
        ``(patch_size, masked_or_raw_sample, labels_tensor)``
    """
    # Separate into (patch_size, sample) pairs and labels
    samples_only = [(ps, sample) for ps, sample, _label in batch]
    labels_list = [label for _ps, _sample, label in batch]

    # Stack labels → (B, H, W)
    labels_tensor = torch.stack([torch.from_numpy(l) for l in labels_list], dim=0)

    # Collate OlmoEarth samples → batched tensors
    patch_size, stacked_sample = collate_olmoearth_pretrain(samples_only)

    # Apply transform
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Apply masking (if provided)
    if masking_strategy is not None:
        masked_sample = masking_strategy.apply_mask(stacked_sample, patch_size)
        return patch_size, masked_sample, labels_tensor
    else:
        # When no masking strategy is provided, create a MaskedOlmoEarthSample
        # with all tokens visible to the online encoder.
        # NOTE: We build manually rather than using from_olmoearthsample()
        # because that method uses compute_expected_shape which omits the batch dim.
        masked_dict: dict[str, torch.Tensor | None] = {}
        for key, val in stacked_sample.as_dict(include_nones=True).items():
            if key == "timestamps":
                masked_dict[key] = val
            elif val is None:
                masked_dict[key] = None
                masked_dict[MaskedOlmoEarthSample.get_masked_modality_name(key)] = None
            else:
                masked_dict[key] = val
                masked_dict[MaskedOlmoEarthSample.get_masked_modality_name(key)] = (
                    torch.ones_like(val) * MaskValue.ONLINE_ENCODER.value
                )
        masked_sample = MaskedOlmoEarthSample(**masked_dict)

    return patch_size, masked_sample, labels_tensor


# =============================================================================
# IterableDataset — simplified _IterableDatasetWrapper (no DDP)
# =============================================================================


class _MarshIterableDataset(torch.utils.data.IterableDataset):
    """Simplified IterableDataset for Marsh training (single-GPU).

    Modelled after ``_IterableDatasetWrapper`` but stripped of DDP / multi-rank
    logic.  Each worker yields pre-collated batches.
    """

    def __init__(self, data_loader: MarshDataLoader) -> None:
        self.data_loader = data_loader

    def __iter__(self):
        dl = self.data_loader
        indices = np.arange(len(dl.dataset))

        if dl.shuffle:
            rng = np.random.default_rng(dl.seed + dl._epoch)
            rng.shuffle(indices)

        # Multi-worker: each worker takes every N-th item (item-level split).
        # Batching, drop_last, and collation are handled by the outer DataLoader.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            indices = indices[worker_info.id :: worker_info.num_workers]

        for idx in indices:
            args = GetItemArgs(
                idx=int(idx),
                patch_size=dl.patch_size,
                sampled_hw_p=dl.sampled_hw_p,
                token_budget=dl.token_budget,
            )
            yield dl.dataset[args]


# =============================================================================
# MarshDataLoader — simplified OlmoEarthDataLoader (no DDP)
# =============================================================================


class MarshDataLoader:
    """Simplified dataloader for Marsh SFT training (single-GPU, no DDP).

    Follows the same ``_iter_batches`` pattern as ``OlmoEarthDataLoader`` but
    without distributed training complexity.  Internally wraps a
    ``_MarshIterableDataset`` that yields pre-collated batches.

    Args:
        dataset: A prepared ``MarshDataset``.
        batch_size: Number of samples per batch.
        patch_size: Patch size passed to ``GetItemArgs`` / ``FlexiPatchEmbed``.
        sampled_hw_p: Spatial size in patch units
            (``sampled_hw_p * patch_size`` = pixel crop size).
        token_budget: If set, spatial cropping is applied in ``__getitem__``.
        shuffle: Shuffle indices each epoch.
        num_workers: ``torch.utils.data.DataLoader`` workers.
        seed: Base random seed.
        drop_last: Drop the last incomplete batch.
        transform: Optional ``Transform`` applied in the collator.
        masking_strategy: Optional ``MaskingStrategy`` applied in the collator.
        pin_memory: Pin CUDA memory in the underlying DataLoader.
    """

    def __init__(
        self,
        dataset: MarshDataset,
        batch_size: int,
        patch_size: int,
        sampled_hw_p: int,
        token_budget: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        transform: Transform | None = None,
        masking_strategy: MaskingStrategy | None = None,
        pin_memory: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.sampled_hw_p = sampled_hw_p
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self._epoch: int = 0

        self.collator = functools.partial(
            collate_marsh_batched,
            transform=transform,
            masking_strategy=masking_strategy,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch (affects shuffle RNG seed)."""
        self._epoch = epoch

    def _iter_batches(self) -> torch.utils.data.DataLoader:
        """Build a ``torch.utils.data.DataLoader`` over ``_MarshIterableDataset``."""
        return torch.utils.data.DataLoader(
            _MarshIterableDataset(self),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def __iter__(self):
        return iter(self._iter_batches())

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# =============================================================================
# MarshDataLoaderConfig
# =============================================================================


@dataclass
class MarshDataLoaderConfig(Config):
    """Configuration for ``MarshDataLoader``."""

    batch_size: int = 16
    patch_size: int = 8
    sampled_hw_p: int = 128  # 128 * 8 = 1024 px → full image
    token_budget: int | None = None
    shuffle: bool = True
    num_workers: int = 0
    seed: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    transform_config: TransformConfig | None = None
    masking_config: MaskingConfig | None = None

    def build(self, dataset: MarshDataset) -> MarshDataLoader:
        """Build and return a ``MarshDataLoader``."""
        dataset.prepare()
        transform = self.transform_config.build() if self.transform_config else None
        masking_strategy = self.masking_config.build() if self.masking_config else None
        return MarshDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            sampled_hw_p=self.sampled_hw_p,
            token_budget=self.token_budget,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            seed=self.seed,
            drop_last=self.drop_last,
            transform=transform,
            masking_strategy=masking_strategy,
            pin_memory=self.pin_memory,
        )
