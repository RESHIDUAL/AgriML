"""
dataset.py
==========
PyTorch Dataset classes for PlantVillage and PlantDoc leaf-disease datasets.

Provides:
- Albumentations-based train/val transforms (with version-safe CoarseDropout).
- LeafDiseaseDataset: a generic Dataset that maps raw folder names to unified
  labels via the dictionaries exported by class_mapping.py.
- Helper factories for PlantVillage and PlantDoc dataset pairs.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

try:
    from src.disease.class_mapping import (
        PD_CLASS_TO_IDX,
        PD_TO_UNIFIED,
        PV_CLASS_TO_IDX,
        PV_TO_UNIFIED,
    )
except ImportError:
    from class_mapping import (
        PD_CLASS_TO_IDX,
        PD_TO_UNIFIED,
        PV_CLASS_TO_IDX,
        PV_TO_UNIFIED,
    )

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _make_coarse_dropout(p: float = 0.3):
    """Return a CoarseDropout transform that works across albumentations versions.

    albumentations v2.x changed the API to range-based parameters
    (``num_holes_range``, ``hole_height_range``, ``hole_width_range``),
    while older versions use ``max_holes``, ``max_height``, ``max_width``.
    We try the new API first, then fall back, and silently skip if neither
    works.
    """
    # Try new (v2.x) API first
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            p=p,
        )
    except TypeError:
        pass

    # Fall back to legacy API
    try:
        return A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            p=p,
        )
    except TypeError:
        pass

    # If both fail, return None so caller can skip it
    return None


def _make_random_resized_crop(img_size: int, scale=(0.7, 1.0)):
    try:
        return A.RandomResizedCrop(size=(img_size, img_size), scale=scale)
    except Exception:
        return A.RandomResizedCrop(height=img_size, width=img_size, scale=scale)


def _make_resize(height: int, width: int):
    try:
        return A.Resize(height=height, width=width)
    except Exception:
        return A.Resize(size=(height, width))


def _make_center_crop(height: int, width: int):
    try:
        return A.CenterCrop(height=height, width=width)
    except Exception:
        return A.CenterCrop(size=(height, width))


def get_transforms(is_train: bool, img_size: int = 224) -> A.Compose:
    """Build an albumentations ``Compose`` pipeline.

    Parameters
    ----------
    is_train : bool
        If ``True``, return an augmented training pipeline.
        Otherwise return a deterministic validation / test pipeline.
    img_size : int, optional
        Spatial size of the output tensor (default ``224``).

    Returns
    -------
    A.Compose
    """
    if is_train:
        coarse = _make_coarse_dropout(p=0.3)
        transforms: list = [
            _make_random_resized_crop(img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
            ),
        ]
        if coarse is not None:
            transforms.append(coarse)
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return A.Compose(transforms)

    # Validation / Test
    return A.Compose([
        _make_resize(256, 256),
        _make_center_crop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LeafDiseaseDataset(Dataset):
    """Generic leaf-disease dataset backed by a folder-per-class layout.

    Parameters
    ----------
    root_dir : str
        Path to the split folder (e.g. ``PlantVillage/train``).
    label_map : dict
        Mapping from raw folder name → unified label string
        (``PV_TO_UNIFIED`` or ``PD_TO_UNIFIED``).
    class_to_idx : dict
        Mapping from unified label string → integer index
        (``PV_CLASS_TO_IDX`` or ``PD_CLASS_TO_IDX``).
    transform : albumentations.Compose, optional
        Albumentations transform pipeline to apply to each image.
    """

    def __init__(
        self,
        root_dir: str,
        label_map: Dict[str, str],
        class_to_idx: Dict[str, int],
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.num_classes = len(class_to_idx)

        self.samples: List[Tuple[str, int]] = []
        self._scan_directory()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_directory(self) -> None:
        """Walk *root_dir* and collect ``(image_path, label_idx)`` tuples."""
        if not self.root_dir.is_dir():
            print(f"[WARNING] Root directory does not exist: {self.root_dir}")
            return

        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue
            folder_name = subdir.name
            if folder_name not in self.label_map:
                print(
                    f"[WARNING] Skipping unknown subdirectory: '{folder_name}' "
                    f"(not in label_map)"
                )
                continue

            unified_label = self.label_map[folder_name]
            label_idx = self.class_to_idx[unified_label]

            for img_file in sorted(subdir.iterdir()):
                if img_file.suffix.lower() in _IMG_EXTENSIONS:
                    self.samples.append((str(img_file), label_idx))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load an image, apply transforms, and return ``(tensor, label_idx)``.

        Corrupted / unreadable images are handled gracefully by falling
        through to the next valid sample.
        """
        for offset in range(len(self.samples)):
            actual_idx = (idx + offset) % len(self.samples)
            img_path, label_idx = self.samples[actual_idx]
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise IOError(f"cv2.imread returned None for {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.transform is not None:
                    augmented = self.transform(image=img)
                    img = augmented["image"]
                else:
                    # Fallback: simple HWC uint8 → CHW float tensor
                    img = torch.from_numpy(
                        img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    )

                return img, label_idx

            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARNING] Could not load image {img_path}: {exc}. "
                    f"Skipping to next sample."
                )

        # Should never happen unless the entire dataset is corrupted
        raise RuntimeError("All samples in the dataset are corrupted.")

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights.

        Returns a 1-D ``float32`` tensor of length ``num_classes`` where
        ``weight[c] = total_samples / (num_classes * count[c])``.
        Classes with zero samples receive a weight of ``0.0``.
        """
        counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_idx, count in counts.items():
            if count > 0:
                weights[cls_idx] = total / (self.num_classes * count)
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Return per-sample weights suitable for ``WeightedRandomSampler``.

        Each sample's weight equals the inverse-frequency weight of its
        class, so that under-represented classes are sampled more often.
        """
        class_weights = self.get_class_weights()
        return torch.tensor(
            [class_weights[label].item() for _, label in self.samples],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_plantvillage_datasets(
    data_dir: str = "PlantVillage",
    img_size: int = 224,
) -> Tuple[LeafDiseaseDataset, LeafDiseaseDataset]:
    """Create train and validation ``LeafDiseaseDataset`` instances for PlantVillage.

    Parameters
    ----------
    data_dir : str
        Root folder containing ``train/`` and ``val/`` subdirectories.
    img_size : int
        Spatial size passed to ``get_transforms``.

    Returns
    -------
    tuple[LeafDiseaseDataset, LeafDiseaseDataset]
        ``(train_dataset, val_dataset)``
    """
    train_ds = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, "train"),
        label_map=PV_TO_UNIFIED,
        class_to_idx=PV_CLASS_TO_IDX,
        transform=get_transforms(is_train=True, img_size=img_size),
    )
    val_ds = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, "val"),
        label_map=PV_TO_UNIFIED,
        class_to_idx=PV_CLASS_TO_IDX,
        transform=get_transforms(is_train=False, img_size=img_size),
    )
    return train_ds, val_ds


def create_plantdoc_datasets(
    data_dir: str = "PlantDoc-Dataset",
    img_size: int = 224,
) -> Tuple[LeafDiseaseDataset, LeafDiseaseDataset]:
    """Create train and test ``LeafDiseaseDataset`` instances for PlantDoc.

    Parameters
    ----------
    data_dir : str
        Root folder containing ``train/`` and ``test/`` subdirectories.
    img_size : int
        Spatial size passed to ``get_transforms``.

    Returns
    -------
    tuple[LeafDiseaseDataset, LeafDiseaseDataset]
        ``(train_dataset, test_dataset)``
    """
    train_ds = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, "train"),
        label_map=PD_TO_UNIFIED,
        class_to_idx=PD_CLASS_TO_IDX,
        transform=get_transforms(is_train=True, img_size=img_size),
    )
    test_ds = LeafDiseaseDataset(
        root_dir=os.path.join(data_dir, "test"),
        label_map=PD_TO_UNIFIED,
        class_to_idx=PD_CLASS_TO_IDX,
        transform=get_transforms(is_train=False, img_size=img_size),
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Main - quick sanity check
# ---------------------------------------------------------------------------

def _print_distribution(dataset: LeafDiseaseDataset, name: str) -> None:
    """Pretty-print the class distribution of a dataset split."""
    counts = Counter(label for _, label in dataset.samples)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    print(f"\n{'=' * 60}")
    print(f"  {name}  -  {len(dataset)} total samples")
    print(f"{'=' * 60}")
    for idx in sorted(counts):
        cls_name = idx_to_class.get(idx, f"<unknown-{idx}>")
        print(f"  [{idx:3d}] {cls_name:45s}  {counts[idx]:5d}")
    print()


if __name__ == "__main__":
    # ---- PlantVillage --------------------------------------------------
    print("\n>>> Creating PlantVillage datasets …")
    pv_train, pv_val = create_plantvillage_datasets()
    _print_distribution(pv_train, "PlantVillage / train")
    _print_distribution(pv_val, "PlantVillage / val")

    if len(pv_train) > 0:
        img, label = pv_train[0]
        print(f"  Sample image shape : {img.shape}")
        print(f"  Sample label index : {label}")

    # ---- PlantDoc ------------------------------------------------------
    print("\n>>> Creating PlantDoc datasets …")
    pd_train, pd_test = create_plantdoc_datasets()
    _print_distribution(pd_train, "PlantDoc / train")
    _print_distribution(pd_test, "PlantDoc / test")

    if len(pd_train) > 0:
        img, label = pd_train[0]
        print(f"  Sample image shape : {img.shape}")
        print(f"  Sample label index : {label}")

    # ---- Summary -------------------------------------------------------
    print("\n>>> Summary")
    print(f"  PlantVillage  train={len(pv_train):,}  val={len(pv_val):,}")
    print(f"  PlantDoc      train={len(pd_train):,}  test={len(pd_test):,}")
