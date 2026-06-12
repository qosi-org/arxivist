"""
gmlp/data/imagenet_dataset.py
-----------------------------
ImageNet-1K dataset with vision augmentation pipeline for gMLP training.

Paper Section 3 + Appendix A.1: "Pay Attention to MLPs" (arXiv:2105.08050)

Augmentation recipe (Table 7):
  - AutoAugment (ImageNet policy)
  - Mixup (α=0.8)
  - CutMix (α=1.0)
  - CutMix/Mixup switch probability: 0.5
  - Label smoothing: 0.1
  - Repeated augmentation: OFF (unlike DeiT)
  - Random erasing probability: 0
  - Input resolution: 224×224

Differences from DeiT: no repeated augmentation, no random erasing.
Stochastic depth controlled in model (not in data pipeline).

Paper ref: Section 3, Appendix A.1, SIR training_pipeline.vision_pretraining
"""

from __future__ import annotations

import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


def build_train_transforms(img_size: int = 224, autoaugment: bool = True) -> T.Compose:
    """
    Training augmentation pipeline matching paper Appendix A.1.
    Note: Mixup and CutMix are applied at the batch level (see MixupCutmixCollator),
    not here at the sample level.
    """
    transforms = [
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
    ]
    if autoaugment:
        # AutoAugment with ImageNet policy (paper Table 7)
        transforms.append(T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
    transforms += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def build_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImageNetDataset(Dataset):
    """
    ImageNet-1K classification dataset via torchvision ImageFolder.

    Args:
        data_dir:    Path to ImageNet root. Expected structure:
                     data_dir/train/{class_name}/*.JPEG
                     data_dir/val/{class_name}/*.JPEG
        split:       'train' or 'val'.
        img_size:    Target resolution. Paper: 224.
        autoaugment: Apply AutoAugment (paper: True).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        autoaugment: bool = True,
    ) -> None:
        import os
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"ImageNet split directory not found: {split_dir}\n"
                f"See data/README_data.md for download instructions."
            )

        transforms = (
            build_train_transforms(img_size, autoaugment)
            if split == "train"
            else build_val_transforms(img_size)
        )
        self.dataset = ImageFolder(split_dir, transform=transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


# ---------------------------------------------------------------------------
# Mixup + CutMix collator (applied at batch level per paper Table 7)
# ---------------------------------------------------------------------------

class MixupCutmixCollator:
    """
    Batch-level collator that applies Mixup and/or CutMix augmentation.

    Paper Table 7:
      - Mixup alpha: 0.8
      - CutMix alpha: 1.0
      - Switch probability: 0.5  (prob of choosing CutMix over Mixup)

    When neither is active (p < 0 threshold), batch is returned unchanged.
    Label smoothing is handled in the loss function (not here).

    Args:
        mixup_alpha:   Beta distribution α for Mixup. Paper: 0.8.
        cutmix_alpha:  Beta distribution α for CutMix. Paper: 1.0.
        switch_prob:   P(CutMix) vs P(Mixup). Paper: 0.5.
        num_classes:   Number of output classes (1000 for ImageNet).
        label_smoothing: Applied to soft labels. Paper: 0.1.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = 1000,
        label_smoothing: float = 0.1,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)   # [B, 3, H, W]
        labels = torch.tensor(labels, dtype=torch.long)

        # One-hot encode with label smoothing
        smooth = self.label_smoothing / self.num_classes
        soft_labels = torch.full((len(labels), self.num_classes), smooth)
        soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing + smooth)

        # Choose augmentation
        use_cutmix = random.random() < self.switch_prob
        if use_cutmix and self.cutmix_alpha > 0:
            images, soft_labels = self._cutmix(images, soft_labels)
        elif self.mixup_alpha > 0:
            images, soft_labels = self._mixup(images, soft_labels)

        return {"pixel_values": images, "labels": soft_labels}

    def _mixup(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Mixup augmentation: linearly interpolate two random samples."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        B = images.shape[0]
        idx = torch.randperm(B)
        images = lam * images + (1 - lam) * images[idx]
        labels = lam * labels + (1 - lam) * labels[idx]
        return images, labels

    def _cutmix(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """CutMix augmentation: paste a rectangular crop from another sample."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        B, C, H, W = images.shape
        idx = torch.randperm(B)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_actual = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        labels = lam_actual * labels + (1 - lam_actual) * labels[idx]
        return images, labels
