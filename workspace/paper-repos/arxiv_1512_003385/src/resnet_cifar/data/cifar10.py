"""CIFAR-10 DataModule with paper-spec augmentations.

Uses torchvision.datasets.CIFAR10 for download and on-disk storage, and applies the
paper's transforms (per-pixel mean subtraction, 4-pixel pad + random crop, random hflip).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from resnet_cifar.data.transforms import (
    build_eval_transform,
    build_train_transform,
    compute_train_mean,
)


class CIFAR10DataModule:
    """Bundles train/val/test CIFAR-10 loaders with paper-spec transforms.

    Args:
        data_dir: where to download / find CIFAR-10.
        batch_size: per-step batch size (paper: 128).
        num_workers: DataLoader workers.
        val_size: number of training samples to hold out as a validation set. 0 = no val split.
        mean_subtraction: 'per_pixel' (paper wording) or 'per_channel' (alt).
        download: if True, allow torchvision to download CIFAR-10 to data_dir.
        seed: seed for the train/val split.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        num_workers: int = 2,
        val_size: int = 0,
        mean_subtraction: str = "per_pixel",
        download: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.mean_subtraction = mean_subtraction
        self.download = download
        self.seed = seed

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load raw uint8 arrays once to compute the training mean.
        base_train = CIFAR10(root=str(self.data_dir), train=True, download=download, transform=None)
        train_images_uint8 = base_train.data  # (50000, 32, 32, 3) uint8
        self.mean = compute_train_mean(train_images_uint8, mode=mean_subtraction)

        self._train_tf = build_train_transform(self.mean, mode=mean_subtraction)
        self._eval_tf = build_eval_transform(self.mean, mode=mean_subtraction)

        full_train = CIFAR10(root=str(self.data_dir), train=True, download=False, transform=self._train_tf)
        full_train_eval = CIFAR10(root=str(self.data_dir), train=True, download=False, transform=self._eval_tf)
        self._test = CIFAR10(root=str(self.data_dir), train=True, download=False)  # placeholder
        self._test = CIFAR10(root=str(self.data_dir), train=False, download=False, transform=self._eval_tf)

        if val_size > 0:
            n = len(full_train)
            if val_size >= n:
                raise ValueError(f"val_size={val_size} must be < train size {n}")
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n)
            val_indices = indices[:val_size].tolist()
            train_indices = indices[val_size:].tolist()
            self._train = Subset(full_train, train_indices)
            self._val = Subset(full_train_eval, val_indices)
        else:
            self._train = full_train
            self._val = None

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_loader(self) -> Optional[DataLoader]:
        if self._val is None:
            return None
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def mean_tensor(self) -> torch.Tensor:
        """The mean used for normalization (handy for inference scripts)."""
        return self.mean
