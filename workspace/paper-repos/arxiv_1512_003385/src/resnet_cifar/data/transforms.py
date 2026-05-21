"""CIFAR-10 transforms — paper Sec. 4.2.

Paper recipe:
  - Per-pixel mean subtraction (computed on training set).
  - 4 pixels padded on each side; random 32x32 crop sampled from padded image or hflip thereof.
  - Random horizontal flip.
  - Test: single 32x32 view, no augmentation.

`mean_subtraction='per_channel'` is offered as a documented alternative for the per-pixel/per-channel
ambiguity (SIR ambiguities[0], confidence 0.75).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torchvision import transforms


class PerPixelMeanSubtract:
    """Subtract a (3, 32, 32) mean tensor from every image. Paper-style normalization."""

    def __init__(self, mean: torch.Tensor) -> None:
        if mean.shape != (3, 32, 32):
            raise ValueError(f"PerPixelMeanSubtract expects mean of shape (3,32,32), got {tuple(mean.shape)}")
        self.mean = mean.clone()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img - self.mean


class PerChannelMeanSubtract:
    """Subtract (R,G,B) per-channel mean. Alternative interpretation of the paper."""

    def __init__(self, mean: torch.Tensor) -> None:
        if mean.numel() != 3:
            raise ValueError(f"PerChannelMeanSubtract expects 3 channel means, got numel={mean.numel()}")
        self.mean = mean.view(3, 1, 1).clone()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img - self.mean


def compute_train_mean(train_images_uint8: np.ndarray, mode: str) -> torch.Tensor:
    """Compute the mean image (or per-channel mean) from a (N,32,32,3) uint8 array, returning
    a float32 tensor scaled to [0,1] convention (i.e., divided by 255 so it matches ToTensor)."""
    if train_images_uint8.dtype != np.uint8:
        raise ValueError(f"expected uint8 images, got dtype={train_images_uint8.dtype}")
    if train_images_uint8.ndim != 4 or train_images_uint8.shape[1:] != (32, 32, 3):
        raise ValueError(f"expected (N,32,32,3), got {train_images_uint8.shape}")

    float_images = train_images_uint8.astype(np.float32) / 255.0  # match ToTensor scaling
    if mode == "per_pixel":
        mean = float_images.mean(axis=0)              # (32,32,3)
        mean_chw = np.transpose(mean, (2, 0, 1))      # (3,32,32)
        return torch.from_numpy(mean_chw).contiguous()
    if mode == "per_channel":
        mean = float_images.mean(axis=(0, 1, 2))      # (3,)
        return torch.from_numpy(mean).contiguous()
    raise ValueError(f"mode must be 'per_pixel' or 'per_channel', got {mode!r}")


def build_train_transform(mean: torch.Tensor, mode: str) -> Callable:
    """Compose paper-specified training augmentations."""
    if mode == "per_pixel":
        subtract = PerPixelMeanSubtract(mean)
    elif mode == "per_channel":
        subtract = PerChannelMeanSubtract(mean)
    else:
        raise ValueError(f"mode must be 'per_pixel' or 'per_channel', got {mode!r}")

    # Paper: 4-pixel pad with zeros, random 32x32 crop, random hflip.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="constant"),  # pad with zeros
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        subtract,
    ])


def build_eval_transform(mean: torch.Tensor, mode: str) -> Callable:
    """Eval-time transform: only ToTensor + mean subtraction (paper: single 32x32 view)."""
    if mode == "per_pixel":
        subtract = PerPixelMeanSubtract(mean)
    elif mode == "per_channel":
        subtract = PerChannelMeanSubtract(mean)
    else:
        raise ValueError(f"mode must be 'per_pixel' or 'per_channel', got {mode!r}")

    return transforms.Compose([
        transforms.ToTensor(),
        subtract,
    ])
