"""
MNIST and Binary MNIST dataset loaders for the VAE experiment.

Paper: Section 5.2 — "trained a variational autoencoder on the MNIST dataset
and on a binarized version of the same dataset, where pixels are thresholded
at 0.5 of their maximum intensity value."

Reference: LeCun & Cortes (2010), Akrami et al. (2022) for binarization.
Confidence: 0.99 (standard dataset, explicitly stated).
"""

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..utils.config import VAEConfig


class BinarizeTransform:
    """Threshold pixel values at 0.5 to produce binary images.

    Paper: Section 5.2 — "pixels are thresholded at 0.5 of their maximum intensity value."
    Akrami et al. (2022). Confidence: 0.99 (explicitly stated).
    """

    def __call__(self, x: Tensor) -> Tensor:
        return (x > 0.5).float()


def get_vae_dataloaders(
    config: VAEConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders for the VAE experiment.

    Downloads MNIST if not already present in config.data.data_dir.
    Applies binarization if config.data.binarize is True.

    Args:
        config: VAEConfig.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_dir = Path(config.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build transform pipeline
    transform_list = [transforms.ToTensor()]
    if config.data.binarize:
        transform_list.append(BinarizeTransform())
    transform = transforms.Compose(transform_list)

    # Download / load MNIST
    train_val_ds = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    # Split train into train + val (use last 10% of training set as val)
    # Paper does not specify an explicit val split for VAE; using standard 50k/10k
    n_total = len(train_val_ds)
    n_val   = n_total // 6   # ~10,000 validation samples
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        train_val_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
