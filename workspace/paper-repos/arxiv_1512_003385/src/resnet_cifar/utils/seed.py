"""Deterministic seeding across Python, NumPy, and PyTorch.

Used by all entrypoints (train.py, evaluate.py, inference.py) to ensure runs are reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python's `random`, NumPy, and all PyTorch RNGs.

    Args:
        seed: integer seed value.
        deterministic: when True, also forces cuDNN into deterministic mode. This can
            slow training noticeably but is necessary for bit-exact reproduction.
    """
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
