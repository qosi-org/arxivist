"""
Reproducibility utilities: seed setting, device selection.

Ensures deterministic behaviour across Python, NumPy, and PyTorch.
Called at the start of every entrypoint (train.py, evaluate.py, scripts/).
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for full reproducibility.

    Seeds Python's random, NumPy, and PyTorch (both CPU and CUDA).
    Optionally enables PyTorch's deterministic algorithm mode, which can
    slow training but guarantees bitwise reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables torch.use_deterministic_algorithms(True).
            See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # benchmark=True can speed up fixed-size inputs
        torch.backends.cudnn.benchmark = True


def get_device(device_str: Optional[str] = "cuda") -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA is unavailable.

    Args:
        device_str: "cuda" or "cpu". If "cuda" but no GPU is available,
            falls back to CPU with a warning.

    Returns:
        torch.device
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str or "cpu")
