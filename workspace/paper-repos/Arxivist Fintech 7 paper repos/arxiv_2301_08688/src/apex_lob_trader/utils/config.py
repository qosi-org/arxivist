"""
utils/config.py — Configuration loading and reproducibility utilities.
Implements seed management required for reproducibility (Section 5).
Paper: arXiv:2301.08688
"""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML config file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If required keys are missing.
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate required config sections exist."""
    required_sections = ["env", "signal", "model", "training", "evaluation"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(
                f"Config missing required section '{section}'. "
                f"Check configs/config.yaml."
            )

    pos_min = cfg["env"]["inventory"]["pos_min"]
    pos_max = cfg["env"]["inventory"]["pos_max"]
    if pos_min >= pos_max:
        raise ValueError(
            f"inventory.pos_min ({pos_min}) must be < pos_max ({pos_max})"
        )

    noise_a = cfg["signal"]["noise_level_a"]
    if noise_a not in [1.1, 1.3, 1.6]:
        print(
            f"[WARNING] signal.noise_level_a={noise_a} is non-standard. "
            "Paper uses 1.1 (high noise), 1.3 (mid), 1.6 (low)."
        )


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer random seed.
        deterministic: If True, sets PyTorch deterministic mode
            (may reduce performance). See Section 5 training setup.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print(f"[Reproducibility] Deterministic mode ON (seed={seed}). Training may be slower.")
    else:
        print(f"[Reproducibility] Seed set to {seed}. Deterministic mode OFF.")


def get_device(cfg: dict[str, Any]) -> torch.device:
    """Resolve torch device from config with CPU fallback.

    Args:
        cfg: Full config dict.

    Returns:
        torch.device
    """
    requested = cfg.get("hardware", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
