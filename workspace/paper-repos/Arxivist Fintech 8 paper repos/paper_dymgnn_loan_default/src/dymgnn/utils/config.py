"""
utils/config.py — Configuration loading and reproducibility utilities.

Paper: "Attention-based dynamic multilayer graph neural networks for loan default prediction"
Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If required sections are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate required config sections."""
    required = ["model", "training", "data", "evaluation"]
    for section in required:
        if section not in cfg:
            raise ValueError(
                f"Config missing required section '{section}'. Check configs/config.yaml."
            )

    valid_gnn = ["GCN", "GAT"]
    valid_rnn = ["LSTM", "GRU"]
    valid_net = ["area", "company", "double"]

    gnn = cfg["model"].get("gnn_type", "")
    if gnn not in valid_gnn:
        raise ValueError(f"model.gnn_type must be one of {valid_gnn}, got '{gnn}'")

    rnn = cfg["model"].get("rnn_type", "")
    if rnn not in valid_rnn:
        raise ValueError(f"model.rnn_type must be one of {valid_rnn}, got '{rnn}'")

    net = cfg["model"].get("network_type", "")
    if net not in valid_net:
        raise ValueError(f"model.network_type must be one of {valid_net}, got '{net}'")


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python random, NumPy, and PyTorch. Table C.1 uses Adam optimizer;
    this function ensures reproducible weight initialization and data shuffling.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables PyTorch deterministic mode (slower).
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
        print(f"[Reproducibility] Deterministic mode ON (seed={seed}).")
    else:
        print(f"[Reproducibility] Seed set to {seed}.")


def get_device(cfg: dict[str, Any]) -> torch.device:
    """Resolve torch device from config with CPU fallback.

    Paper uses NVidia A100 (Table C.2).

    Args:
        cfg: Full config dict.

    Returns:
        torch.device
    """
    requested = cfg.get("hardware", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
