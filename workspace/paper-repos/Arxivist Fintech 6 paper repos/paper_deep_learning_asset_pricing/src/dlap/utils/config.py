"""
utils/config.py — Configuration loading and reproducibility utilities.

Implements: seed setting, config loading, panel weighting (Eq. 3 weight sqrt(T_i/T)).
Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019).
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict) -> None:
    """Validate required configuration fields."""
    required_model = ["sdf_num_layers", "sdf_hidden_units", "sdf_macro_states",
                       "cond_num_layers", "cond_hidden_units", "cond_macro_states",
                       "num_firm_chars", "num_macro_vars"]
    for key in required_model:
        if key not in cfg.get("model", {}):
            raise ValueError(f"Missing required model config key: {key}")

    required_training = ["optimizer", "learning_rate", "dropout", "num_ensemble_models"]
    for key in required_training:
        if key not in cfg.get("training", {}):
            raise ValueError(f"Missing required training config key: {key}")

    lr = cfg["training"]["learning_rate"]
    if not (1e-6 <= lr <= 1.0):
        raise ValueError(f"learning_rate {lr} is outside plausible range [1e-6, 1.0]")

    dropout = cfg["training"]["dropout"]
    if not (0.0 <= dropout <= 1.0):
        raise ValueError(f"dropout {dropout} must be in [0, 1]")


def compute_panel_weights(T_i: torch.Tensor, T: int) -> torch.Tensor:
    """
    Compute per-stock panel weights as sqrt(T_i / T).

    From Eq. (3) in Section III.A: weights assigned higher weight to
    moments estimated more precisely (longer time series).

    Args:
        T_i: [N] tensor of observation counts per stock
        T: total time periods

    Returns:
        weights: [N] tensor of sqrt(T_i / T)
    """
    return torch.sqrt(T_i.float() / T)


def get_device(device_str: str) -> torch.device:
    """Resolve device string, falling back to CPU if CUDA unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
