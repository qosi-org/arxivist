"""
Config loading, seed utilities, and device setup.
Implements reproducibility requirements for DeepVol.
"""
import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    assert cfg.model.num_layers > 0, "num_layers must be > 0"
    assert cfg.model.kernel_size % 2 == 1, "kernel_size must be odd for causal padding"
    assert cfg.training.learning_rate > 0, "learning_rate must be positive"
    assert cfg.data.intervals_per_day > 0, "intervals_per_day must be > 0"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(accelerator: str = "auto") -> torch.device:
    if accelerator == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(accelerator)
