"""
Config loading and reproducibility utilities.

Implements ArXivist's non-negotiable reproducibility requirements: a single seeding
utility that seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA), plus a
deterministic-mode flag, and a config loader that validates required fields are
present before returning.

Paper reference: "Asset Pricing in Pre-trained Transformers" (arXiv:2505.01575),
Section 4.1.5 / Appendix A (Adam optimizer + training loop).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml


REQUIRED_TOP_LEVEL_KEYS = ("model", "training", "data", "evaluation", "hardware")
VALID_MODEL_FAMILIES = (
    "pretrained_transformer",
    "pretrained_transformer_lnf",
    "standard_transformer",
    "sert",
    "sert_lnf",
    "encoder_only_transformer",
)


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or has invalid values."""


@dataclass
class ConfigLoader:
    """Loads and validates the YAML config used by all ArXivist entrypoints.

    Args:
        path: filesystem path to a config YAML file.
    """

    path: str

    def load(self) -> dict[str, Any]:
        """Parse the YAML config at ``self.path`` and validate required fields.

        Returns:
            The parsed config dictionary.

        Raises:
            ConfigError: if a required top-level key is missing or a value is invalid.
        """
        if not os.path.exists(self.path):
            raise ConfigError(f"Config file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for key in REQUIRED_TOP_LEVEL_KEYS:
            if key not in cfg:
                raise ConfigError(f"Config missing required top-level section: '{key}'")

        family = cfg["model"].get("family")
        if family not in VALID_MODEL_FAMILIES:
            raise ConfigError(
                f"model.family='{family}' is invalid. Must be one of {VALID_MODEL_FAMILIES}"
            )

        if cfg["model"]["d_model"] % cfg["model"]["num_heads"] != 0:
            raise ConfigError(
                f"model.d_model ({cfg['model']['d_model']}) must be divisible by "
                f"model.num_heads ({cfg['model']['num_heads']})"
            )

        return cfg

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self.path!r})"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices) for reproducibility.

    Args:
        seed: the random seed to use everywhere.
        deterministic: if True, forces CuDNN deterministic algorithms. This can
            slow down training but is required for bit-for-bit reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(preference: str = "cuda_if_available_else_cpu") -> torch.device:
    """Resolve the hardware.device config string into a torch.device.

    Args:
        preference: one of "cuda_if_available_else_cpu", "cpu", "cuda".

    Returns:
        A torch.device instance.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("hardware.device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    # default: cuda_if_available_else_cpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
