"""
forecast_risk.utils.config
============================
Configuration loading, validation, and reproducibility utilities.
"""

from __future__ import annotations

import os
import random
import numpy as np
import yaml
from pathlib import Path
from typing import Any


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for full reproducibility.

    Seeds Python, NumPy, and optionally PyTorch.

    Args:
        seed:          Integer seed value.
        deterministic: If True, enables PyTorch deterministic mode
                       (may slow GPU operations significantly).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not available; OK for metrics-only mode


class Config:
    """
    Flat configuration object loaded from a YAML file.

    All paths in the config are relative to the project root by default.
    """

    def __init__(self, data: dict):
        self._data = data

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        if item in self._data:
            val = self._data[item]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"Config has no key '{item}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return dict(self._data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If required top-level keys are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Config file is empty: {path}")

        _validate_config(data)
        return cls(data)


def _validate_config(data: dict) -> None:
    """Validate required top-level config sections."""
    required = ["data", "evaluation", "models", "metrics", "output"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Config missing required sections: {missing}. "
            f"See configs/default_config.yaml for reference."
        )

    # Validate horizons
    horizons = data["evaluation"].get("horizons", [])
    if not horizons:
        raise ValueError("evaluation.horizons must be a non-empty list")

    # Validate loss function
    valid_losses = {"squared_error", "absolute_error", "log_score"}
    loss_fn = data["evaluation"].get("loss_function", "squared_error")
    if loss_fn not in valid_losses:
        raise ValueError(
            f"evaluation.loss_function must be one of {valid_losses}, got '{loss_fn}'"
        )
