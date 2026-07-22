"""
Config loading and reproducibility utilities.

Implements the config system defined in Stage 3's architecture plan
(config_schema section) for arXiv:2607.18311. All hyperparameters used
anywhere in this repo must flow through this Config object -- no
hardcoded values elsewhere.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class Config:
    """Typed accessor over the flat config.yaml produced by the Architecture Planner.

    Args:
        raw: the raw nested dict loaded from YAML.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load and validate a config YAML file.

        Args:
            path: filesystem path to a config.yaml file.

        Returns:
            A populated Config instance.

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if required top-level sections are missing.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p, "r") as f:
            raw = yaml.safe_load(f)

        required_sections = ["model", "training", "data", "evaluation", "hardware"]
        missing = [s for s in required_sections if s not in raw]
        if missing:
            raise ValueError(
                f"Config at {path} is missing required section(s): {missing}. "
                f"See configs/config.yaml for the expected schema."
            )
        cfg = cls(raw=raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Sanity-check a handful of values that would silently break training."""
        m, t = self.raw["model"], self.raw["training"]
        if m["num_gin_layers"] < 1:
            raise ValueError("model.num_gin_layers must be >= 1")
        if not (0.0 <= t["learning_rate"] < 1.0):
            raise ValueError(f"training.learning_rate looks wrong: {t['learning_rate']}")
        splits = self.raw["data"]["train_val_test_split"]
        if abs(sum(splits) - 1.0) > 1e-6:
            raise ValueError(f"data.train_val_test_split must sum to 1.0, got {splits}")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def __repr__(self) -> str:  # noqa: D105
        return f"Config(sections={list(self.raw.keys())})"


def resolve_device(device_str: str) -> torch.device:
    """Resolve the config's 'cuda_if_available_else_cpu' style device string."""
    if device_str == "cuda_if_available_else_cpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (Sec 4.3: 'fixed seed' train/val/test split).

    Args:
        seed: the integer seed.
        deterministic: if True, forces PyTorch's deterministic algorithms
            (slower, but bit-exact reproducibility). Default False, matching
            the paper's non-specification of full determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
