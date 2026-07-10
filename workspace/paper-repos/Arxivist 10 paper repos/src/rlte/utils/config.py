"""Config loading and reproducibility (seeding) utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    `deterministic=True` additionally forces PyTorch's deterministic
    algorithms, which may slow down training (documented tradeoff).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


@dataclass
class Config:
    """Flat wrapper around the YAML config dict with dot-style access."""
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = Config(raw=raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        required_top = ["model", "training", "data", "evaluation", "hardware"]
        for key in required_top:
            if key not in self.raw:
                raise ValueError(f"Config missing required section: '{key}'")
        if self.raw["model"].get("K", 0) < 2:
            raise ValueError("model.K must be >= 2 (simplex dimension)")
        if self.raw["training"].get("learning_rate", 0) <= 0:
            raise ValueError("training.learning_rate must be > 0")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
