"""
utils/config.py
================
Configuration loading, validation, and reproducibility (seeding) utilities.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or has invalid values."""


@dataclass
class Config:
    """Typed wrapper around the parsed config.yaml."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def deterministic(self) -> bool:
        return bool(self.raw.get("deterministic", True))

    def section(self, *keys: str) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise ConfigError(f"Missing config section: {'.'.join(keys)}")
            node = node[k]
        return node

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            raise ConfigError(f"Config file at {path} is empty or invalid YAML.")
        cfg = Config(raw=raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        for key in ["model", "data", "evaluation", "training"]:
            if key not in self.raw:
                raise ConfigError(f"config.yaml is missing required top-level section: '{key}'")

    def set_seed(self, seed: int | None = None) -> None:
        """Seed Python's random, NumPy, and torch (CPU + CUDA if available)."""
        s = self.seed if seed is None else seed
        random.seed(s)
        np.random.seed(s)
        try:
            import torch

            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:  # pragma: no cover
            pass

    def device(self) -> str:
        """Resolve hardware.device, falling back to cpu if cuda is requested but unavailable."""
        requested = self.raw.get("hardware", {}).get("device", "cpu")
        if requested == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                print("[Config] hardware.device='cuda' requested but no GPU available; falling back to 'cpu'.")
                return "cpu"
            except ImportError:  # pragma: no cover
                return "cpu"
        return requested

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(seed={self.seed}, keys={list(self.raw.keys())})"
