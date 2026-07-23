"""Config loading and global seeding utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        required = ["simulation", "heston", "ou", "rbergomi", "signature", "xgboost"]
        missing = [s for s in required if s not in raw]
        if missing:
            raise ValueError(f"Config missing required section(s): {missing}")
        return cls(raw=raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def apply_debug_overrides(self) -> "Config":
        import copy

        dbg = copy.deepcopy(self.raw)
        dbg["simulation"]["n_paths_per_class"] = 500
        dbg["simulation"]["n_test_per_class"] = 200
        dbg["simulation"]["n_steps"] = 20
        return Config(raw=dbg)
