"""Config loading and global seeding utilities.

Reproducibility requirement (ArXivist standard): every entrypoint calls seed_everything()
so Python, NumPy, and PyTorch RNGs are all fixed from one seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices) from a single seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    """Typed wrapper around the parsed config.yaml. Raises ValueError on missing required keys."""

    model: dict
    training: dict
    data: dict
    evaluation: dict
    hardware: dict
    raw: dict = field(repr=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Config file not found: {path}")
        with open(p, "r") as f:
            raw = yaml.safe_load(f)

        for section in ("model", "training", "data", "evaluation", "hardware"):
            if section not in raw:
                raise ValueError(f"Config missing required section: '{section}'")

        return cls(
            model=raw["model"],
            training=raw["training"],
            data=raw["data"],
            evaluation=raw["evaluation"],
            hardware=raw["hardware"],
            raw=raw,
        )
