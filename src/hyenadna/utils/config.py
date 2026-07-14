"""Config loading and reproducibility utilities.

Implements the seed-setting requirement (SIR reproducibility) and validates
config values at load time. No hardcoded paths — everything flows from YAML.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: the random seed.
        deterministic: if True, force deterministic cuDNN (may slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: deterministic mode can noticeably slow training.


def resolve_device(device: str) -> torch.device:
    """Resolve an 'auto'|'cuda'|'cpu' string to a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available.")
    return torch.device(device)


@dataclass
class Config:
    """Typed view over the YAML config."""

    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Config(model.variant={self.model.get('variant')}, "
            f"data.dataset={self.data.get('dataset')}, "
            f"epochs={self.training.get('epochs')})"
        )


def _require(d: Dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise ValueError(f"config.{section} is missing required key '{key}'")
    return d[key]


def load_config(path: str) -> Config:
    """Load and validate a YAML config file.

    Raises ValueError with helpful messages if required keys are missing or
    values are out of range.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        model=raw.get("model", {}),
        training=raw.get("training", {}),
        data=raw.get("data", {}),
        evaluation=raw.get("evaluation", {}),
        hardware=raw.get("hardware", {}),
    )

    # --- validation ---
    _require(cfg.model, "variant", "model")
    _require(cfg.model, "num_classes", "model")
    _require(cfg.data, "dataset", "data")
    _require(cfg.data, "source", "data")

    if cfg.model["num_classes"] < 2:
        raise ValueError("model.num_classes must be >= 2")
    if cfg.training.get("lr", 1e-4) <= 0:
        raise ValueError("training.lr must be > 0")
    if cfg.data["source"] not in ("genomic-benchmarks", "nt-benchmarks"):
        raise ValueError(
            "data.source must be 'genomic-benchmarks' or 'nt-benchmarks', "
            f"got {cfg.data['source']!r}"
        )
    return cfg


VALID_METRICS: List[str] = ["accuracy", "mcc", "f1"]
