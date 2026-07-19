"""
Config loading and reproducibility (seeding) utilities.

There is no PyTorch model being trained in this repo, but we still seed torch (if
installed) for the optional fbsde_stub.py heuristic, per ArXivist's reproducibility
requirements.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml


@dataclass
class ExperimentConfig:
    """Holds the full experiment configuration loaded from configs/config.yaml.

    Args:
        model: model/simulation parameters (n_assets, C, sigma_mode, T)
        training: time-discretization / seed parameters (no NN training happens here)
        data: data section (always synthetic for this paper — no real dataset exists)
        evaluation: numerical-comparison tolerances
        hardware: device / worker settings (kept for ArXivist schema compatibility)
    """

    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return ExperimentConfig(
            model=raw.get("model", {}),
            training=raw.get("training", {}),
            data=raw.get("data", {}),
            evaluation=raw.get("evaluation", {}),
            hardware=raw.get("hardware", {}),
        )

    def seed_everything(self, seed: int | None = None) -> int:
        """Seed python's random, numpy, and torch (if installed). Returns the seed used."""
        s = seed if seed is not None else int(self.training.get("seed", 0))
        random.seed(s)
        np.random.seed(s)
        try:
            import torch  # optional dependency, only used by fbsde_stub.py

            torch.manual_seed(s)
        except ImportError:
            pass
        return s
