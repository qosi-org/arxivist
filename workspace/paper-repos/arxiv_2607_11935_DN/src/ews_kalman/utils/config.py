"""
Configuration loading and global random-seed management for the
arXiv:2607.11935 (TVP-Kalman EWS) reproduction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when a loaded config fails validation."""


@dataclass
class EWSConfig:
    """Typed wrapper around the raw config.yaml dictionary."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def hardware(self) -> Dict[str, Any]:
        return self.raw["hardware"]

    def __repr__(self) -> str:  # noqa: D105
        return f"EWSConfig(sections={list(self.raw.keys())})"


def load_config(path: str) -> EWSConfig:
    """Load and validate a config.yaml file.

    Args:
        path: filesystem path to the YAML config.

    Returns:
        A validated EWSConfig.

    Raises:
        ConfigError: if a required top-level section is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_sections = ["model", "data", "evaluation", "hardware"]
    missing = [s for s in required_sections if s not in raw]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")

    return EWSConfig(raw=raw)


def set_global_seed(seed: int) -> None:
    """Seed Python's random module and NumPy for reproducible simulations.

    Args:
        seed: integer seed applied to random and numpy.random.
    """
    random.seed(seed)
    np.random.seed(seed)
