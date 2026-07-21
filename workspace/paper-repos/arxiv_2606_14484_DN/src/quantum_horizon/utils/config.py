"""
Configuration loading and global random-seed management for the
Quantum Horizon reproduction (arXiv:2606.14484).
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
class QuantumHorizonConfig:
    """Typed wrapper around the raw config.yaml dictionary."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def mining(self) -> Dict[str, Any]:
        return self.raw["mining"]

    @property
    def exposure(self) -> Dict[str, Any]:
        return self.raw["exposure"]

    @property
    def mempool_race(self) -> Dict[str, Any]:
        return self.raw["mempool_race"]

    @property
    def migration(self) -> Dict[str, Any]:
        return self.raw["migration"]

    @property
    def readiness_survey(self) -> Dict[str, Any]:
        return self.raw["readiness_survey"]

    @property
    def hardware(self) -> Dict[str, Any]:
        return self.raw["hardware"]

    def __repr__(self) -> str:  # noqa: D105
        return f"QuantumHorizonConfig(sections={list(self.raw.keys())})"


def load_config(path: str) -> QuantumHorizonConfig:
    """Load and validate a config.yaml file.

    Args:
        path: filesystem path to the YAML config.

    Returns:
        A validated QuantumHorizonConfig.

    Raises:
        ConfigError: if a required top-level section is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_sections = [
        "model", "mining", "exposure", "mempool_race", "migration", "readiness_survey", "hardware",
    ]
    missing = [s for s in required_sections if s not in raw]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")

    return QuantumHorizonConfig(raw=raw)


def set_global_seed(seed: int) -> None:
    """Seed Python's random module and NumPy.

    Args:
        seed: the integer seed to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
