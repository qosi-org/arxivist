"""
Configuration loading and global random-seed management.

Implements arXiv:2607.12990 reproducibility requirements: every entrypoint
(train.py, evaluate.py, inference.py, run_hardware_calibration.py) loads its
settings from a single YAML file via `load_config`, and calls `set_global_seed`
before any random operation (NumPy path simulation, PyTorch variational
training, Qiskit finite-shot sampling).
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
class QuantumCVAConfig:
    """Typed wrapper around the raw config.yaml dictionary.

    Args:
        raw: the full parsed YAML config as a nested dict.
    """

    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

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
        return f"QuantumCVAConfig(sections={list(self.raw.keys())})"


def load_config(path: str) -> QuantumCVAConfig:
    """Load and validate a config.yaml file.

    Args:
        path: filesystem path to the YAML config.

    Returns:
        A validated QuantumCVAConfig.

    Raises:
        ConfigError: if a required top-level section is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required_sections = ["model", "training", "data", "evaluation", "hardware"]
    missing = [s for s in required_sections if s not in raw]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")

    _validate_model_section(raw["model"])
    return QuantumCVAConfig(raw=raw)


def _validate_model_section(model_cfg: Dict[str, Any]) -> None:
    """Validate qubit-count fields required by every circuit module."""
    required_keys = [
        "num_time_qubits_m",
        "num_asset_qubits_per_underlying",
        "num_underlyings_d",
        "num_ancillas",
    ]
    for key in required_keys:
        if key not in model_cfg:
            raise ConfigError(f"model config missing required key: {key}")
        if not isinstance(model_cfg[key], int) or model_cfg[key] < 1:
            raise ConfigError(f"model.{key} must be a positive integer, got {model_cfg[key]!r}")


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python's random module, NumPy, and (if installed) PyTorch.

    Args:
        seed: the integer seed to apply everywhere.
        deterministic: if True, also request deterministic algorithms in
            PyTorch (may reduce performance -- see paper's Appendix A.1 which
            fixes NMC seed=100000 for the classical Monte Carlo benchmark;
            the quantum-training seed itself is not stated in the paper and
            is assumed here for internal run-to-run reproducibility, SIR
            confidence 0.5).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
