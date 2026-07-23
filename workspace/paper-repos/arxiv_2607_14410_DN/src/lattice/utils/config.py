"""
Config loading and reproducibility (seeding) utilities.

Implements the non-negotiable reproducibility requirements from the ArXivist
Code Generator protocol: a single seeding entrypoint that seeds Python's
`random`, NumPy, and PyTorch (CPU + CUDA if available), plus an optional
deterministic-algorithms flag.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class ConfigError(ValueError):
    """Raised when a config value fails validation."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a LATTICE config YAML file.

    Args:
        path: path to a config.yaml file (see configs/config.yaml for the
            canonical, fully-commented example).

    Returns:
        The parsed config as a nested dict.

    Raises:
        ConfigError: if a required top-level section is missing.
    """
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    required_sections = ["model", "training", "data", "evaluation", "hardware"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ConfigError(
            f"Config at {path} is missing required section(s): {missing}. "
            f"See configs/config.yaml for the expected schema."
        )

    _validate_training_section(cfg["training"])
    _validate_model_section(cfg["model"])
    return cfg


def _validate_training_section(training_cfg: dict[str, Any]) -> None:
    if training_cfg.get("recon_loss") not in ("huber", "mse"):
        raise ConfigError(
            "training.recon_loss must be 'huber' (Appendix H default, ASSUMED) "
            f"or 'mse' (main-text Eq.6 literal form); got {training_cfg.get('recon_loss')!r}"
        )
    if not (0.0 < training_cfg.get("masking_ratio", 0.15) < 1.0):
        raise ConfigError("training.masking_ratio must be in (0, 1)")


def _validate_model_section(model_cfg: dict[str, Any]) -> None:
    pair = model_cfg.get("projection_head", {}).get("aligned_modality_pair")
    if pair is not None and len(pair) != 2:
        raise ConfigError(
            "model.projection_head.aligned_modality_pair must be a 2-element "
            f"[a, b] list of modality indices; got {pair!r}"
        )


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: the seed value. The paper uses seed=42 for single-sample runs
            and seed=7 (plus a stability sweep over 11 seeds) for joint
            multisample analysis (SIR training_pipeline, Appendix H).
        deterministic: if True, sets `torch.use_deterministic_algorithms(True)`
            and disables cuDNN benchmarking. This can noticeably slow down
            training and is off by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass
class RunSummary:
    """Small struct printed at the start of training (Stage 4 requirement)."""

    model_name: str
    num_params: int
    num_samples: int
    spots_per_sample: int
    steps_per_epoch: int
    device: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        lines = [
            "=" * 60,
            f"LATTICE run summary — {self.model_name}",
            "=" * 60,
            f"  Parameters:      {self.num_params:,}",
            f"  Samples:         {self.num_samples}",
            f"  Spots/sample:    {self.spots_per_sample}",
            f"  Steps/epoch:     {self.steps_per_epoch}",
            f"  Device:          {self.device}",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
