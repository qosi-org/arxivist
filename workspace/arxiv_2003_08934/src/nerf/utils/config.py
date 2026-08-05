"""
utils/config.py — YAML config loading, validation, and global seeding.

Implements the reproducibility requirements from architecture_plan.json:
  - a single `NeRFConfig` class that loads/validates configs/*.yaml
  - a `set_global_seed` utility seeding Python, NumPy, and TensorFlow

Paper reference: arXiv:2003.08934, Appendix A ("Implementation Details").
No equations are implemented in this file; it is pure engineering scaffolding.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

_REQUIRED_TOP_LEVEL_KEYS = ("model", "model_variant", "training", "data", "evaluation", "hardware")
_VALID_MODEL_VARIANTS = (
    "full",
    "no_positional_encoding",
    "no_view_dependence",
    "no_hierarchical_sampling",
)
_VALID_DATASET_TYPES = ("blender", "llff", "deepvoxels")


@dataclass
class NeRFConfig:
    """
    Typed, validated view over a config.yaml dictionary.

    Args:
        raw: the parsed YAML dictionary (see configs/config.yaml for the full
            documented schema, including which fields are paper-derived vs
            `# ASSUMED`).
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "NeRFConfig":
        """Load and validate a config YAML file.

        Args:
            path: filesystem path to a config.yaml-style file.

        Returns:
            A validated NeRFConfig instance.

        Raises:
            FileNotFoundError: if `path` does not exist.
            ValueError: if required keys are missing or values are invalid.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Config file is empty: {path}")
        return cls(raw=raw)

    def merge_overrides(self, overrides: dict[str, Any]) -> "NeRFConfig":
        """Return a new NeRFConfig with a shallow-per-section dict merge applied.

        Args:
            overrides: dict shaped like the config sections, e.g.
                {"training": {"num_training_steps": 100}}.

        Returns:
            A new, re-validated NeRFConfig.
        """
        merged = {k: dict(v) if isinstance(v, dict) else v for k, v in self.raw.items()}
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(merged.get(section), dict):
                merged[section].update(values)
            else:
                merged[section] = values
        return NeRFConfig(raw=merged)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in self.raw]
        if missing:
            raise ValueError(f"Config is missing required top-level keys: {missing}")

        variant = self.raw["model_variant"].get("name")
        if variant not in _VALID_MODEL_VARIANTS:
            raise ValueError(
                f"model_variant.name must be one of {_VALID_MODEL_VARIANTS}, got {variant!r}"
            )

        dataset_type = self.raw["data"].get("dataset_type")
        if dataset_type not in _VALID_DATASET_TYPES:
            raise ValueError(
                f"data.dataset_type must be one of {_VALID_DATASET_TYPES}, got {dataset_type!r}"
            )

        nc = self.raw["model"].get("num_coarse_samples", 0)
        nf = self.raw["model"].get("num_fine_samples", 0)
        if nc <= 0:
            raise ValueError(f"model.num_coarse_samples must be > 0, got {nc}")
        if self.raw["model"].get("use_hierarchical_sampling", False) and nf <= 0:
            raise ValueError(
                "model.use_hierarchical_sampling=true requires model.num_fine_samples > 0"
            )

        lr_init = self.raw["training"].get("learning_rate_init")
        lr_final = self.raw["training"].get("learning_rate_final")
        if lr_init is None or lr_final is None or lr_init <= 0 or lr_final <= 0:
            raise ValueError("training.learning_rate_init/final must both be > 0")

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def __getitem__(self, section: str) -> Any:
        return self.raw[section]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def __repr__(self) -> str:  # noqa: D105
        variant = self.raw.get("model_variant", {}).get("name")
        dataset = self.raw.get("data", {}).get("dataset_type")
        steps = self.raw.get("training", {}).get("num_training_steps")
        return f"NeRFConfig(variant={variant!r}, dataset={dataset!r}, num_training_steps={steps})"


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed: the integer seed to apply everywhere.
        deterministic: if True, also enables TensorFlow's deterministic ops
            mode. This can noticeably slow down training/rendering and is
            recommended only for debugging (see configs/config.yaml comment
            on training.deterministic).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
