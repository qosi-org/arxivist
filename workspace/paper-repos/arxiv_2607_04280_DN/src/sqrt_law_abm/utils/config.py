"""
Configuration loading and global seeding utilities.

Implements the reproducibility requirements of the ArXivist code-generation
protocol: a single `set_global_seed` entrypoint that seeds Python's `random`,
NumPy, and PyTorch, plus a lightweight `Config` wrapper around the YAML
config file described in `architecture_plan.json.config_schema`.

This paper (arxiv_2607_04280) is a discrete-event agent-based simulation, not
a gradient-trained model, so there is no optimizer/lr-schedule config here —
only simulation and agent-population parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover - torch is a listed dependency but guard anyway
    torch = None


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility.

    Args:
        seed: Global random seed.
        deterministic: If True, also request deterministic PyTorch algorithms.
            This can noticeably slow down GPU execution; it has no effect on
            the CPU-only NumPy simulation path used by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """Thin, validated wrapper around the YAML config described in
    `architecture_plan.json.config_schema`.

    Args:
        raw: The parsed YAML content as a nested dict.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load and validate a config YAML file.

        Args:
            path: Path to a config YAML file (e.g. configs/config.yaml).

        Returns:
            A validated Config instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If required top-level sections are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        required_sections = ["model", "simulation", "data", "evaluation", "hardware"]
        missing = [s for s in required_sections if s not in raw]
        if missing:
            raise ValueError(
                f"Config at {path} is missing required section(s): {missing}"
            )

        cfg = cls(raw=raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Sanity-check numeric ranges. Raises ValueError with a helpful message."""
        m = self.raw["model"]
        if m["n_hft_min"] > m["n_hft_max"]:
            raise ValueError("model.n_hft_min must be <= model.n_hft_max")
        if m["n_retail_min"] > m["n_retail_max"]:
            raise ValueError("model.n_retail_min must be <= model.n_retail_max")
        if not (0.0 <= m["hft"]["replenish_prob"] <= 1.0):
            raise ValueError("model.hft.replenish_prob must be in [0, 1]")

        sim = self.raw["simulation"]
        if sim["warmup_steps"] >= sim["n_steps"]:
            raise ValueError("simulation.warmup_steps must be < simulation.n_steps")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested-key convenience getter, e.g. cfg.get('model', 'hft', 'replenish_prob')."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def apply_debug_overrides(self) -> "Config":
        """Return a copy of this config with a much smaller footprint for --debug runs."""
        import copy

        dbg = copy.deepcopy(self.raw)
        dbg["simulation"]["n_steps"] = 5_000
        dbg["simulation"]["warmup_steps"] = 500
        dbg["simulation"]["n_stocks_full"] = 4
        dbg["simulation"]["n_stocks_counterfactual"] = 2
        return Config(raw=dbg)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Config(sections={list(self.raw.keys())})"
