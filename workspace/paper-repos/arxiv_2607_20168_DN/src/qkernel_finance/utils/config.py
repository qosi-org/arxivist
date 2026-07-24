"""
Config loading utility for the quantum-kernels finance reproduction (arXiv:2607.20168).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p, "r") as f:
            raw = yaml.safe_load(f)
        required = ["data", "universe", "walk_forward", "feature_selection", "quantum", "krr", "classical_baselines", "evaluation"]
        missing = [s for s in required if s not in raw]
        if missing:
            raise ValueError(f"Config at {path} missing required section(s): {missing}")
        return cls(raw=raw)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __repr__(self) -> str:  # noqa: D105
        return f"Config(sections={list(self.raw.keys())})"
