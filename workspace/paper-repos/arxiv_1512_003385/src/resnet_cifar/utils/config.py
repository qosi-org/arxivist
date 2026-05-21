"""YAML config loading + CLI override merging.

Configs follow the schema in configs/resnet20.yaml. CLI overrides use dotted keys, e.g.
`--override training.learning_rate=0.05 hardware.device=cpu`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and validate the top-level structure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} did not parse to a dict")

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Minimal validation; trainer-specific validation lives in the trainer."""
    required_top = {"model", "training", "data", "evaluation", "hardware"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"config missing required top-level keys: {sorted(missing)}")

    name = cfg["model"].get("name")
    valid_names = {"resnet20", "resnet32", "resnet44", "resnet56", "resnet110"}
    if name not in valid_names:
        raise ValueError(f"model.name must be one of {sorted(valid_names)}, got {name!r}")

    opt = cfg["training"].get("optimizer", "sgd")
    if opt != "sgd":
        raise ValueError(f"only optimizer='sgd' is supported (paper Sec. 4.2); got {opt!r}")


def merge_cli_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides, parsing the value as YAML so types are preserved."""
    if not overrides:
        return cfg

    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override must be 'dotted.key=value', got {spec!r}")
        key, raw_value = spec.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(cfg, key.split("."), value)

    return cfg


def _set_nested(d: dict[str, Any], path: list[str], value: Any) -> None:
    current = d
    for k in path[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[path[-1]] = value
