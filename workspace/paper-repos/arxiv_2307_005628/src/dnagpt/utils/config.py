"""Config loading, seeding, and the GSR task registry.

Task registry maps each GSR task to its species, sequence length, and default
metric (all GSR tasks are binary classification; paper reports acc/f1/mcc).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
import yaml

# task -> {species, max_len, metric}. Lengths: PAS 606, TIS 603 (300bp flanks).
TASKS: Dict[str, Dict[str, Any]] = {
    "human_pas_aataaa": {"species": "Human",    "max_len": 606, "metric": "accuracy"},
    "human_pas_all":    {"species": "Human",    "max_len": 606, "metric": "accuracy"},
    "human_tis_atg":    {"species": "Human",    "max_len": 603, "metric": "accuracy"},
    "mouse_pas_aataaa": {"species": "Mouse",    "max_len": 606, "metric": "accuracy"},
    "mouse_tis_atg":    {"species": "Mouse",    "max_len": 603, "metric": "accuracy"},
    "bovine_pas_aataaa":{"species": "Bovine",   "max_len": 606, "metric": "accuracy"},
    "bovine_tis_atg":   {"species": "Bovine",   "max_len": 603, "metric": "accuracy"},
    "fruitfly_pas_aataaa": {"species": "FruitFly", "max_len": 606, "metric": "accuracy"},
    "fruitfly_tis_atg":    {"species": "FruitFly", "max_len": 603, "metric": "accuracy"},
}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto'|'cuda'|'cpu' to a torch.device."""
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
        return f"Config(variant={self.model.get('variant')}, task={self.data.get('task')})"


def load_config(path: str) -> Config:
    """Load and validate a YAML config."""
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
    task = cfg.data.get("task")
    if task not in TASKS:
        raise ValueError(f"data.task must be one of {list(TASKS)}, got {task!r}")
    if cfg.training.get("lr", 3e-5) <= 0:
        raise ValueError("training.lr must be > 0")
    variant = cfg.model.get("variant", "M")
    if variant not in ("H", "M", "S-512", "B-512"):
        raise ValueError(f"model.variant must be H|M|S-512|B-512, got {variant!r}")
    return cfg


def task_info(task: str) -> Dict[str, Any]:
    """Return the registry entry for a task."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Options: {list(TASKS)}")
    return TASKS[task]
