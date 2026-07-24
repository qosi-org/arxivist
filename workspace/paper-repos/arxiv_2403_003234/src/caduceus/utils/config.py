"""Config loading, seeding, and the Genomic Benchmarks task registry.

The task registry maps each supported downstream task to its evaluation metric
and class count, using the genomic_benchmarks dataset names (Caduceus Table 1).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
import yaml

# task -> {metric, num_classes, default max_len}. Genomic Benchmarks (Table 1):
# all top-1 accuracy, binary unless noted. Names follow the genomic_benchmarks API.
TASKS: Dict[str, Dict[str, Any]] = {
    "human_nontata_promoters":          {"metric": "accuracy", "num_classes": 2, "max_len": 500,  "benchmark": "genomic_benchmarks"},
    "human_enhancers_cohn":             {"metric": "accuracy", "num_classes": 2, "max_len": 500,  "benchmark": "genomic_benchmarks"},
    "human_enhancers_ensembl":          {"metric": "accuracy", "num_classes": 2, "max_len": 573,  "benchmark": "genomic_benchmarks"},
    "human_ocr_ensembl":                {"metric": "accuracy", "num_classes": 2, "max_len": 593,  "benchmark": "genomic_benchmarks"},
    "human_ensembl_regulatory":         {"metric": "accuracy", "num_classes": 3, "max_len": 802,  "benchmark": "genomic_benchmarks"},
    "demo_coding_vs_intergenomic_seqs": {"metric": "accuracy", "num_classes": 2, "max_len": 200,  "benchmark": "genomic_benchmarks"},
    "demo_human_or_worm":               {"metric": "accuracy", "num_classes": 2, "max_len": 200,  "benchmark": "genomic_benchmarks"},
    "dummy_mouse_enhancers_ensembl":    {"metric": "accuracy", "num_classes": 2, "max_len": 4707, "benchmark": "genomic_benchmarks"},
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
        return f"Config(model={self.model.get('model_name')}, task={self.data.get('task')})"


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
    if cfg.training.get("lr", 1e-3) <= 0:
        raise ValueError("training.lr must be > 0")
    variant = cfg.model.get("variant", "ph").lower()
    if variant not in ("ph", "ps"):
        raise ValueError(f"model.variant must be 'ph' or 'ps', got {variant!r}")
    return cfg


def task_info(task: str) -> Dict[str, Any]:
    """Return the registry entry for a task."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Options: {list(TASKS)}")
    return TASKS[task]
