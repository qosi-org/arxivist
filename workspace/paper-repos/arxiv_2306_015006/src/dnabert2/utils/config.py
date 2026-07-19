"""Config loading, seeding, and the GUE task registry.

The GUE task registry encodes, per task, the evaluation metric and number of
classes (from DNABERT-2 Table 1 / Table 12). Nothing is hardcoded elsewhere.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
import yaml

# GUE task registry: task -> {metric, num_classes, default max_len, subsets, hf_prefix}
# Source: DNABERT-2 Table 1 & Table 12 for metric/classes/lengths.
# `hf_prefix` maps our paper-facing task name to the config naming used by the
# GUE dataset on the HuggingFace hub (verified against its BuilderConfig list),
# e.g. promoter_detection/all -> "prom_300_all".
GUE_TASKS: Dict[str, Dict[str, Any]] = {
    "promoter_detection":       {"metric": "mcc", "num_classes": 2, "max_len": 128, "hf_prefix": "prom_300_",  "subsets": ["all", "notata", "tata"]},
    "core_promoter_detection":  {"metric": "mcc", "num_classes": 2, "max_len": 32,  "hf_prefix": "prom_core_", "subsets": ["all", "notata", "tata"]},
    "tf_human":                 {"metric": "mcc", "num_classes": 2, "max_len": 128, "hf_prefix": "human_tf_",  "subsets": ["0", "1", "2", "3", "4"]},
    "tf_mouse":                 {"metric": "mcc", "num_classes": 2, "max_len": 128, "hf_prefix": "mouse_",     "subsets": ["0", "1", "2", "3", "4"]},
    "epigenetic_marks":         {"metric": "mcc", "num_classes": 2, "max_len": 512, "hf_prefix": "emp_",       "subsets": ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac"]},
    "splice":                   {"metric": "mcc", "num_classes": 3, "max_len": 128, "hf_prefix": "splice_",    "subsets": ["reconstructed"]},
    "covid_variant":            {"metric": "f1",  "num_classes": 9, "max_len": 256, "hf_prefix": "virus_",     "subsets": ["covid"]},
}


def hf_config_name(task: str, subset: str) -> str:
    """Map a (task, subset) to the GUE HuggingFace BuilderConfig name."""
    info = GUE_TASKS[task]
    return f"{info['hf_prefix']}{subset}"


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
        return (f"Config(model={self.model.get('model_name')}, "
                f"task={self.data.get('task')}/{self.data.get('subset')})")


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
    if task not in GUE_TASKS:
        raise ValueError(f"data.task must be one of {list(GUE_TASKS)}, got {task!r}")
    if cfg.training.get("lr", 3e-5) <= 0:
        raise ValueError("training.lr must be > 0")
    return cfg


def task_info(task: str) -> Dict[str, Any]:
    """Return the GUE registry entry for a task."""
    if task not in GUE_TASKS:
        raise ValueError(f"Unknown GUE task {task!r}. Options: {list(GUE_TASKS)}")
    return GUE_TASKS[task]
