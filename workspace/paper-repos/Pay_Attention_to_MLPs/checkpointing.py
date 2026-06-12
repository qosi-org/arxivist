"""
gmlp/utils/checkpointing.py
---------------------------
Checkpoint save/load utilities for gMLP experiments.
Handles NLP pretraining, vision training, and finetuning checkpoints.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path:       Full file path (e.g. 'outputs/checkpoint_step_10000.pt').
        model:      Model with state_dict().
        optimizer:  Optimizer with state_dict().
        scheduler:  LR scheduler with state_dict().
        step:       Current global step or epoch.
        metadata:   Optional extra fields (e.g. best_metric, config).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    logger.info(f"Checkpoint saved → {path}  (step={step})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        path:      Path to checkpoint file.
        model:     Model instance (modified in-place).
        optimizer: Optimizer to restore (optional).
        scheduler: LR scheduler to restore (optional).
        device:    Device to load tensors onto.
        strict:    Strict state_dict loading (set False for partial loads).

    Returns:
        Full checkpoint dict (for accessing metadata like global_step).
    """
    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt.get("global_step", 0)
    logger.info(f"Checkpoint loaded ← {path}  (step={step})")
    return ckpt


def find_latest_checkpoint(output_dir: str, pattern: str = "checkpoint_step_*.pt") -> Optional[str]:
    """
    Find the most recent step checkpoint in output_dir.
    Returns None if no checkpoint found.
    """
    matches = glob.glob(os.path.join(output_dir, pattern))
    if not matches:
        return None
    # Sort by step number extracted from filename
    def _step(p: str) -> int:
        base = os.path.basename(p)
        parts = base.replace(".pt", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    return max(matches, key=_step)


def load_pretrained_weights(model: torch.nn.Module, path: str, strict: bool = False) -> None:
    """
    Load only model weights from a checkpoint (no optimizer/scheduler).
    Used when initialising a model for finetuning from a pretraining checkpoint.

    strict=False allows loading a pretrained MLM checkpoint into a classification
    model (missing/unexpected keys for head layers are tolerated).
    """
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
    logger.info(f"Pretrained weights loaded ← {path}")
