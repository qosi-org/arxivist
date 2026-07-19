#!/usr/bin/env python
"""Fine-tune DNABERT-2 on a GUE task.

Reproduction entrypoint for DNABERT-2 (arXiv:2306.15006). Loads official
pretrained weights + BPE tokenizer and fine-tunes a classification head per
the paper's Appendix A.3 recipe.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.dnabert2.data.gue import GUEDataset, collate_factory, load_gue_split
from src.dnabert2.data.tokenizer import DNATokenizer
from src.dnabert2.models.classifier import DNABERT2Classifier
from src.dnabert2.training.trainer import Trainer
from src.dnabert2.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DNABERT-2 on GUE")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default=None, help="override GUE task")
    p.add_argument("--subset", default=None, help="override task subset")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="tiny subset + few steps")
    p.add_argument("--dry-run", action="store_true", help="build components, skip training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.task:
        cfg.data["task"] = args.task
    if args.subset:
        cfg.data["subset"] = args.subset

    seed = args.seed if args.seed is not None else cfg.hardware.get("seed", 42)
    seed_everything(seed, cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    task = cfg.data["task"]
    subset = cfg.data["subset"]
    info = task_info(task)
    metric = info["metric"]
    num_classes = info["num_classes"]
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = DNATokenizer(cfg.model["model_name"], max_len=max_len)

    tr_seqs, tr_labels = load_gue_split(task, subset, "train", cfg.data["data_dir"])
    va_seqs, va_labels = load_gue_split(task, subset, "dev", cfg.data["data_dir"])
    if args.debug:
        tr_seqs, tr_labels = tr_seqs[:128], tr_labels[:128]
        va_seqs, va_labels = va_seqs[:128], va_labels[:128]
        cfg.training["epochs"] = 1

    train_ds = GUEDataset(tr_seqs, tr_labels)
    val_ds = GUEDataset(va_seqs, va_labels)

    model = DNABERT2Classifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        pool=cfg.model.get("pool", "mean"), device=str(device),
        attention_dropout=cfg.model.get("attention_dropout", 0.1),
    )

    collate = collate_factory(tokenizer, max_len)
    bs = cfg.training.get("batch_size", 32)
    nw = cfg.hardware.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)

    trainer = Trainer(model, train_loader, val_loader, cfg.training, device, metric)

    if args.dry_run:
        print("[dry-run] all components built successfully. Skipping training.")
        print(f"[dry-run] {trainer} | {model} | task={task}/{subset} metric={metric}")
        return

    best = trainer.fit()
    print(f"[done] best validation metrics: {best}")


if __name__ == "__main__":
    main()
