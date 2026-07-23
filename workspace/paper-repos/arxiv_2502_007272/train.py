#!/usr/bin/env python
"""Fine-tune GENERator on a genomic classification task.

Reproduction entrypoint for GENERator (arXiv:2502.07272). Loads the official
1.2B LLaMA-decoder weights + 6-mer tokenizer and fine-tunes a classification
head on the last-token (<EOS>) embedding, per Appendix C.4.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.generator.data.benchmarks import GenomicDataset, collate_factory, load_split
from src.generator.data.tokenizer import GenomicTokenizer
from src.generator.models.classifier import GENERatorClassifier
from src.generator.training.trainer import Trainer
from src.generator.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune GENERator")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default=None, help="override benchmark task")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="tiny subset + few steps")
    p.add_argument("--dry-run", action="store_true", help="build components, skip training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.task:
        cfg.data["task"] = args.task

    seed = args.seed if args.seed is not None else cfg.hardware.get("seed", 42)
    seed_everything(seed, cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    task = cfg.data["task"]
    info = task_info(task)
    metric = info["metric"]
    num_classes = info["num_classes"]
    benchmark = cfg.data.get("benchmark", info["benchmark"])
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = GenomicTokenizer(cfg.model["model_name"], max_len=max_len)

    tr_seqs, tr_labels, _ = load_split(benchmark, task, "train", cfg.data["data_dir"])
    te_seqs, te_labels, _ = load_split(benchmark, task, "test", cfg.data["data_dir"])
    if args.debug:
        tr_seqs, tr_labels = tr_seqs[:64], tr_labels[:64]
        te_seqs, te_labels = te_seqs[:64], te_labels[:64]
        cfg.training["epochs"] = 1

    train_ds = GenomicDataset(tr_seqs, tr_labels)
    val_ds = GenomicDataset(te_seqs, te_labels)

    model = GENERatorClassifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        device=str(device), load_in_8bit=cfg.model.get("load_in_8bit", False),
    )

    collate = collate_factory(tokenizer, max_len)
    bs = cfg.training.get("batch_size", 64)
    nw = cfg.hardware.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)

    trainer = Trainer(model, train_loader, val_loader, cfg.training, device, metric)

    if args.dry_run:
        print("[dry-run] all components built successfully. Skipping training.")
        print(f"[dry-run] {trainer} | {model} | task={task} metric={metric}")
        return

    best = trainer.fit()
    print(f"[done] best validation metrics: {best}")


if __name__ == "__main__":
    main()
