#!/usr/bin/env python
"""Fine-tune HyenaDNA on a genomic classification dataset.

Reproduction entrypoint for HyenaDNA (arXiv:2306.15794). Loads official
pretrained weights from HuggingFace and fine-tunes a classification head.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.hyenadna.data.dataset import build_dataset
from src.hyenadna.data.tokenizer import CharTokenizer
from src.hyenadna.models.pretrained import HyenaDNAClassifier
from src.hyenadna.training.trainer import Trainer
from src.hyenadna.utils.config import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune HyenaDNA")
    p.add_argument("--config", default="configs/config.yaml", help="path to YAML config")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--debug", action="store_true", help="tiny subset + few steps")
    p.add_argument("--dry-run", action="store_true", help="build everything, skip training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.hardware.get("seed", 42)
    seed_everything(seed, cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    tokenizer = CharTokenizer(max_len=cfg.data.get("max_len", 1024))
    train_ds, n_cls = build_dataset(cfg.data, "train", tokenizer)
    val_ds, _ = build_dataset(cfg.data, cfg.evaluation.get("split", "test"), tokenizer)

    if args.debug:
        train_ds.sequences, train_ds.labels = train_ds.sequences[:128], train_ds.labels[:128]
        val_ds.sequences, val_ds.labels = val_ds.sequences[:128], val_ds.labels[:128]
        cfg.training["epochs"] = 1

    num_classes = cfg.model.get("num_classes") or n_cls
    model = HyenaDNAClassifier.from_pretrained(
        variant=cfg.model["variant"],
        num_classes=num_classes,
        d_model=cfg.model.get("d_model", 128),
        pool=cfg.model.get("pool", "mean"),
        device=str(device),
    )
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[resume] loaded {args.resume}")

    bs = cfg.training.get("batch_size", 32)
    nw = cfg.hardware.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    trainer = Trainer(model, train_loader, val_loader, cfg.training, device,
                      cfg.evaluation.get("metrics", ["accuracy"]))

    if args.dry_run:
        print("[dry-run] all components built successfully. Skipping training.")
        print(f"[dry-run] {trainer} | {model}")
        return

    best = trainer.fit()
    print(f"[done] best validation metrics: {best}")


if __name__ == "__main__":
    main()
