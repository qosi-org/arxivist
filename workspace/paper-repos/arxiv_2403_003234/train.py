#!/usr/bin/env python
"""Fine-tune Caduceus on a Genomic Benchmarks classification task.

Reproduction entrypoint for Caduceus (arXiv:2403.03234). Loads the official
`kuleshov-group/caduceus-*` weights + char tokenizer and fine-tunes a mean-pool
+ linear classification head (Sec 5.2.1). Caduceus-Ph applies post-hoc
conjoining (RC-ensemble) at evaluation.
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.caduceus.data.benchmarks import GenomicDataset, collate_factory, load_split
from src.caduceus.data.tokenizer import CharTokenizer
from src.caduceus.models.classifier import CaduceusClassifier
from src.caduceus.training.trainer import Trainer
from src.caduceus.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Caduceus")
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
    variant = cfg.model.get("variant", "ph").lower()

    tokenizer = CharTokenizer(cfg.model["model_name"], max_len=max_len)

    tr_seqs, tr_labels, _ = load_split(benchmark, task, "train", cfg.data["data_dir"])
    te_seqs, te_labels, _ = load_split(benchmark, task, "test", cfg.data["data_dir"])
    if args.debug:
        tr_seqs, tr_labels = tr_seqs[:64], tr_labels[:64]
        te_seqs, te_labels = te_seqs[:64], te_labels[:64]
        cfg.training["epochs"] = 1

    rc_aug = bool(cfg.training.get("rc_aug", variant == "ph"))
    train_ds = GenomicDataset(tr_seqs, tr_labels, rc_aug=rc_aug)
    val_ds = GenomicDataset(te_seqs, te_labels, rc_aug=False)

    model = CaduceusClassifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        variant=variant, device=str(device),
        rc_complement_ids=tokenizer.complement_id_map(),
    )

    collate = collate_factory(tokenizer, max_len)
    bs = cfg.training.get("batch_size", 128)
    nw = cfg.hardware.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)

    conjoin = bool(cfg.evaluation.get("conjoin", variant == "ph"))
    trainer = Trainer(model, train_loader, val_loader, cfg.training, device, metric, conjoin=conjoin)

    if args.dry_run:
        print("[dry-run] all components built successfully. Skipping training.")
        print(f"[dry-run] {trainer} | {model} | task={task} metric={metric}")
        return

    best = trainer.fit()
    print(f"[done] best validation metrics: {best}")


if __name__ == "__main__":
    main()
