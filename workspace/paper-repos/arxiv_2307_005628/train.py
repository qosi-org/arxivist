#!/usr/bin/env python
"""Fine-tune DNAGPT on a GSR recognition task.

Reproduction entrypoint for DNAGPT (arXiv:2307.05628). Builds the from-scratch
DNAGPT (optionally loading the official .pth), then fine-tunes the classification
head on a GSR (PAS/TIS) task per Sec 4.1 / Fig S4.
"""
from __future__ import annotations

import argparse
import os

from torch.utils.data import DataLoader

from src.dnagpt.data.gsr import GSRDataset, collate_factory, load_gsr_split
from src.dnagpt.data.tokenizer import DNAGPTTokenizer
from src.dnagpt.models.dnagpt import DNAGPT, DNAGPTConfig
from src.dnagpt.training.trainer import Trainer
from src.dnagpt.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DNAGPT on GSR recognition")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default=None, help="override GSR task")
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
    species = info["species"]
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = DNAGPTTokenizer(k=cfg.model.get("k", 6))

    # model: load official weights if the .pth exists, else from-scratch.
    variant = cfg.model.get("variant", "M")
    model_cfg = DNAGPTConfig.from_variant(
        variant, vocab_size=tokenizer.vocab_size, seq_len=max_len, num_classes=2,
    )
    ckpt = cfg.model.get("ckpt_path")
    if ckpt and os.path.isfile(ckpt):
        model = DNAGPT.from_pretrained(ckpt, model_cfg, device=str(device))
    else:
        if ckpt:
            print(f"[model] {ckpt} not found -> training from scratch "
                  f"(run data/download.py for official weights). {model_cfg}")
        model = DNAGPT(model_cfg).to(device)
    print(model)

    tr_seqs, tr_labels = load_gsr_split(task, "train", cfg.data["data_dir"], length=max_len)
    te_seqs, te_labels = load_gsr_split(task, "test", cfg.data["data_dir"], length=max_len)
    if args.debug:
        tr_seqs, tr_labels = tr_seqs[:64], tr_labels[:64]
        te_seqs, te_labels = te_seqs[:64], te_labels[:64]
        cfg.training["epochs"] = 1

    train_ds = GSRDataset(tr_seqs, tr_labels)
    val_ds = GSRDataset(te_seqs, te_labels)
    collate = collate_factory(tokenizer, species, max_len)
    bs = cfg.training.get("batch_size", 8)
    nw = cfg.hardware.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)

    trainer = Trainer(model, train_loader, val_loader, cfg.training, device, metric)

    if args.dry_run:
        print("[dry-run] all components built successfully. Skipping training.")
        print(f"[dry-run] {trainer} | task={task} metric={metric} species={species}")
        return

    best = trainer.fit()

    from src.dnagpt.data import gsr as _gsr
    if _gsr.USING_SYNTHETIC:
        print(f"[done] SYNTHETIC best metrics: {best}")
        print("[done] ^^ SMOKE TEST on synthetic data — NOT a reproduction of the paper.")
        print("[done]    Provide real DeepGSR data (data/README_data.md) for Table S2 numbers.")
    else:
        print(f"[done] best validation metrics (real DeepGSR data): {best}")
        print(f"[done]    Paper DNAGPT-M target for {task}: see README 'Expected results'.")


if __name__ == "__main__":
    main()
