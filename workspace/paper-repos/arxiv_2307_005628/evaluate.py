#!/usr/bin/env python
"""Evaluate a fine-tuned DNAGPT checkpoint on the GSR test split."""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.dnagpt.data.gsr import GSRDataset, collate_factory, load_gsr_split
from src.dnagpt.data.tokenizer import DNAGPTTokenizer
from src.dnagpt.models.dnagpt import DNAGPT, DNAGPTConfig
from src.dnagpt.training.trainer import Trainer
from src.dnagpt.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DNAGPT")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True, help="path to best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    task = cfg.data["task"]
    info = task_info(task)
    metric, species = info["metric"], info["species"]
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = DNAGPTTokenizer(k=cfg.model.get("k", 6))
    model_cfg = DNAGPTConfig.from_variant(
        cfg.model.get("variant", "M"), vocab_size=tokenizer.vocab_size,
        seq_len=max_len, num_classes=2,
    )
    model = DNAGPT(model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    te_seqs, te_labels = load_gsr_split(task, "test", cfg.data["data_dir"], length=max_len)
    val_ds = GSRDataset(te_seqs, te_labels)
    collate = collate_factory(tokenizer, species, max_len)
    loader = DataLoader(val_ds, batch_size=cfg.training.get("batch_size", 8),
                        shuffle=False, collate_fn=collate)
    trainer = Trainer(model, loader, loader, cfg.training, device, metric)
    print(f"[eval] {trainer.evaluate(loader)}")


if __name__ == "__main__":
    main()
