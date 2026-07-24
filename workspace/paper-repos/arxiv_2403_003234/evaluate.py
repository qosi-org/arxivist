#!/usr/bin/env python
"""Evaluate a fine-tuned Caduceus checkpoint on the test split.

Applies Caduceus-Ph post-hoc conjoining (RC-ensemble) when the config selects
variant "ph" and evaluation.conjoin is true.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.caduceus.data.benchmarks import GenomicDataset, collate_factory, load_split
from src.caduceus.data.tokenizer import CharTokenizer
from src.caduceus.models.classifier import CaduceusClassifier
from src.caduceus.training.trainer import Trainer
from src.caduceus.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Caduceus")
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
    metric, num_classes = info["metric"], info["num_classes"]
    benchmark = cfg.data.get("benchmark", info["benchmark"])
    max_len = cfg.data.get("max_len") or info["max_len"]
    variant = cfg.model.get("variant", "ph").lower()

    tokenizer = CharTokenizer(cfg.model["model_name"], max_len=max_len)
    te_seqs, te_labels, _ = load_split(benchmark, task, "test", cfg.data["data_dir"])
    val_ds = GenomicDataset(te_seqs, te_labels, rc_aug=False)

    model = CaduceusClassifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        variant=variant, device=str(device),
        rc_complement_ids=tokenizer.complement_id_map(),
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    collate = collate_factory(tokenizer, max_len)
    loader = DataLoader(val_ds, batch_size=cfg.training.get("batch_size", 128),
                        shuffle=False, collate_fn=collate)
    conjoin = bool(cfg.evaluation.get("conjoin", variant == "ph"))
    trainer = Trainer(model, loader, loader, cfg.training, device, metric, conjoin=conjoin)
    print(f"[eval] {trainer.evaluate(loader)}")


if __name__ == "__main__":
    main()
