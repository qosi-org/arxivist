#!/usr/bin/env python
"""Evaluate a fine-tuned HyenaDNA checkpoint on the test split."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hyenadna.data.dataset import build_dataset
from src.hyenadna.data.tokenizer import CharTokenizer
from src.hyenadna.evaluation.metrics import compute_metrics
from src.hyenadna.models.pretrained import HyenaDNAClassifier
from src.hyenadna.utils.config import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HyenaDNA")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True, help="path to finetuned checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42), cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    tokenizer = CharTokenizer(max_len=cfg.data.get("max_len", 1024))
    test_ds, n_cls = build_dataset(cfg.data, cfg.evaluation.get("split", "test"), tokenizer)

    model = HyenaDNAClassifier.from_pretrained(
        variant=cfg.model["variant"],
        num_classes=cfg.model.get("num_classes") or n_cls,
        d_model=cfg.model.get("d_model", 128),
        pool=cfg.model.get("pool", "mean"),
        device=str(device),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(test_ds, batch_size=cfg.training.get("batch_size", 32), shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.tolist())

    metrics = compute_metrics(preds, labels, cfg.evaluation.get("metrics", ["accuracy"]))
    print(f"[eval] {cfg.data['dataset']} | {metrics}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{cfg.data['dataset']}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": cfg.data["dataset"], "metrics": metrics}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
