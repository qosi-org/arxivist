#!/usr/bin/env python
"""Evaluate a fine-tuned DNABERT-2 checkpoint on a GUE test split."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dnabert2.data.gue import GUEDataset, collate_factory, load_gue_split
from src.dnabert2.data.tokenizer import DNATokenizer
from src.dnabert2.evaluation.metrics import compute_metrics
from src.dnabert2.models.classifier import DNABERT2Classifier
from src.dnabert2.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DNABERT-2 on GUE")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42), cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    task, subset = cfg.data["task"], cfg.data["subset"]
    info = task_info(task)
    metric, num_classes = info["metric"], info["num_classes"]
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = DNATokenizer(cfg.model["model_name"], max_len=max_len)
    te_seqs, te_labels = load_gue_split(task, subset, cfg.evaluation.get("split", "test"), cfg.data["data_dir"])
    test_ds = GUEDataset(te_seqs, te_labels)

    model = DNABERT2Classifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        pool=cfg.model.get("pool", "mean"), device=str(device),
        attention_dropout=cfg.model.get("attention_dropout", 0.1),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    collate = collate_factory(tokenizer, max_len)
    loader = DataLoader(test_ds, batch_size=cfg.training.get("batch_size", 32), shuffle=False, collate_fn=collate)
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())

    metrics = compute_metrics(preds, labels, metric)
    print(f"[eval] {task}/{subset} | {metrics}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{task}_{subset}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": task, "subset": subset, "metrics": metrics}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
