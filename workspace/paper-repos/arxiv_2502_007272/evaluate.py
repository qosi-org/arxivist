#!/usr/bin/env python
"""Evaluate a fine-tuned GENERator checkpoint on a benchmark test split."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generator.data.benchmarks import GenomicDataset, collate_factory, load_split
from src.generator.data.tokenizer import GenomicTokenizer
from src.generator.evaluation.metrics import compute_metrics
from src.generator.models.classifier import GENERatorClassifier
from src.generator.utils.config import load_config, resolve_device, seed_everything, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GENERator")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.hardware.get("seed", 42), cfg.hardware.get("deterministic", False))
    device = resolve_device(cfg.hardware.get("device", "auto"))

    task = cfg.data["task"]
    info = task_info(task)
    metric, num_classes = info["metric"], info["num_classes"]
    benchmark = cfg.data.get("benchmark", info["benchmark"])
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = GenomicTokenizer(cfg.model["model_name"], max_len=max_len)
    te_seqs, te_labels, _ = load_split(benchmark, task, "test", cfg.data["data_dir"])
    test_ds = GenomicDataset(te_seqs, te_labels)

    model = GENERatorClassifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=num_classes,
        device=str(device), load_in_8bit=cfg.model.get("load_in_8bit", False),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    collate = collate_factory(tokenizer, max_len)
    loader = DataLoader(test_ds, batch_size=cfg.training.get("batch_size", 64), shuffle=False, collate_fn=collate)
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())

    metrics = compute_metrics(preds, labels, metric)
    print(f"[eval] {task} | {metrics}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{task}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": task, "metrics": metrics}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
