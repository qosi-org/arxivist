#!/usr/bin/env python
"""Classify a single DNA sequence with DNABERT-2."""
from __future__ import annotations

import argparse

import torch

from src.dnabert2.data.tokenizer import DNATokenizer
from src.dnabert2.models.classifier import DNABERT2Classifier
from src.dnabert2.utils.config import load_config, resolve_device, task_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DNABERT-2 single-sequence inference")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--sequence", required=True, help="raw DNA string (A/C/G/T)")
    p.add_argument("--checkpoint", default=None, help="optional finetuned checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.hardware.get("device", "auto"))
    info = task_info(cfg.data["task"])
    max_len = cfg.data.get("max_len") or info["max_len"]

    tokenizer = DNATokenizer(cfg.model["model_name"], max_len=max_len)
    enc = tokenizer.encode_batch([args.sequence], max_len=max_len)

    model = DNABERT2Classifier.from_pretrained(
        model_name=cfg.model["model_name"], num_classes=info["num_classes"],
        pool=cfg.model.get("pool", "mean"), device=str(device),
        attention_dropout=cfg.model.get("attention_dropout", 0.1),
    )
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    print(f"[inference] predicted class={int(probs.argmax())} | probs={probs.tolist()}")


if __name__ == "__main__":
    main()
