#!/usr/bin/env python
"""Classify a single DNA sequence with HyenaDNA."""
from __future__ import annotations

import argparse

import torch

from src.hyenadna.data.tokenizer import CharTokenizer
from src.hyenadna.models.pretrained import HyenaDNAClassifier
from src.hyenadna.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyenaDNA single-sequence inference")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--sequence", required=True, help="raw DNA string (A/C/G/T/N)")
    p.add_argument("--checkpoint", default=None, help="optional finetuned checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.hardware.get("device", "auto"))

    tokenizer = CharTokenizer(max_len=cfg.data.get("max_len", 1024))
    ids = torch.tensor([tokenizer.encode(args.sequence)], dtype=torch.long, device=device)

    model = HyenaDNAClassifier.from_pretrained(
        variant=cfg.model["variant"],
        num_classes=cfg.model.get("num_classes", 2),
        d_model=cfg.model.get("d_model", 128),
        pool=cfg.model.get("pool", "mean"),
        device=str(device),
    )
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    pred = int(probs.argmax())
    print(f"[inference] predicted class={pred} | probs={probs.tolist()}")


if __name__ == "__main__":
    main()
