#!/usr/bin/env python
"""
Single-window inference entrypoint: predicts next-period excess returns for all stocks
given one 102-month (train_window) factor window.

Usage:
    python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt \
        --input-window data/raw/sample_window.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sert_asset_pricing.models.transformer_variants import build_model  # noqa: E402
from sert_asset_pricing.utils.config import ConfigLoader, get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-window inference.")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--input-window", type=str, required=True,
                         help="path to a single train_window-length factor CSV [T, num_factors]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader(args.config).load()
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["hardware"]["device"])

    if not os.path.exists(args.checkpoint):
        print(f"[inference.py] Checkpoint not found: {args.checkpoint}. Run train.py first.")
        return
    if not os.path.exists(args.input_window):
        print(f"[inference.py] Input window not found: {args.input_window}")
        return

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    window = pd.read_csv(args.input_window, index_col=0, parse_dates=True)
    x = torch.tensor(window.to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, "output_embedding"):
            y_shifted = torch.zeros(1, x.shape[1], 1, device=device)
            pred = model(x, y_shifted).cpu().numpy().squeeze(0)
        else:
            pred = model(x).cpu().numpy().squeeze(0)

    out_df = pd.DataFrame(pred, index=window.index, columns=["predicted_excess_return"])
    print(out_df.tail(5).to_string())
    out_path = "predictions.csv"
    out_df.to_csv(out_path)
    print(f"[inference.py] Full predictions written to {out_path}")


if __name__ == "__main__":
    main()
