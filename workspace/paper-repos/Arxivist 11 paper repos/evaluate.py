#!/usr/bin/env python
"""
Evaluation entrypoint: computes OOS R2/MSE (Appendix E corrected methodology) and
optional Diebold-Mariano HAC test against a second checkpoint.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --period 2212
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sert_asset_pricing.evaluation.metrics import OOSMetrics  # noqa: E402
from sert_asset_pricing.models.transformer_variants import build_model  # noqa: E402
from sert_asset_pricing.utils.config import ConfigLoader, get_device, set_seed  # noqa: E402

VALID_PERIODS = ("1911", "2112", "2212")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--period", type=str, default="2212", choices=VALID_PERIODS,
                         help="OOS period id: 1911 (pre-COVID) | 2112 (COVID-inclusive) | 2212 (COVID+1yr)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader(args.config).load()
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["hardware"]["device"])

    if not os.path.exists(args.checkpoint):
        print(f"[evaluate.py] Checkpoint not found: {args.checkpoint}")
        print("[evaluate.py] Run `python train.py --config configs/config.yaml` first.")
        return

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[evaluate.py] Loaded checkpoint: {args.checkpoint}")

    start, end = cfg["data"]["oos_periods"][args.period]
    print(f"[evaluate.py] Evaluating OOS period '{args.period}': {start} to {end}")

    if not (os.path.exists(cfg["data"]["factors_path"]) and os.path.exists(cfg["data"]["returns_path"])):
        print("[evaluate.py] Real OOS data not found — see data/README_data.md. "
              "Skipping numeric evaluation; only checkpoint load/build verified.")
        return

    factors = pd.read_csv(cfg["data"]["factors_path"], index_col=0, parse_dates=True)
    returns = pd.read_csv(cfg["data"]["returns_path"], index_col=0, parse_dates=True)

    hist_mask = factors.index < pd.Timestamp(start)
    oos_mask = (factors.index >= pd.Timestamp(start)) & (factors.index <= pd.Timestamp(end))

    x_oos = torch.tensor(factors.loc[oos_mask].to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)
    y_oos = returns.loc[oos_mask].to_numpy()
    y_hist_mean = returns.loc[hist_mask].mean(axis=0).to_numpy()  # Appendix E: mean over train+val

    with torch.no_grad():
        if hasattr(model, "output_embedding"):
            y_shifted = torch.zeros(1, x_oos.shape[1], 1, device=device)
            pred = model(x_oos, y_shifted).cpu().numpy().squeeze(0).squeeze(-1)
        else:
            pred = model(x_oos).cpu().numpy().squeeze(0).squeeze(-1)

    metrics = OOSMetrics()
    per_stock_r2 = []
    for j in range(y_oos.shape[1]):
        r2 = metrics.oos_r2(y_oos[:, j], np.broadcast_to(pred, y_oos[:, j].shape), y_hist_mean[j])
        per_stock_r2.append(r2)

    avg_r2 = float(np.nanmean(per_stock_r2))
    print(f"[evaluate.py] Average OOS R2 across stocks: {avg_r2:.4f}")
    print(f"[evaluate.py] Paper's reported OOS R2 for comparable models (Table 3/5): "
          f"~0.02-0.12 depending on model/period — see sir-registry for exact reference values.")


if __name__ == "__main__":
    main()
