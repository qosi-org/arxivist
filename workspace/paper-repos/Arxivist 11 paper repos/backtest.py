#!/usr/bin/env python
"""
Backtesting entrypoint: reproduces Section 5.3's sign-signal / softmax-filtered
trading strategy backtests (equal- and value-weighted, static/dynamic transaction cost).

Usage:
    python backtest.py --config configs/config.yaml --checkpoint checkpoints/best.pt \
        --weighting equal --tc-mode static
    python backtest.py --config configs/config.yaml --checkpoint checkpoints/best.pt \
        --weighting value --tc-mode dynamic --softmax-filter
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sert_asset_pricing.evaluation.backtest import Backtester  # noqa: E402
from sert_asset_pricing.models.transformer_variants import build_model  # noqa: E402
from sert_asset_pricing.utils.config import ConfigLoader, get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trading-strategy backtests.")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--weighting", type=str, default="equal", choices=["equal", "value"])
    parser.add_argument("--tc-mode", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--softmax-filter", action="store_true",
                         help="apply the paper's softmax trading-signal filter (Section 5.3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigLoader(args.config).load()
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["hardware"]["device"])

    if not os.path.exists(args.checkpoint):
        print(f"[backtest.py] Checkpoint not found: {args.checkpoint}. Run train.py first.")
        return

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if not (os.path.exists(cfg["data"]["factors_path"]) and os.path.exists(cfg["data"]["returns_path"])):
        print("[backtest.py] Real data not found — see data/README_data.md.")
        print("[backtest.py] Running a synthetic smoke test instead.")
        rng = np.random.default_rng(cfg["training"]["seed"])
        preds = rng.normal(0, 0.03, size=(60, 20))
        actuals = rng.normal(0, 0.05, size=(60, 20))
        market_caps = rng.uniform(1e9, 1e11, size=(60, 20))
    else:
        factors = pd.read_csv(cfg["data"]["factors_path"], index_col=0, parse_dates=True)
        returns = pd.read_csv(cfg["data"]["returns_path"], index_col=0, parse_dates=True)
        x = torch.tensor(factors.to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "output_embedding"):
                y_shifted = torch.zeros(1, x.shape[1], 1, device=device)
                pred_scalar = model(x, y_shifted).cpu().numpy().squeeze(0).squeeze(-1)
            else:
                pred_scalar = model(x).cpu().numpy().squeeze(0).squeeze(-1)
        actuals = returns.to_numpy()
        preds = np.tile(pred_scalar[:, None], (1, actuals.shape[1]))
        market_caps = None
        if args.weighting == "value":
            print("[backtest.py] WARNING: market cap data not provided; falling back to equal weighting.")
            args.weighting = "equal"

    backtester = Backtester(
        transaction_cost_static_bps=cfg["evaluation"]["transaction_cost_static_bps"],
        transaction_cost_dynamic_bps=cfg["evaluation"]["transaction_cost_dynamic_bps"],
        softmax_filter_pct=cfg["evaluation"]["softmax_filter_pct"],
    )
    result = backtester.run(
        preds=preds, actuals=actuals, weighting=args.weighting, tc_mode=args.tc_mode,
        softmax_filter=args.softmax_filter, market_caps=market_caps,
    )

    print(f"[backtest.py] weighting={args.weighting}, tc_mode={args.tc_mode}, "
          f"softmax_filter={args.softmax_filter}")
    print(f"  Annualized Return (AR): {result['AR']:.4f}")
    print(f"  Sharpe Ratio (SR):      {result['SR']:.4f}  (Ann.SR: {result['Ann_SR']:.4f})")
    print(f"  Sortino Ratio (SO):     {result['SO']:.4f}  (Ann.SO: {result['Ann_SO']:.4f})")
    print(f"  Max Drawdown (MDD):     {result['MDD']:.4f}")
    print(f"  Mean turnover:          {result['turnover']:.4f}")


if __name__ == "__main__":
    main()
