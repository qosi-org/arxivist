"""
train_all.py — Train all 13 model variants and produce Table 1 equivalent.

Runs the full 30-year recursive evaluation for every model in the paper,
then produces a comparison table equivalent to Table 1 (monthly R²_oos).

Usage:
  python train_all.py --config configs/config.yaml
  python train_all.py --config configs/config.yaml --debug
  python train_all.py --config configs/config.yaml --models NN3 RF GBRT
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

from asset_pricing_ml.evaluation.metrics import ReturnMetrics
from asset_pricing_ml.evaluation.portfolios import PortfolioConstructor
from asset_pricing_ml.training.trainer import RecursiveTrainer
from asset_pricing_ml.utils.config import Config, set_seed

ALL_MODELS = ["OLS", "OLS3", "ENet", "PCR", "PLS", "GLM",
              "RF", "GBRT", "NN1", "NN2", "NN3", "NN4", "NN5"]

# Paper Table 1 benchmark values (% per month) for comparison
PAPER_R2 = {
    "OLS":  -3.46, "OLS3": 0.16, "ENet": 0.11, "PCR": 0.26, "PLS": 0.27,
    "GLM":   0.19, "RF":   0.33, "GBRT": 0.34,
    "NN1":   0.29, "NN2":  0.31, "NN3":  0.40, "NN4": 0.35, "NN5": 0.35,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train all 13 models (Table 1 replication)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--models", nargs="+", default=None, help="Subset of models to run")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output", default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.hardware.seed)
    os.makedirs(args.output, exist_ok=True)

    models_to_run = args.models or ALL_MODELS

    # Load data once
    if args.debug:
        from asset_pricing_ml.data.dataset import SyntheticDataGenerator
        print("[DEBUG] Using synthetic data")
        gen = SyntheticDataGenerator(n_stocks=150, n_months=720, seed=42)
        data = gen.generate()
    else:
        from asset_pricing_ml.data.dataset import StockReturnDataset
        print("Loading CRSP data...")
        data = StockReturnDataset(cfg.data).load()
    print(f"Data: {data}\n")

    results = {}
    t_total = time.time()

    for model_name in models_to_run:
        print(f"{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        cfg.model.variant = model_name
        t0 = time.time()

        trainer = RecursiveTrainer(cfg, model_factory=None)
        try:
            pred, actual = trainer.run(data)
            r2 = ReturnMetrics.oos_r2(actual, pred) * 100
            sr = _compute_ls_sharpe(trainer)
            results[model_name] = {"R2_oos_pct": round(r2, 3), "LS_Sharpe": round(sr, 2)}
            print(f"  → R²_oos = {r2:.3f}%  |  L-S Sharpe = {sr:.2f}  |  {time.time()-t0:.0f}s")

            pkl = os.path.join(args.output, f"{model_name}_results.pkl")
            trainer.save(pkl)
        except Exception as e:
            results[model_name] = {"R2_oos_pct": None, "LS_Sharpe": None, "error": str(e)}
            print(f"  ✗ Error: {e}")

    # Print comparison table (mirrors Table 1 format)
    print(f"\n{'='*65}")
    print(f"Table 1 Equivalent — Monthly R²_oos (% per month)")
    print(f"{'='*65}")
    print(f"{'Model':<10} {'Ours':>10} {'Paper':>10} {'Δ':>8}")
    print(f"{'-'*40}")
    for model in models_to_run:
        r = results.get(model, {})
        ours = r.get("R2_oos_pct")
        paper = PAPER_R2.get(model, None)
        if ours is not None and paper is not None:
            delta = ours - paper
            print(f"{model:<10} {ours:>10.3f} {paper:>10.3f} {delta:>+8.3f}")
        else:
            err = r.get("error", "N/A")
            print(f"{model:<10} {'ERROR':>10}  paper={paper}  ({err[:30]})")

    print(f"\nTotal training time: {time.time()-t_total:.0f}s")

    # Save summary CSV
    df = pd.DataFrame(results).T
    df.index.name = "model"
    csv_path = os.path.join(args.output, "table1_equivalent.csv")
    df.to_csv(csv_path)
    print(f"Results table saved: {csv_path}")


def _compute_ls_sharpe(trainer: RecursiveTrainer) -> float:
    """Compute long-short Sharpe ratio from trainer results."""
    try:
        ls = PortfolioConstructor.long_short_portfolio(
            trainer.test_predictions_, trainer.test_actuals_, trainer.test_mktcap_
        )
        return ls.sharpe_ratio
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
