"""
train.py — Main training script for Gu, Kelly, Xiu (2020).

Runs the full 30-year recursive out-of-sample evaluation for one model variant.
Saves predictions and fitted models for evaluation.

Usage:
  python train.py --config configs/config.yaml --model NN3
  python train.py --config configs/config.yaml --model RF --device cpu
  python train.py --config configs/config.yaml --debug        # synthetic data, fast
  python train.py --config configs/config.yaml --dry-run      # validate setup only

To replicate Table 1 (all 13 models), use train_all.py instead.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from asset_pricing_ml.evaluation.metrics import ReturnMetrics
from asset_pricing_ml.evaluation.portfolios import PortfolioConstructor, PortfolioMetrics
from asset_pricing_ml.training.trainer import RecursiveTrainer
from asset_pricing_ml.utils.config import Config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train asset pricing ML model (Gu-Kelly-Xiu 2020)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None, help="Override config model variant")
    p.add_argument("--device", default=None, help="Override config device (cpu/cuda)")
    p.add_argument("--output", default=None, help="Output directory for results")
    p.add_argument("--debug", action="store_true", help="Synthetic data, fast run")
    p.add_argument("--dry-run", action="store_true", help="Validate setup, no training")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.model:
        cfg.model.variant = args.model
    if args.device:
        cfg.hardware.device = args.device
    if args.output:
        cfg.eval.output_dir = args.output

    set_seed(cfg.hardware.seed, cfg.hardware.deterministic)
    print(f"Config: {cfg}")

    if args.dry_run:
        print("\n[DRY RUN] Config validated. All imports successful.")
        print(f"  Model variant: {cfg.model.variant}")
        print(f"  Features: {cfg.data.total_features}")
        print(f"  Test period: {cfg.data.val_end_year+1}–{cfg.data.test_end_year}")
        return

    # Load data
    if args.debug:
        print("\n[DEBUG] Using synthetic data generator...")
        from asset_pricing_ml.data.dataset import SyntheticDataGenerator
        gen = SyntheticDataGenerator(n_stocks=200, n_months=720, seed=cfg.hardware.seed)
        data = gen.generate()
        print(f"  {data}")
    else:
        from asset_pricing_ml.data.dataset import StockReturnDataset
        print("\nLoading CRSP data...")
        ds = StockReturnDataset(cfg.data)
        data = ds.load()
        print(f"  {data}")

    # Run recursive training
    os.makedirs(cfg.eval.output_dir, exist_ok=True)
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    trainer = RecursiveTrainer(cfg, model_factory=None)
    t0 = time.time()
    all_pred, all_actual = trainer.run(data)
    print(f"\nTraining complete in {time.time()-t0:.1f}s")

    # Compute overall test R²
    r2 = ReturnMetrics.oos_r2(all_actual, all_pred)
    print(f"\nFinal Test R²_oos: {r2*100:.4f}% per month")
    print(f"  Paper benchmark (NN3): 0.40% per month (Table 1)")

    # Save results
    results_path = os.path.join(cfg.eval.output_dir, f"{cfg.model.variant}_results.pkl")
    trainer.save(results_path)
    print(f"\nResults saved: {results_path}")

    # Quick portfolio analysis
    if len(trainer.test_predictions_) > 0:
        r_hat_series = trainer.test_predictions_
        r_actual_series = trainer.test_actuals_
        mktcap_series = trainer.test_mktcap_

        ls_port = PortfolioConstructor.long_short_portfolio(
            r_hat_series, r_actual_series, mktcap_series, weighting="value"
        )
        print(f"\nLong-Short Decile Portfolio (value-weighted):")
        print(f"  Annualized Sharpe ratio: {ls_port.sharpe_ratio:.2f}")
        print(f"  Max drawdown:            {ls_port.max_drawdown*100:.1f}%")
        print(f"  Paper benchmark (NN3):   Sharpe = 1.35 (Table 7)")


if __name__ == "__main__":
    main()
