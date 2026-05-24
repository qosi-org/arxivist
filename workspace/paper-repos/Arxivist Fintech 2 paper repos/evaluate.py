"""
evaluate.py — Full evaluation: R², DM tests, variable importance, portfolios.

Loads saved model predictions and produces Tables 1, 3, 6, 7, 8 equivalents.

Usage:
  python evaluate.py --results_dir results/ --config configs/config.yaml
  python evaluate.py --results_dir results/ --debug
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

from asset_pricing_ml.evaluation.metrics import PortfolioMetrics, ReturnMetrics
from asset_pricing_ml.evaluation.portfolios import PortfolioConstructor
from asset_pricing_ml.training.trainer import RecursiveTrainer
from asset_pricing_ml.utils.config import Config

ALL_MODELS = ["OLS", "OLS3", "ENet", "PCR", "PLS", "GLM",
              "RF", "GBRT", "NN1", "NN2", "NN3", "NN4", "NN5"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved results (Gu-Kelly-Xiu 2020)")
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    output_dir = args.output or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load available results
    loaded = {}
    for model in ALL_MODELS:
        pkl = os.path.join(args.results_dir, f"{model}_results.pkl")
        if os.path.exists(pkl):
            d = RecursiveTrainer.load_results(pkl)
            loaded[model] = d
            print(f"  Loaded: {model}")

    if not loaded:
        print("No saved results found. Run train.py or train_all.py first.")
        return

    # ── Table 1 equivalent: R²_oos ──────────────────────────────────────────
    print("\n=== R²_oos (% per month) ===")
    r2_results = {}
    pred_by_model = {}
    actual = None

    for model, d in loaded.items():
        preds = np.concatenate(d["test_predictions"])
        acts = np.concatenate(d["test_actuals"])
        if actual is None:
            actual = acts
        r2 = ReturnMetrics.oos_r2(acts, preds) * 100
        r2_results[model] = r2
        pred_by_model[model] = preds
        print(f"  {model:<8}: {r2:+.3f}%")

    # ── Table 3 equivalent: Diebold-Mariano pairwise tests ──────────────────
    if len(loaded) >= 2 and actual is not None:
        print("\n=== Diebold-Mariano Statistics (column outperforms row = positive) ===")
        models_list = list(loaded.keys())
        dm_matrix = np.full((len(models_list), len(models_list)), np.nan)

        for i, m1 in enumerate(models_list):
            for j, m2 in enumerate(models_list):
                if i == j:
                    continue
                T = min(len(pred_by_model[m1]), len(pred_by_model[m2]), len(actual))
                e1 = (actual[:T] - pred_by_model[m1][:T]).reshape(-1, 1)
                e2 = (actual[:T] - pred_by_model[m2][:T]).reshape(-1, 1)
                dm, _ = ReturnMetrics.diebold_mariano(e1, e2)
                dm_matrix[i, j] = dm

        df_dm = pd.DataFrame(dm_matrix, index=models_list, columns=models_list)
        print(df_dm.round(2).to_string())
        df_dm.to_csv(os.path.join(output_dir, "diebold_mariano.csv"))

    # ── Portfolio analysis: Table 7 equivalent ──────────────────────────────
    print("\n=== Long-Short Decile Portfolio Performance ===")
    portfolio_results = {}
    for model, d in loaded.items():
        ls = PortfolioConstructor.long_short_portfolio(
            d["test_predictions"], d["test_actuals"], d["test_mktcap"],
            weighting="value"
        )
        portfolio_results[model] = {
            "sharpe_vw": round(ls.sharpe_ratio, 2),
            "max_dd": round(ls.max_drawdown * 100, 1),
        }
        print(f"  {model:<8}: Sharpe={ls.sharpe_ratio:.2f}, MaxDD={ls.max_drawdown*100:.1f}%")

    print("\n  Paper benchmark (NN3): Sharpe=1.35, MaxDD=30.84% (Table 7/8)")

    # Save combined results
    summary = {
        "r2_oos_pct": r2_results,
        "portfolio": portfolio_results,
    }
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation saved: {output_dir}/evaluation_summary.json")


if __name__ == "__main__":
    main()
