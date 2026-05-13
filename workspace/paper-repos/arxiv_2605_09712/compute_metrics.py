"""
compute_metrics.py
==================
Compute risk-adjusted metrics from pre-computed loss files.

Usage:
    python compute_metrics.py \\
        --losses-dir results/losses/ \\
        --benchmark ar4 \\
        --horizon 1 \\
        --output results/metrics_table.csv

Paper: "Quantifying the Risk-Return Tradeoff in Forecasting"
Philippe Goulet Coulombe, arXiv: 2605.09712
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forecast_risk.evaluation.report import RiskAdjustedReport
from forecast_risk.utils.config import set_seed


def load_losses(losses_dir: str, horizon: int) -> dict[str, np.ndarray]:
    """
    Load loss arrays from .npy files.
    Expected naming: {model_name}_h{horizon}.npy
    """
    losses_dir = Path(losses_dir)
    losses = {}
    pattern = f"*_h{horizon}.npy"
    files = list(losses_dir.glob(pattern))
    if not files:
        print(f"[ERROR] No loss files matching '{pattern}' in {losses_dir}")
        sys.exit(1)
    for f in files:
        name = f.stem.replace(f"_h{horizon}", "")
        losses[name] = np.load(f)
        print(f"  Loaded {name}: {len(losses[name])} periods")
    return losses


def main():
    parser = argparse.ArgumentParser(
        description="Compute risk-adjusted forecast evaluation metrics"
    )
    parser.add_argument("--losses-dir", default="results/losses/",
                        help="Directory containing .npy loss files")
    parser.add_argument("--benchmark", default="ar4",
                        help="Benchmark model key (must be present in losses-dir)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon (1, 2, or 4)")
    parser.add_argument("--output", default="results/metrics_table.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  forecast_risk — Risk-Adjusted Metrics")
    print(f"  Horizon: h={args.horizon} | Benchmark: {args.benchmark}")
    print(f"{'='*55}\n")

    # Load losses
    print(f"Loading losses from: {args.losses_dir}")
    losses = load_losses(args.losses_dir, args.horizon)

    if args.benchmark not in losses:
        print(f"[ERROR] Benchmark '{args.benchmark}' not found in losses.")
        print(f"Available: {list(losses.keys())}")
        sys.exit(1)

    # Generate report
    reporter = RiskAdjustedReport(horizon=args.horizon)
    df = reporter.generate(losses, benchmark_key=args.benchmark)

    print("\nRisk-Adjusted Performance Table:")
    print(df.round(3).to_string())

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")

    # Also save LaTeX
    latex_path = out_path.with_suffix(".tex")
    latex_path.write_text(reporter.to_latex(df, caption=f"Risk-Adjusted Metrics (h={args.horizon})"))
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
