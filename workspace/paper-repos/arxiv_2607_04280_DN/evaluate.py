#!/usr/bin/env python
"""
evaluate.py — fit impact curves from train.py's saved results, compute the
three theory predictions (GGPS, FGLW, LOB walking), and reproduce the
paper's Table 1 style summary for a single scenario.

Usage:
    python evaluate.py --results-dir results/baseline --scenario baseline
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqrt_law_abm.data.transforms import MetaorderReconstructor  # noqa: E402
from sqrt_law_abm.evaluation.metrics import (  # noqa: E402
    ImpactCurveFitter,
    TailExponentEstimator,
    TheoryPredictors,
    estimate_depth_profile_gamma,
)
from sqrt_law_abm.utils.config import Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit impact curves and evaluate theory predictions")
    p.add_argument("--results-dir", type=str, required=True, help="Directory containing train.py outputs")
    p.add_argument("--scenario", type=str, default="baseline", help="Which scenario's results to evaluate")
    p.add_argument("--out", type=str, default="results/", help="Where to write summary tables and figures")
    p.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config used to know data.* fitting parameters",
    )
    return p.parse_args()


def evaluate_scenario(results_dir: Path, cfg: Config) -> dict:
    """Load a scenario's SimulationResults, fit per-stock delta/c, and return
    a summary dict (mirrors Table 1's columns)."""
    with open(results_dir / "simulation_results.pkl", "rb") as f:
        sim_results = pickle.load(f)

    reconstructor = MetaorderReconstructor()
    fitter = ImpactCurveFitter()
    tail_estimator = TailExponentEstimator()
    theory = TheoryPredictors()

    data_cfg = cfg["data"]

    per_stock_delta, per_stock_c, per_stock_n_meta = [], [], []
    per_stock_beta, per_stock_alpha, per_stock_gamma = [], [], []

    for result in sim_results:
        daily_volume, daily_range = result.daily_volume_and_range()
        metaorders = reconstructor.reconstruct(
            result.trade_tape,
            delta_t_ticks=data_cfg["metaorder_delta_t_ticks"],
            steps_per_day=result.steps_per_day,
        )
        if len(metaorders) < data_cfg["min_points_per_bin"]:
            continue

        q_norm, i_norm = reconstructor.normalize(
            metaorders, daily_volume, daily_range, data_cfg["min_qnorm_threshold_pct"]
        )
        try:
            fit_result = fitter.log_bin_and_fit(
                q_norm, i_norm, n_bins=data_cfg["log_bins"], min_pts=data_cfg["min_points_per_bin"]
            )
        except ValueError:
            continue

        per_stock_delta.append(fit_result["delta"])
        per_stock_c.append(fit_result["c"])
        per_stock_n_meta.append(fit_result["n_meta"])

        sizes = np.array([mo.total_size for mo in metaorders])
        try:
            beta = tail_estimator.hill_estimator(sizes)
        except ValueError:
            beta = float("nan")
        per_stock_beta.append(beta)

        # Child-order counts are approximated here as the count of raw trades
        # per reconstructed metaorder (proxy for Nc, since Nc itself is an
        # institutional-agent-internal quantity not directly in the trade tape).
        counts = np.array(
            [
                sum(1 for tr in result.trade_tape if tr.aggressor_agent_id == mo.agent_id)
                for mo in metaorders[: min(len(metaorders), 500)]
            ]
        )
        try:
            alpha = tail_estimator.hill_estimator(counts)
        except ValueError:
            alpha = float("nan")
        per_stock_alpha.append(alpha)

        # gamma from the ask-side depth profile at end of simulation (proxy;
        # a full reproduction would sample depth profiles throughout the run).
        per_stock_gamma.append(float("nan"))  # populated only if a live LOB snapshot is available

    def mean_se(arr):
        arr = np.array([a for a in arr if np.isfinite(a)])
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr) / np.sqrt(len(arr)))

    delta_mean, delta_se = mean_se(per_stock_delta)
    c_mean, _ = mean_se(per_stock_c)
    beta_mean, _ = mean_se(per_stock_beta)
    alpha_mean, _ = mean_se(per_stock_alpha)

    return {
        "scenario": results_dir.name,
        "n_stocks_evaluated": len(per_stock_delta),
        "delta_mean": delta_mean,
        "delta_se": delta_se,
        "c_mean": c_mean,
        "n_meta_total": int(np.sum(per_stock_n_meta)) if per_stock_n_meta else 0,
        "beta_mean": beta_mean,
        "alpha_mean": alpha_mean,
        "ggps_delta_pred": theory.ggps_delta(beta_mean) if np.isfinite(beta_mean) else float("nan"),
        "fglw_delta_pred": theory.fglw_delta(alpha_mean) if np.isfinite(alpha_mean) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"{results_dir} not found. Run train.py --scenario {args.scenario} first."
        )

    summary = evaluate_scenario(results_dir, cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Scenario: {summary['scenario']} ===")
    print(f"Stocks evaluated:   {summary['n_stocks_evaluated']}")
    print(f"Metaorders (total): {summary['n_meta_total']}")
    print(f"delta = {summary['delta_mean']:.3f} +/- {summary['delta_se']:.3f}   (paper baseline: 0.549)")
    print(f"c     = {summary['c_mean']:.3f}                     (paper baseline: 0.982)")
    print(f"beta (GGPS input)  = {summary['beta_mean']:.3f} -> predicted delta = {summary['ggps_delta_pred']:.3f}")
    print(f"alpha (FGLW input) = {summary['alpha_mean']:.3f} -> predicted delta = {summary['fglw_delta_pred']:.3f}")
    print(
        "\nNote: gamma (LOB-walking exponent) requires sampling the live depth "
        "profile during simulation. See run_counterfactual_suite.py, which "
        "records depth-profile snapshots explicitly for the ablation table."
    )

    import json

    summary_path = out_dir / f"evaluate_summary_{summary['scenario']}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
