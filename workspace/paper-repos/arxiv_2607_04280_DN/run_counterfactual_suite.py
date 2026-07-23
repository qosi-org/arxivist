#!/usr/bin/env python
"""
run_counterfactual_suite.py — paper-specific script (Section 4.3) that runs
all 8 scenarios across the 20 counterfactual stocks and reproduces Table 1
and Figure 7 directly.

Usage:
    python run_counterfactual_suite.py --config configs/config.yaml
    python run_counterfactual_suite.py --config configs/config.yaml --out results/counterfactual/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqrt_law_abm.data.dataset import StockParameterSampler  # noqa: E402
from sqrt_law_abm.data.transforms import MetaorderReconstructor  # noqa: E402
from sqrt_law_abm.evaluation.metrics import ImpactCurveFitter, TailExponentEstimator, TheoryPredictors  # noqa: E402
from sqrt_law_abm.training.trainer import BatchSimulationRunner  # noqa: E402
from sqrt_law_abm.utils.config import Config, set_global_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table 1 / Figure 7 counterfactual ablation suite")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p.add_argument("--out", type=str, default="results/counterfactual/", help="Output directory")
    p.add_argument("--debug", action="store_true", help="Small/fast smoke-test run")
    return p.parse_args()


def fit_scenario_delta(sim_results, cfg) -> tuple[float, float, float, float]:
    """Fit per-stock delta for one scenario's results; return (mean, se, beta_mean, alpha_mean)."""
    reconstructor = MetaorderReconstructor()
    fitter = ImpactCurveFitter()
    tail_estimator = TailExponentEstimator()
    data_cfg = cfg["data"]

    deltas, betas, alphas = [], [], []
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
        deltas.append(fit_result["delta"])

        sizes = np.array([mo.total_size for mo in metaorders])
        try:
            betas.append(tail_estimator.hill_estimator(sizes))
        except ValueError:
            pass

    def mean_se(arr):
        arr = np.array([a for a in arr if np.isfinite(a)])
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr) / np.sqrt(len(arr)))

    d_mean, d_se = mean_se(deltas)
    b_mean, _ = mean_se(betas)
    return d_mean, d_se, b_mean, float("nan")


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.debug:
        cfg = cfg.apply_debug_overrides()

    seed = cfg.get("seed", default=42)
    set_global_seed(seed)

    sampler = StockParameterSampler(cfg["model"], cfg["simulation"])
    n_stocks = cfg["simulation"]["n_stocks_counterfactual"]
    stock_configs = sampler.sample(n_stocks=n_stocks, seed=seed)

    scenarios = cfg["evaluation"]["scenarios"]
    runner = BatchSimulationRunner(cfg["model"], cfg["simulation"])

    print(f"Running counterfactual suite: {len(scenarios)} scenarios x {n_stocks} stocks each")
    suite_results = runner.run_counterfactual_suite(
        stock_configs, scenarios, n_jobs=cfg.get("hardware", "num_workers", default=-1)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    theory = TheoryPredictors()
    rows = []
    print("\n" + "=" * 78)
    print(f"{'Scenario':<16}{'delta':>10}{'SE':>8}{'beta':>8}{'GGPS pred':>12}")
    print("=" * 78)
    for scenario, sim_results in suite_results.items():
        d_mean, d_se, b_mean, _ = fit_scenario_delta(sim_results, cfg)
        ggps_pred = theory.ggps_delta(b_mean) if np.isfinite(b_mean) else float("nan")
        rows.append(
            {"scenario": scenario, "delta": d_mean, "delta_se": d_se, "beta": b_mean, "ggps_pred": ggps_pred}
        )
        print(f"{scenario:<16}{d_mean:>10.3f}{d_se:>8.3f}{b_mean:>8.3f}{ggps_pred:>12.3f}")
    print("=" * 78)
    print(f"Paper's Table 1 baseline: delta=0.549, no_splitting=0.324, no_hft=0.386")

    import csv

    table_path = out_dir / "table1_reproduction.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "delta", "delta_se", "beta", "ggps_pred"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved Table 1 reproduction to: {table_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        scenario_names = [r["scenario"] for r in rows]
        delta_vals = [r["delta"] for r in rows]
        delta_errs = [r["delta_se"] for r in rows]
        ax.bar(scenario_names, delta_vals, yerr=delta_errs, capsize=4)
        ax.axhline(0.5, linestyle="--", color="gray", label="delta = 1/2 (SRL)")
        ax.set_ylabel("delta")
        ax.set_title("Figure 7 reproduction: fitted delta per counterfactual scenario")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig_path = out_dir / "figure7_reproduction.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Saved Figure 7 reproduction to: {fig_path}")
    except ImportError:
        print("matplotlib not available; skipped Figure 7 reproduction plot.")


if __name__ == "__main__":
    main()
