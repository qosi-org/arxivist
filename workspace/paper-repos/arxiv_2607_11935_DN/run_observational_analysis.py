#!/usr/bin/env python3
"""
run_observational_analysis.py -- Run the TVP-Kalman elasticity filter and
classical EWS on the NASA AIRS regional data, reproducing Table 1, Table 2,
Figure 1, and Figure 2 of arXiv:2607.11935.

Example:
    python run_observational_analysis.py --config configs/config.yaml --region all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ews_kalman.data import AIRSDataLoader
from ews_kalman.ews import ClassicalEWS
from ews_kalman.evaluation import RegionSummaryComputer
from ews_kalman.kalman import TVPKalmanFilter
from ews_kalman.utils import load_config, plot_figure1_overlay, plot_figure2_leadlag_bars, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--region", type=str, default="all", choices=["arctic", "tropics", "monsoon", "all"],
        help="Which region to analyze",
    )
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write tables/figures")
    return parser.parse_args()


def analyze_region(cfg, region_name: str) -> dict:
    """Run the full pipeline (Kalman + classical EWS + Table 1/2) for one region."""
    data_cfg = cfg.data
    loader = AIRSDataLoader()
    region = loader.load_region(
        region_name,
        data_dir="data/raw",
        use_synthetic_fallback=data_cfg["use_synthetic_fallback"],
        seed=cfg.hardware["seed"],
    )
    T, q, dates = region["T"], region["q"], region["dates"]

    kf = TVPKalmanFilter(
        R=cfg.model["kalman_R"],
        Q_diag=tuple(cfg.model["kalman_Q_diag"]),
        dt=cfg.model["kalman_dt_years"],
        mode="loglog",
    )
    beta_result = kf.estimate_beta(T, q)

    ews = ClassicalEWS()
    ar1_window = data_cfg["ar1_variance_window_months"]
    pe_window = data_cfg["permutation_entropy_window_months"]
    mi_window = data_cfg["mutual_information_window_months"]

    ar1_T = ews.rolling_ar1(T, window=ar1_window)
    ar1_q = ews.rolling_ar1(q, window=ar1_window)
    var_T = ews.rolling_variance(T, window=ar1_window)
    var_q = ews.rolling_variance(q, window=ar1_window)
    perm_ent_T = ews.rolling_permutation_entropy(
        T, embedding_dim=data_cfg["permutation_entropy_embedding_dim"], window=pe_window
    )
    mi = ews.rolling_mutual_information(
        T, q, window=mi_window, n_neighbors=data_cfg["mutual_information_n_neighbors"]
    )

    computer = RegionSummaryComputer()
    table1_row = computer.compute_table1_row(
        beta_result["beta"], ar1_T, mi, beta_result["beta_double_prime"]
    )

    classical_signals = {
        "AR1_T": ar1_T, "AR1_q": ar1_q, "Var_T": var_T, "Var_q": var_q,
        "MI": mi, "PermEnt_T": perm_ent_T,
    }
    table2_row = computer.compute_table2_row(
        beta_result, classical_signals, max_lag=cfg.evaluation["max_cross_correlation_lag_months"]
    )

    return {
        "region": region_name,
        "table1": table1_row,
        "table2": table2_row,
        "dates": dates,
        "beta": beta_result["beta"],
        "ar1_T": ar1_T,
        "mi": mi,
        "perm_ent_T": perm_ent_T,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = cfg.data["regions"] if args.region == "all" else [args.region]

    all_results = {}
    for region_name in regions:
        print(f"\n=== Analyzing region: {region_name} ===")
        result = analyze_region(cfg, region_name)
        all_results[region_name] = result

        t1 = result["table1"]
        print(
            f"N={t1['N']}  |beta|={t1['abs_beta_mean']:.3f}  sigma_beta={t1['sigma_beta']:.3f}  "
            f"r(beta,AR1)={t1['r_beta_ar1']:+.2f} (p={t1['r_beta_ar1_pvalue']:.2f})  "
            f"r(beta,MI)={t1['r_beta_mi']:+.2f} (p={t1['r_beta_mi_pvalue']:.2f})  "
            f"transitions={t1['n_transitions']}"
        )

    # Save Table 1 as JSON
    table1_summary = {name: r["table1"] for name, r in all_results.items()}
    with open(output_dir / "table1_region_summary.json", "w") as f:
        json.dump(table1_summary, f, indent=2)

    # Save Table 2 as JSON (only the 'beta' derivative row, matching the paper's Table 2)
    table2_summary = {name: r["table2"]["beta"] for name, r in all_results.items()}
    with open(output_dir / "table2_leadlag_summary.json", "w") as f:
        json.dump(table2_summary, f, indent=2)

    print(f"\nTables written to {output_dir}/")

    # Figures (best-effort; skip if region data shapes are inconsistent for plotting)
    try:
        min_len = min(len(r["beta"]) for r in all_results.values())
        fig1_data = {
            name: {
                "dates": r["dates"][-min_len:],
                "beta": r["beta"][-min_len:],
                "ar1_T": np.pad(r["ar1_T"], (min_len - len(r["ar1_T"]), 0), mode="edge")[-min_len:],
                "mi": np.pad(r["mi"], (min_len - len(r["mi"]), 0), mode="edge")[-min_len:],
                "perm_ent_T": np.pad(r["perm_ent_T"], (min_len - len(r["perm_ent_T"]), 0), mode="edge")[-min_len:],
            }
            for name, r in all_results.items()
        }
        plot_figure1_overlay(fig1_data, save_path=str(output_dir / "figure1_overlay.png"))
        plot_figure2_leadlag_bars(table2_summary, save_path=str(output_dir / "figure2_leadlag_bars.png"))
        print(f"Figures written to {output_dir}/")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[WARN] Figure generation skipped due to: {exc}")


if __name__ == "__main__":
    main()
