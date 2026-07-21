#!/usr/bin/env python3
"""
run_timeline_forecast.py -- Run the systemic Monte-Carlo break-year forecast
(blending physics and survey estimators) and reproduce Figure 2 and its
cumulative-probability headline numbers (arXiv:2606.14484, Section 3.2).

Example:
    python run_timeline_forecast.py --config configs/config.yaml --sensitivity-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_horizon.timeline import PhysicsBasedEstimator, SurveyBasedEstimator, SystemicForecastModel
from quantum_horizon.utils import load_config, plot_figure2_forecast, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--n-draws", type=int, default=None, help="Number of Monte-Carlo draws")
    parser.add_argument("--survey-weight", type=float, default=None, help="Survey-vs-physics blend weight")
    parser.add_argument("--sensitivity-sweep", action="store_true", help="Also run the 0.25-0.75 weight sweep")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.hardware["seed"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_draws = args.n_draws or cfg.model["monte_carlo_n_draws"]
    survey_weight = args.survey_weight if args.survey_weight is not None else cfg.model["survey_weight_default"]

    physics = PhysicsBasedEstimator(
        q0_2026=cfg.model["physics_calibration_2026_physical_qubits"],
        r0_2026=cfg.model["physics_calibration_2026_physical_qubits_required"],
        doubling_time_range=tuple(cfg.model["hardware_doubling_time_years_range"]),
        halving_time_range=tuple(cfg.model["resource_halving_time_years_range"]),
        fault_tolerance_lag_range=tuple(cfg.model["fault_tolerance_lag_years_range"]),
    )
    survey = SurveyBasedEstimator(
        mode_year=cfg.model["survey_estimator_mode_year"],
        sigma=cfg.model["survey_estimator_sigma"],
    )
    forecast = SystemicForecastModel(physics, survey)

    print(f"Running systemic forecast: n_draws={n_draws}, survey_weight={survey_weight}")
    result = forecast.run(n_draws=n_draws, survey_weight=survey_weight, seed=cfg.hardware["seed"])

    print(f"\nMedian break-year: {result['median']:.1f}  (paper: ~2046-2047)")
    print(f"P(CRQC by 2035): {result['cdf_2035']:.1%}  (paper: ~1-in-6, ~16.7%)")
    print(f"P(CRQC by 2040): {result['cdf_2040']:.1%}  (paper: ~30%)")
    print(f"P(CRQC by 2050): {result['cdf_2050']:.1%}  (paper: ~60%)")
    print(f"80% range: {result['range_80pct'][0]:.0f}-{result['range_80pct'][1]:.0f}  (paper: ~2032-2060)")

    summary = {
        "n_draws": n_draws,
        "survey_weight": survey_weight,
        "median": result["median"],
        "cdf_2035": result["cdf_2035"],
        "cdf_2040": result["cdf_2040"],
        "cdf_2050": result["cdf_2050"],
        "range_80pct": result["range_80pct"],
    }

    if args.sensitivity_sweep:
        print("\nRunning survey-weight sensitivity sweep (paper: 0.25-0.75)...")
        sweep_df = forecast.sensitivity_sweep([0.25, 0.5, 0.75], n_draws=n_draws, seed=cfg.hardware["seed"])
        print(sweep_df.to_string(index=False))
        sweep_df.to_csv(output_dir / "timeline_sensitivity_sweep.csv", index=False)
        summary["sensitivity_sweep"] = sweep_df.to_dict(orient="records")

    with open(output_dir / "timeline_forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_figure2_forecast(
        result["survey_samples"], result["physics_samples"], result["combined_samples"],
        save_path=str(output_dir / "figure2_forecast.png"),
    )
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
