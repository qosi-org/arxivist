"""
run_evaluation.py
==================
Full expanding-window forecast evaluation pipeline.

Usage:
    python run_evaluation.py --config configs/default_config.yaml

Paper: Section 3 — Application 1: Predictive Personalities & Macro Forecasting
"Quantifying the Risk-Return Tradeoff in Forecasting"
Philippe Goulet Coulombe, arXiv: 2605.09712

Pipeline:
  1. Load config
  2. Download / load FRED-QD panel and SPF data
  3. Build predictor matrix (MARX transformation)
  4. For each target x horizon x evaluation window:
       a. Run expanding-window evaluation for all models
       b. Save loss arrays to results/losses/
  5. Compute risk-adjusted metrics → results/metrics/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from forecast_risk.utils.config import Config, set_seed
from forecast_risk.evaluation.expanding_window import ExpandingWindowEvaluator
from forecast_risk.evaluation.report import RiskAdjustedReport
from forecast_risk.models.linear import ARModel, FAARModel, RidgeForecaster, KernelRidgeForecaster
from forecast_risk.models.tree_models import (
    RandomForestForecaster, LGBForecaster, LGBPlusForecaster, LGBAltForecaster
)
from forecast_risk.models.neural import NeuralNetworkForecaster, HemisphereNeuralNetwork


# ─── Evaluation periods (Section 3) ──────────────────────────────────────────
EVAL_WINDOWS = {
    "pre_covid":  ("2007Q2", "2019Q4"),
    "post_covid": ("2021Q1", "2024Q2"),
}
# 2020 excluded to avoid COVID shock contamination (paper Sec 3)
EXCLUDE_COVID = [("2020Q1", "2020Q4")]

TARGETS = {
    "gdp_growth":       "GDPC1",        # Real GDP (growth rate)
    "cpi_inflation":    "CPIAUCSL",     # CPI inflation
    "unemployment_rate": "UNRATE",      # Unemployment rate
    "housing_starts":   "HOUST",       # Housing starts (log growth)
}


def build_models(cfg: Config) -> dict[str, object]:
    """Instantiate all enabled models from config."""
    enabled = cfg.models.get("enabled", [])
    mc = cfg.models
    models = {}

    if "ar4" in enabled:
        models["ar4"] = ARModel(lags=mc.get("ar4", {}).get("lags", 4))
    if "faar" in enabled:
        m = mc.get("faar", {})
        models["faar"] = FAARModel(n_factors=m.get("n_factors", 4), lags=m.get("lags", 4))
    if "ridge" in enabled:
        m = mc.get("ridge", {})
        models["ridge"] = RidgeForecaster(
            lambda_grid=m.get("lambda_grid"),
            cv_folds=m.get("cv_folds", 5),
        )
    if "kernel_ridge" in enabled:
        m = mc.get("kernel_ridge", {})
        models["kernel_ridge"] = KernelRidgeForecaster(
            kernels=m.get("kernels"),
            sigma_grid=m.get("sigma_grid"),
            lambda_grid=m.get("lambda_grid"),
            cv_folds=m.get("cv_folds", 5),
        )
    if "random_forest" in enabled:
        m = mc.get("random_forest", {})
        models["rf"] = RandomForestForecaster(
            n_estimators=m.get("n_estimators", 500),
            subsample=m.get("subsample", 0.75),
            min_samples_leaf=m.get("min_samples_leaf", 5),
        )
    if "lgb" in enabled:
        m = mc.get("lgb", {})
        models["lgb"] = LGBForecaster(
            num_leaves=m.get("num_leaves", 31),
            learning_rate=m.get("learning_rate", 0.05),
            n_estimators=m.get("n_estimators", 1000),
            early_stopping_rounds=m.get("early_stopping_rounds", 50),
        )
    if "lgb_plus" in enabled:
        m = mc.get("lgb_plus", {})
        models["lgb+"] = LGBPlusForecaster(
            num_leaves=m.get("num_leaves", 31),
            learning_rate=m.get("learning_rate", 0.05),
            n_estimators=m.get("n_estimators", 1000),
        )
    if "lgb_alt" in enabled:
        m = mc.get("lgb_alt", {})
        models["lgba+"] = LGBAltForecaster(
            num_leaves=m.get("num_leaves", 31),
            learning_rate=m.get("learning_rate", 0.05),
            n_estimators=m.get("n_estimators", 1000),
        )
    if "neural_net" in enabled:
        m = mc.get("neural_net", {})
        models["nn"] = NeuralNetworkForecaster(
            hidden_sizes=m.get("hidden_sizes", [400, 400, 400]),
            dropout=m.get("dropout", 0.2),
            learning_rate=m.get("learning_rate", 0.001),
            max_epochs=m.get("max_epochs", 200),
            early_stopping=m.get("early_stopping_patience", 20),
        )
    if "hnn" in enabled:
        m = mc.get("hnn", {})
        models["hnn"] = HemisphereNeuralNetwork(
            hidden_sizes=m.get("hidden_sizes", [400, 400]),
            bootstrap_samples=m.get("bootstrap_samples", 1000),
            learning_rate=m.get("learning_rate", 0.001),
        )

    return models


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FRED-QD panel and SPF data.

    Returns:
        (fred_panel, spf_data) DataFrames.
        fred_panel: [T, N_series] with quarterly PeriodIndex
        spf_data:   SPF median forecasts
    """
    data_cfg = cfg.data
    fred_path = Path(data_cfg.get("output_dir", "data/")) / "FRED-QD.csv"

    if not fred_path.exists():
        print(f"[ERROR] FRED-QD not found at {fred_path}")
        print("  Run: python data/download.py --output-dir data/")
        sys.exit(1)

    print(f"  Loading FRED-QD from {fred_path}...")
    # Row 1: transformation codes; rows 2+: data
    raw = pd.read_csv(fred_path, skiprows=0)

    # Transformation codes in row 0
    tcodes = raw.iloc[0, 1:].astype(float)
    data = raw.iloc[1:].copy()
    data.columns = raw.columns
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.PeriodIndex(data["date"], freq="Q")
    data = data.set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")

    # Apply McCracken-Ng (2020) stationarity transformations
    data = _apply_transformations(data, tcodes)

    # Load SPF (placeholder — returns empty df if not present)
    spf_path = Path(data_cfg.get("spf_file", "data/spf_median.csv"))
    if spf_path.exists() and spf_path.stat().st_size > 100:
        spf = pd.read_csv(spf_path)
    else:
        print("  [WARNING] SPF data not found; SPF will be excluded from evaluation.")
        spf = pd.DataFrame()

    return data, spf


def _apply_transformations(data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """
    Apply McCracken-Ng (2020) transformation codes to achieve stationarity.

    Transformation codes (FRED-QD):
      1: no transformation (level)
      2: first difference
      3: second difference
      4: log
      5: log first difference (growth rate)
      6: log second difference
      7: first difference of percent change
    """
    result = data.copy()
    for col in data.columns:
        if col not in tcodes.index:
            continue
        tc = tcodes[col]
        s = data[col]
        if tc == 1:
            result[col] = s
        elif tc == 2:
            result[col] = s.diff()
        elif tc == 3:
            result[col] = s.diff().diff()
        elif tc == 4:
            result[col] = np.log(s.clip(lower=1e-8))
        elif tc == 5:
            result[col] = np.log(s.clip(lower=1e-8)).diff()
        elif tc == 6:
            result[col] = np.log(s.clip(lower=1e-8)).diff().diff()
        elif tc == 7:
            result[col] = s.pct_change().diff()
    return result


def build_predictors(
    data: pd.DataFrame,
    target_col: str,
    lags: int = 4,
    ma_orders: list = None,
) -> tuple[np.ndarray, np.ndarray, pd.PeriodIndex]:
    """
    Build predictor matrix with MARX transformation.

    Paper: Section 3 — "Predictors drawn from FRED-QD, augmented with 4 lags
    plus moving averages of order 2, 4, and 8 (MARX transformation)."

    Args:
        data:       Full FRED-QD panel [T, N].
        target_col: Column name for the target variable.
        lags:       Number of AR lags (paper: 4).
        ma_orders:  Moving average orders (paper: [2, 4, 8]).

    Returns:
        (X, y, dates) where X: [T, N_features], y: [T], dates: PeriodIndex
    """
    ma_orders = ma_orders or [2, 4, 8]

    # Target
    y_series = data[target_col].copy()

    # Build predictor panel: lags of all series + moving averages
    predictor_parts = []

    # Lagged values (lags 1..lags for all series)
    for lag in range(1, lags + 1):
        lagged = data.shift(lag)
        lagged.columns = [f"{c}_L{lag}" for c in data.columns]
        predictor_parts.append(lagged)

    # Moving averages
    for m in ma_orders:
        ma = data.rolling(m).mean().shift(1)  # shift to avoid lookahead
        ma.columns = [f"{c}_MA{m}" for c in data.columns]
        predictor_parts.append(ma)

    X_df = pd.concat(predictor_parts, axis=1)

    # Align and drop NaN rows
    combined = pd.concat([y_series, X_df], axis=1).dropna()
    dates = combined.index
    y = combined.iloc[:, 0].values
    X = combined.iloc[:, 1:].values

    # Standardize (paper: "standardized to zero mean and unit variance over training sample")
    # Note: standardization is done per training fold in the expanding window
    return X, y, dates


def run_window(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.PeriodIndex,
    horizon: int,
    window_name: str,
    window_start: str,
    window_end: str,
    losses_dir: Path,
    target: str,
    loss_fn: str = "squared_error",
    refit_every: int = 8,
) -> dict[str, np.ndarray]:
    """Run evaluation for one target/horizon/window combination."""
    evaluator = ExpandingWindowEvaluator(
        loss_fn_name=loss_fn,
        refit_every=refit_every,
        verbose=True,
    )
    eval_indices = evaluator.get_eval_indices(
        dates=dates,
        start=window_start,
        end=window_end,
        exclude_ranges=EXCLUDE_COVID if window_name == "post_covid" else None,
    )

    if not eval_indices:
        print(f"  [SKIP] No eval periods for {window_name}")
        return {}

    losses = evaluator.run(
        models=models,
        X=X,
        y=y,
        eval_indices=eval_indices,
        horizon=horizon,
    )

    # Save losses
    for name, arr in losses.items():
        fname = losses_dir / f"{target}_{name}_{window_name}_h{horizon}.npy"
        np.save(fname, arr)

    return losses


def main():
    parser = argparse.ArgumentParser(description="Run forecast risk evaluation")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Override target list")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Override horizon list")
    parser.add_argument("--window", choices=["pre_covid", "post_covid", "both"],
                        default="both")
    args = parser.parse_args()

    # Load config
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.hardware.get("seed", 42),
             deterministic=cfg.hardware.get("deterministic", False))

    # Output dirs
    losses_dir = Path(cfg.output.get("losses_dir", "results/losses/"))
    metrics_dir = Path(cfg.output.get("metrics_dir", "results/metrics/"))
    losses_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Targets and horizons
    targets = args.targets or cfg.data.get("targets", list(TARGETS.keys()))
    horizons = args.horizons or cfg.evaluation.get("horizons", [1, 2, 4])
    benchmark = cfg.evaluation.get("benchmark", "ar4")
    loss_fn = cfg.evaluation.get("loss_function", "squared_error")
    refit_every = cfg.evaluation.get("refit_every", 8)

    print(f"\n{'='*60}")
    print(f"  forecast_risk — Full Evaluation Pipeline")
    print(f"  Targets:  {targets}")
    print(f"  Horizons: {horizons}")
    print(f"  Loss:     {loss_fn}")
    print(f"{'='*60}\n")

    # Load data once
    print("Loading data...")
    fred_panel, spf_data = load_data(cfg)

    # Build models once
    models = build_models(cfg)
    print(f"Models: {list(models.keys())}\n")

    # Determine windows
    windows_to_run = {}
    if args.window in ("pre_covid", "both"):
        windows_to_run["pre_covid"] = EVAL_WINDOWS["pre_covid"]
    if args.window in ("post_covid", "both"):
        windows_to_run["post_covid"] = EVAL_WINDOWS["post_covid"]

    # Main evaluation loop
    all_results = {}

    for target in targets:
        fred_col = TARGETS.get(target)
        if fred_col is None or fred_col not in fred_panel.columns:
            print(f"[SKIP] Target '{target}' (col '{fred_col}') not found in FRED-QD.")
            continue

        print(f"\n{'─'*50}")
        print(f"  Target: {target} ({fred_col})")
        print(f"{'─'*50}")

        X, y, dates = build_predictors(
            data=fred_panel,
            target_col=fred_col,
            lags=cfg.data.get("lags", 4),
            ma_orders=cfg.data.get("ma_orders", [2, 4, 8]),
        )

        for horizon in horizons:
            print(f"\n  Horizon h={horizon}")

            for window_name, (w_start, w_end) in windows_to_run.items():
                print(f"  Window: {window_name} ({w_start}–{w_end})")

                losses = run_window(
                    models=models,
                    X=X, y=y, dates=dates,
                    horizon=horizon,
                    window_name=window_name,
                    window_start=w_start,
                    window_end=w_end,
                    losses_dir=losses_dir,
                    target=target,
                    loss_fn=loss_fn,
                    refit_every=refit_every,
                )

                if not losses or benchmark not in losses:
                    continue

                # Compute risk-adjusted metrics
                reporter = RiskAdjustedReport(horizon=horizon)
                df = reporter.generate(losses, benchmark_key=benchmark)

                # Save
                key = f"{target}_{window_name}_h{horizon}"
                all_results[key] = df
                out_csv = metrics_dir / f"{key}.csv"
                df.to_csv(out_csv)
                print(f"  → Metrics saved: {out_csv}")
                print(df.round(3).to_string())

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"  Losses: {losses_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
