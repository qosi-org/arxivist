"""
forecast_risk.evaluation.expanding_window
==========================================
Expanding-window out-of-sample forecast evaluation engine.

Implements the recursive out-of-sample design described in Section 3 of:
  "Quantifying the Risk-Return Tradeoff in Forecasting"
  Philippe Goulet Coulombe, arXiv: 2605.09712

Design (from paper):
  - Expanding window from train_start to current evaluation period
  - Models re-estimated every `refit_every` quarters (default: 8)
  - Direct forecasting: separate model per horizon h
  - Pre-COVID: 2007Q2–2019Q4; Post-COVID: 2021Q1–2024Q2
  - 2020 excluded to avoid COVID contamination
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional
from tqdm import tqdm

from ..models.base import BaseForecaster


def squared_error(y_true: float, y_pred: float) -> float:
    """Squared error loss. Sec 2.1: L(y, y_hat) can be any loss function."""
    return (y_true - y_pred) ** 2


def absolute_error(y_true: float, y_pred: float) -> float:
    """Absolute error loss."""
    return abs(y_true - y_pred)


LOSS_FUNCTIONS: dict[str, Callable] = {
    "squared_error": squared_error,
    "absolute_error": absolute_error,
}


class ExpandingWindowEvaluator:
    """
    Expanding-window out-of-sample evaluator.

    Paper: Section 3 — "The out-of-sample evaluation proceeds on an expanding
    window ... with models re-estimated every 8 quarters (2 years)."

    Args:
        loss_fn_name:  Name of loss function ('squared_error', 'absolute_error').
        refit_every:   Re-estimate models every N periods (paper: 8 quarters).
        verbose:       Show progress bar.
    """

    def __init__(
        self,
        loss_fn_name: str = "squared_error",
        refit_every: int = 8,
        verbose: bool = True,
    ):
        if loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(
                f"loss_fn_name must be one of {list(LOSS_FUNCTIONS)}, got '{loss_fn_name}'"
            )
        self.loss_fn = LOSS_FUNCTIONS[loss_fn_name]
        self.loss_fn_name = loss_fn_name
        self.refit_every = refit_every
        self.verbose = verbose

    def run(
        self,
        models: dict[str, BaseForecaster],
        X: np.ndarray,
        y: np.ndarray,
        eval_indices: list[int],
        horizon: int = 1,
        min_train_size: int = 40,
    ) -> dict[str, np.ndarray]:
        """
        Run expanding-window evaluation for all models.

        Args:
            models:        {name: model} dict.
            X:             Full predictor matrix [T_full, N_features].
            y:             Full target series [T_full].
            eval_indices:  List of time indices to evaluate at (e.g. [T_start, ..., T_end]).
            horizon:       Forecast horizon h (separate model per horizon in paper).
            min_train_size: Minimum training observations before first evaluation.

        Returns:
            {model_name: loss_series [T_eval]} dict.
        """
        n_eval = len(eval_indices)
        losses = {name: np.full(n_eval, np.nan) for name in models}
        fitted = {name: False for name in models}

        iterator = tqdm(enumerate(eval_indices), total=n_eval, desc=f"h={horizon}") \
            if self.verbose else enumerate(eval_indices)

        for step_i, t in iterator:
            # Training data: everything up to t (not including t+h which is target)
            train_end = t - horizon  # direct forecasting: target is y[t]
            if train_end < min_train_size:
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[t:t + 1]
            y_true = y[t]

            for name, model in models.items():
                # Re-fit on schedule or first time
                should_refit = (
                    not fitted[name]
                    or (step_i % self.refit_every == 0)
                )
                if should_refit:
                    try:
                        model.fit(X_train, y_train)
                        fitted[name] = True
                    except Exception as e:
                        print(f"[WARNING] {name} failed to fit at t={t}: {e}")
                        continue

                try:
                    y_pred = float(model.predict(X_test)[0])
                    losses[name][step_i] = self.loss_fn(float(y_true), y_pred)
                except Exception as e:
                    print(f"[WARNING] {name} failed to predict at t={t}: {e}")

        return losses

    def get_eval_indices(
        self,
        dates: pd.PeriodIndex,
        start: str,
        end: str,
        exclude_ranges: Optional[list[tuple[str, str]]] = None,
    ) -> list[int]:
        """
        Get integer indices for the evaluation window, with optional exclusions.

        Paper: 2020 excluded to avoid COVID contamination.

        Args:
            dates:          Full date index of the data.
            start:          First evaluation period (e.g. '2007Q2').
            end:            Last evaluation period (e.g. '2019Q4').
            exclude_ranges: List of (start, end) ranges to exclude (e.g. [('2020Q1','2020Q4')]).

        Returns:
            List of integer indices.
        """
        indices = []
        for i, d in enumerate(dates):
            d_str = str(d)
            if d_str < start or d_str > end:
                continue
            if exclude_ranges:
                skip = False
                for ex_start, ex_end in exclude_ranges:
                    if ex_start <= d_str <= ex_end:
                        skip = True
                        break
                if skip:
                    continue
            indices.append(i)
        return indices
