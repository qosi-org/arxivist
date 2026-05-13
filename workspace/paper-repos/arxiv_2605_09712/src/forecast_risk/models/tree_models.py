"""
forecast_risk.models.tree_models
===================================
Tree-based forecasting models: Random Forest, LightGBM, LGB+, LGBA+.

Paper: Section 3 — Models
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

Model descriptions (from paper):
  RF:    Random Forest. 500 trees, 75% subsampling, min node size 5.
  LGB:   LightGBM. Histogram-based gradient boosting with cross-validated params.
  LGB+:  Hybrid boosting (Goulet Coulombe 2026) — STUB: see note below.
  LGBA+: Alternating variant of LGB+ — STUB: see note below.

STUB NOTE for LGB+ and LGBA+:
  These are non-standard variants described in Goulet Coulombe (2026), a working
  paper not fully available. We approximate them using LightGBM with linear_tree=True.
  The actual LGB+ algorithm alternates tree-based and linear updates at each
  boosting step, selecting the winner via out-of-bag validation. Replace this
  approximation with the actual implementation when available.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from .base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest with parameters explicitly stated in the paper.

    Paper: Section 3 — "Aggregates 500 trees grown on bootstrap samples with
    random feature subsets; 75% subsampling, minimum node size 5."

    Args:
        n_estimators:    Number of trees (paper: 500).
        subsample:       Row subsampling fraction (paper: 0.75).
        min_samples_leaf: Min samples per leaf (paper: 5).
        max_features:    Feature subsampling (ASSUMED: 'sqrt').
        random_state:    Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        subsample: float = 0.75,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",  # ASSUMED
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model = None

    @property
    def name(self) -> str:
        return "RF"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_samples=self.subsample,          # 75% subsampling (paper-specified)
            min_samples_leaf=self.min_samples_leaf,  # min node size 5 (paper-specified)
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LGBForecaster(BaseForecaster):
    """
    LightGBM gradient boosting forecaster.

    Paper: Section 3 — "Sequential gradient boosting with histogram-based splits;
    learning rate, depth, and sampling fractions cross-validated with early stopping."

    Args:
        num_leaves:            LGB complexity control.
        learning_rate:         Step size.
        n_estimators:          Max boosting rounds.
        early_stopping_rounds: Early stopping patience (ASSUMED: 50).
        random_state:          Random seed.
    """

    def __init__(
        self,
        num_leaves: int = 31,           # ASSUMED
        learning_rate: float = 0.05,    # ASSUMED
        n_estimators: int = 1000,       # ASSUMED
        early_stopping_rounds: int = 50,  # ASSUMED: not specified in paper
        random_state: int = 42,
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model = None

    @property
    def name(self) -> str:
        return "LGB"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Use last 20% of training data as validation for early stopping
        n_val = max(int(0.2 * len(y)), 1)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self._model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LGBPlusForecaster(BaseForecaster):
    """
    LGB+: Hybrid LightGBM (Goulet Coulombe, 2026).

    STUB: The actual LGB+ algorithm alternates tree-based and linear updates
    at each boosting step, selecting the winner via out-of-bag validation.
    This implementation uses LightGBM with linear_tree=True as an approximation.

    Reference: Goulet Coulombe (2026) — "LGB+: A macroeconomic forecasting road test"
    (working paper — full implementation not available from this paper alone).

    Paper: Section 3 — "At each boosting step, a tree-based and a linear update
    compete; the winner is selected via out-of-bag validation."
    """

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,  # ASSUMED
        random_state: int = 42,
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model = None

    @property
    def name(self) -> str:
        return "LGB+"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # STUB: approximation using linear_tree
        # TODO: Replace with actual LGB+ from Goulet Coulombe (2026)
        n_val = max(int(0.2 * len(y)), 1)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self._model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            linear_tree=True,               # Approximation of hybrid tree+linear
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LGBAltForecaster(BaseForecaster):
    """
    LGBA+: Alternating variant of LGB+ (Goulet Coulombe, 2026).

    STUB: "A more computationally economical version that alternates tree ensembles
    with linear corrections in a fixed pattern each boosting cycle."

    Same approximation strategy as LGB+ above.
    """

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,  # ASSUMED
        random_state: int = 42,
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model = None

    @property
    def name(self) -> str:
        return "LGBA+"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # STUB: approximation — TODO: Replace with actual LGBA+ from Goulet Coulombe (2026)
        n_val = max(int(0.2 * len(y)), 1)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self._model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            linear_tree=True,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
