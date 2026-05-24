"""
models/trees.py — Tree ensemble models for Gu, Kelly, Xiu (2020).

Implements two tree ensemble methods from Section 1.6:
  - GBRT: Gradient Boosted Regression Trees (boosting + shrinkage)
  - RF:   Random Forest (bootstrap aggregation + random feature subsets)

Key findings from paper:
  - Trees and neural networks outperform all linear methods
  - Predictive gains trace to nonlinear predictor interactions
  - RF typically uses 1-5 layers on average (shallow trees)
  - GBRT uses ~50-100 characteristics in its ensemble

Paper reference: Section 1.6, Equations (15) and (16)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from asset_pricing_ml.models.linear import _oos_r2


class GBRTModel:
    """Gradient Boosted Regression Trees.

    Paper Section 1.6: "boosting recursively combines forecasts from many
    oversimplified trees." Each shallow tree is fitted to residuals from
    previous ensemble, shrunk by factor nu in (0,1).

    Final prediction = sum of B shallow trees, each shrunk by nu.

    Tuning parameters (Section 1.6): L (depth), nu (shrinkage), B (n_trees)
    All selected via validation sample.

    Paper finding: "trees with fewer than six leaves on average" →
    depth L=1 or L=2 (binary trees with 2-4 leaves).

    Paper reference: Section 1.6, Algorithm 4 in Internet Appendix B.2.
    """

    def __init__(
        self,
        L: int = 2,          # Tree depth
        nu: float = 0.01,    # Shrinkage factor (learning rate)
        B: int = 1000,       # Number of boosting iterations
    ):
        self.L = L
        self.nu = nu
        self.B = B
        self.model_: Optional[GradientBoostingRegressor] = None
        self.best_params_: Optional[dict] = None

    def fit(
        self,
        Z: np.ndarray,
        R: np.ndarray,
        L: Optional[int] = None,
        nu: Optional[float] = None,
        B: Optional[int] = None,
    ) -> "GBRTModel":
        """
        Args:
            Z: [NT, P] feature matrix
            R: [NT] excess returns
        """
        L = L or self.L
        nu = nu or self.nu
        B = B or self.B

        # L2 impurity per paper Eq. (16): H(theta, C) = (1/|C|) * sum (r-theta)^2
        self.model_ = GradientBoostingRegressor(
            max_depth=L,
            learning_rate=nu,
            n_estimators=B,
            loss="squared_error",
            subsample=0.5,         # stochastic gradient boosting for regularization
            random_state=42,
        )
        self.model_.fit(Z, R)
        self.L, self.nu, self.B = L, nu, B
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        L_grid: List[int],
        nu_grid: List[float],
        B_grid: List[int],
    ) -> dict:
        """Grid search for (L, nu, B) via validation R²."""
        best_r2 = -np.inf
        best_params = {"L": self.L, "nu": self.nu, "B": self.B}
        for L in L_grid:
            for nu in nu_grid:
                for B in B_grid:
                    self.fit(Z_train, R_train, L=L, nu=nu, B=B)
                    r2 = _oos_r2(R_val, self.predict(Z_val))
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {"L": L, "nu": nu, "B": B}
        self.fit(Z_train, R_train, **best_params)
        self.best_params_ = best_params
        return best_params

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(Z).astype(np.float32)

    def feature_importance(self) -> np.ndarray:
        """Mean decrease in impurity across all trees.

        Paper Section 1.9: For tree models, variable importance is measured
        by 'mean decrease in impurity' (since SSD is not differentiable for trees).
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.feature_importances_

    def __repr__(self) -> str:
        return f"GBRTModel(L={self.L}, nu={self.nu}, B={self.B})"


class RandomForestModel:
    """Random Forest — bootstrap aggregation with random feature subsets.

    Paper Section 1.6: "Random forests use a variation on bagging designed
    to reduce the correlation among trees in different bootstrap samples."

    At each split, only a random subset of m predictors is considered —
    this decorrelates trees and prevents one dominant feature from
    appearing at the root of every tree.

    Paper finding: RF trees tend to be "deep" (1-5 layers average).
    Depth, m (features per split), and B (bootstrap samples) tuned via validation.

    Paper reference: Section 1.6, Algorithm 3 in Internet Appendix B.2.
    """

    def __init__(
        self,
        L: int = 5,           # Max tree depth (-1 = unlimited)
        m: str = "sqrt",      # Features per split ("sqrt" or "third")
        B: int = 300,         # Number of bootstrap trees (ASSUMED: Breiman 2001 default)
    ):
        self.L = L
        self.m = m
        self.B = B
        self.model_: Optional[RandomForestRegressor] = None
        self.best_params_: Optional[dict] = None

    def fit(
        self,
        Z: np.ndarray,
        R: np.ndarray,
        L: Optional[int] = None,
        m: Optional[str] = None,
        B: Optional[int] = None,
    ) -> "RandomForestModel":
        """
        Args:
            Z: [NT, P] feature matrix
            R: [NT] excess returns
        """
        L = L if L is not None else self.L
        m = m or self.m
        B = B or self.B

        max_depth = None if L == -1 else L
        max_features = {"sqrt": "sqrt", "third": max(1, Z.shape[1] // 3)}.get(m, m)

        self.model_ = RandomForestRegressor(
            n_estimators=B,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        self.model_.fit(Z, R)
        self.L, self.m, self.B = L, m, B
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        L_grid: List[int],
        m_grid: List[str],
    ) -> dict:
        """Grid search for (L, m) via validation R²."""
        best_r2 = -np.inf
        best_params = {"L": self.L, "m": self.m}
        for L in L_grid:
            for m in m_grid:
                self.fit(Z_train, R_train, L=L, m=m)
                r2 = _oos_r2(R_val, self.predict(Z_val))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {"L": L, "m": m}
        self.fit(Z_train, R_train, **best_params)
        self.best_params_ = best_params
        return best_params

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(Z).astype(np.float32)

    def feature_importance(self) -> np.ndarray:
        """Mean decrease in impurity for variable importance (Section 1.9)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.feature_importances_

    def __repr__(self) -> str:
        return f"RandomForestModel(L={self.L}, m={self.m}, B={self.B})"
