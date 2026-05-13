"""
forecast_risk.models.linear
=============================
Linear forecasting models: AR(4), FAAR, Ridge Regression, Kernel Ridge Regression.

Paper: Section 3 — Models
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

Model descriptions:
  AR(4): Fourth-order autoregression on the target variable.
  FAAR:  Factor-Augmented AR — augments AR(4) with 4 principal components (Stock & Watson 2002).
  RR:    Ridge Regression with L2 penalty; lambda selected by cross-validation.
  KRR:   Kernel Ridge Regression with Gaussian/Laplacian kernels; cross-validated.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class ARModel(BaseForecaster):
    """
    AR(4): Fourth-order autoregression on target only.

    Paper: Section 3 — "The benchmark that any serious forecaster must beat."
    Uses only 4 lags of the target variable as predictors.

    Args:
        lags: Number of autoregressive lags (default 4, explicitly stated in paper).
    """

    def __init__(self, lags: int = 4):
        self.lags = lags
        self._coef = None
        self._intercept = None

    @property
    def name(self) -> str:
        return f"AR({self.lags})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit OLS autoregression using first `lags` columns of X as lagged targets.
        Assumes X[:, :lags] contains lags of y.
        """
        X_ar = X[:, :self.lags]
        X_aug = np.column_stack([np.ones(len(X_ar)), X_ar])
        # OLS: beta = (X'X)^{-1} X'y
        self._coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ar = X[:, :self.lags]
        X_aug = np.column_stack([np.ones(len(X_ar)), X_ar])
        return X_aug @ self._coef


class FAARModel(BaseForecaster):
    """
    FAAR: Factor-Augmented AR(4).

    Paper: Section 3 — "Augments the AR(4) with four principal components
    extracted from the predictor panel, exploiting cross-variable information
    while avoiding the curse of dimensionality." (Stock & Watson, 2002)

    Args:
        n_factors: Number of principal components to extract (default 4, paper-specified).
        lags:      AR lags for target variable (default 4).
    """

    def __init__(self, n_factors: int = 4, lags: int = 4):
        self.n_factors = n_factors
        self.lags = lags
        self._pca = PCA(n_components=n_factors)
        self._ridge = Ridge(alpha=0.0, fit_intercept=True)  # OLS via Ridge with alpha=0

    @property
    def name(self) -> str:
        return "FAAR"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_ar = X[:, :self.lags]
        X_panel = X[:, self.lags:]      # Remaining columns: full predictor panel
        factors = self._pca.fit_transform(X_panel)  # [T, n_factors]
        X_faar = np.column_stack([X_ar, factors])
        self._ridge.fit(X_faar, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ar = X[:, :self.lags]
        X_panel = X[:, self.lags:]
        factors = self._pca.transform(X_panel)
        X_faar = np.column_stack([X_ar, factors])
        return self._ridge.predict(X_faar)


class RidgeForecaster(BaseForecaster):
    """
    Ridge Regression with L2 penalty.

    Paper: Section 3 — "High-dimensional linear prediction with L2 penalty;
    regularization parameter λ selected by cross-validation."

    Args:
        lambda_grid: Grid of alpha (lambda) values to search.
        cv_folds:    Number of time-series CV folds.
    """

    def __init__(
        self,
        lambda_grid: list[float] | None = None,
        cv_folds: int = 5,
    ):
        # ASSUMED: lambda grid not specified in paper
        self.lambda_grid = lambda_grid or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.cv_folds = cv_folds
        self._model = None

    @property
    def name(self) -> str:
        return "Ridge"

    def _cv_lambda(self, X: np.ndarray, y: np.ndarray) -> float:
        """Select best lambda by expanding-window time-series CV."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        best_mse = np.inf
        best_lambda = self.lambda_grid[0]
        for lam in self.lambda_grid:
            mses = []
            for train_idx, val_idx in tscv.split(X):
                m = Ridge(alpha=lam, fit_intercept=True)
                m.fit(X[train_idx], y[train_idx])
                preds = m.predict(X[val_idx])
                mses.append(np.mean((y[val_idx] - preds) ** 2))
            avg_mse = np.mean(mses)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lambda = lam
        return best_lambda

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        best_lam = self._cv_lambda(X, y)
        self._model = Ridge(alpha=best_lam, fit_intercept=True)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class KernelRidgeForecaster(BaseForecaster):
    """
    Kernel Ridge Regression with Gaussian and Laplacian kernels.

    Paper: Section 3 — "Extends ridge to nonlinear settings via Gaussian and
    Laplacian kernels; bandwidth σ and λ cross-validated."

    SIR Ambiguity: Grid search range not specified (confidence 0.55).
    Using log-spaced defaults — adjust via config if needed.

    Args:
        kernels:      List of kernel types to search.
        sigma_grid:   Bandwidth grid for RBF/Laplacian kernels.
        lambda_grid:  Regularization lambda grid.
        cv_folds:     Time-series CV folds.
    """

    def __init__(
        self,
        kernels: list[str] | None = None,
        sigma_grid: list[float] | None = None,
        lambda_grid: list[float] | None = None,
        cv_folds: int = 5,
    ):
        # ASSUMED: grids not specified in paper (SIR confidence 0.55)
        self.kernels = kernels or ["rbf", "laplacian"]
        self.sigma_grid = sigma_grid or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.lambda_grid = lambda_grid or [0.001, 0.01, 0.1, 1.0, 10.0]
        self.cv_folds = cv_folds
        self._model = None
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "KRR"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: verify CV scheme from paper
        X_scaled = self._scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        best_mse = np.inf
        best_params = {"kernel": "rbf", "gamma": 1.0, "alpha": 1.0}

        for kernel in self.kernels:
            for sigma in self.sigma_grid:
                gamma = 1.0 / (2 * sigma ** 2)
                for lam in self.lambda_grid:
                    mses = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        m = KernelRidge(kernel=kernel, gamma=gamma, alpha=lam)
                        m.fit(X_scaled[train_idx], y[train_idx])
                        preds = m.predict(X_scaled[val_idx])
                        mses.append(np.mean((y[val_idx] - preds) ** 2))
                    avg_mse = np.mean(mses)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_params = {"kernel": kernel, "gamma": gamma, "alpha": lam}

        self._model = KernelRidge(**best_params)
        self._model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)
