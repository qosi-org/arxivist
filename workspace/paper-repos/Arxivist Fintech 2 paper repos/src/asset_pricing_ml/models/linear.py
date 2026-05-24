"""
models/linear.py — Linear models for Gu, Kelly, Xiu (2020).

Implements four linear approaches from Sections 1.2–1.4:
  - OLS:        Pooled ordinary least squares (Section 1.2)
  - ElasticNet: Penalized regression with L1+L2 penalty (Section 1.3, Eq. 8)
  - PCR:        Principal components regression (Section 1.4)
  - PLS:        Partial least squares via SIMPLS (Section 1.4, Eq. 12)

All models share the same prediction interface:
    fit(Z_train, R_train) + tune(Z_train, R_train, Z_val, R_val) → predict(Z_test)

Paper reference: Sections 1.2, 1.3, 1.4
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler


class OLSModel:
    """Pooled OLS regression. Baseline that fails with 920 predictors due to overfit.

    Paper Section 1.2: "simple linear predictive regression model estimated
    via ordinary least squares."

    L(theta) = (1/NT) * sum_it (r_it+1 - z_it' @ theta)^2

    Paper reference: Section 1.2, Equations (3) and (4)
    """

    def __init__(self, n_predictors: Optional[int] = None):
        """
        Args:
            n_predictors: If set, restrict to first n_predictors features.
                         Set to 3 for OLS-3 (Lewellen 2015 benchmark).
        """
        self.n_predictors = n_predictors
        self.model_ = LinearRegression(fit_intercept=False)

    def fit(self, Z: np.ndarray, R: np.ndarray) -> "OLSModel":
        """
        Args:
            Z: [NT, P] features
            R: [NT]    excess returns
        """
        Z_use = Z[:, :self.n_predictors] if self.n_predictors else Z
        self.model_.fit(Z_use, R)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
            Z: [N, P] features
        Returns:
            R_hat [N] predicted returns
        """
        Z_use = Z[:, :self.n_predictors] if self.n_predictors else Z
        return self.model_.predict(Z_use).astype(np.float32)

    def __repr__(self) -> str:
        suffix = f"-{self.n_predictors}" if self.n_predictors else ""
        return f"OLSModel{suffix}"


class ElasticNetModel:
    """Elastic net penalized regression.

    Paper Section 1.3, Equation (8):
        phi(theta; lambda, rho) = lambda*(1-rho)*sum|theta_j|
                                + (1/2)*lambda*rho*sum(theta_j^2)

    Special cases:
        rho=0 → Lasso (variable selection via sparsity)
        rho=1 → Ridge (shrinkage, no exact zeros)

    Tuning: lambda and rho selected via validation sample.

    Paper reference: Section 1.3, Equations (7) and (8)
    """

    def __init__(self):
        self.model_: Optional[ElasticNet] = None
        self.best_lambda_: Optional[float] = None
        self.best_rho_: Optional[float] = None

    def fit(self, Z: np.ndarray, R: np.ndarray, lambda_: float = 0.01, rho: float = 0.5) -> "ElasticNetModel":
        """Fit with specific hyperparameters.

        sklearn ElasticNet uses l1_ratio = 1 - rho:
            sklearn loss = (1/2n)*||y-Xw||^2 + alpha*l1_ratio*||w||_1 + (alpha/2)*(1-l1_ratio)*||w||^2

        Paper Eq. (8) mapping:
            lambda (paper) → alpha (sklearn)
            rho (paper)    → (1 - l1_ratio) in sklearn
        """
        # Paper Eq. (8): phi = lambda*(1-rho)*L1 + (1/2)*lambda*rho*L2
        # sklearn: alpha*(l1_ratio)*L1 + alpha*(1-l1_ratio)/2*L2
        # → alpha=lambda, l1_ratio=(1-rho)
        self.model_ = ElasticNet(
            alpha=lambda_,
            l1_ratio=max(1e-6, 1.0 - rho),  # avoid pure ridge for numerical stability
            fit_intercept=False,
            max_iter=10000,
        )
        self.model_.fit(Z, R)
        self.best_lambda_ = lambda_
        self.best_rho_ = rho
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        lambda_grid: List[float],
        rho_grid: List[float],
    ) -> Tuple[float, float]:
        """Grid search over validation sample to find best (lambda, rho).

        Paper: "We adaptively optimize the tuning parameters lambda and rho
        using the validation sample." (Section 1.3)

        Returns:
            Tuple of (best_lambda, best_rho)
        """
        best_r2 = -np.inf
        best_lambda, best_rho = lambda_grid[0], rho_grid[0]

        for lam in lambda_grid:
            for rho in rho_grid:
                self.fit(Z_train, R_train, lam, rho)
                pred = self.predict(Z_val)
                r2 = _oos_r2(R_val, pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_lambda, best_rho = lam, rho

        # Refit on training with best params
        self.fit(Z_train, R_train, best_lambda, best_rho)
        return best_lambda, best_rho

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() or tune() first.")
        return self.model_.predict(Z).astype(np.float32)

    def __repr__(self) -> str:
        return f"ElasticNetModel(lambda={self.best_lambda_}, rho={self.best_rho_})"


class PCRModel:
    """Principal Components Regression.

    Paper Section 1.4: Two-step procedure:
      1. Reduce Z [NT, P] → Z_reduced [NT, K] via SVD (Eq. 11)
      2. OLS regression of R on Z_reduced

    Omega_K is the P×K matrix of principal component loadings.

    PCR does NOT use the return target in its dimension reduction step
    (contrast with PLS). It maximizes variance among predictors.

    Paper reference: Section 1.4, Equation (11)
    """

    def __init__(self):
        self.pca_: Optional[PCA] = None
        self.ols_: Optional[LinearRegression] = None
        self.best_K_: Optional[int] = None

    def fit(self, Z: np.ndarray, R: np.ndarray, K: int = 20) -> "PCRModel":
        """
        Args:
            Z: [NT, P] feature matrix
            R: [NT]    excess returns
            K: Number of principal components
        """
        # Step 1: PCA dimension reduction  (Eq. 11)
        self.pca_ = PCA(n_components=K)
        Z_reduced = self.pca_.fit_transform(Z)  # [NT, K]

        # Step 2: OLS on reduced features
        self.ols_ = LinearRegression(fit_intercept=False)
        self.ols_.fit(Z_reduced, R)
        self.best_K_ = K
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        K_grid: List[int],
    ) -> int:
        """Select K via validation R²."""
        best_r2, best_K = -np.inf, K_grid[0]
        for K in K_grid:
            self.fit(Z_train, R_train, K)
            r2 = _oos_r2(R_val, self.predict(Z_val))
            if r2 > best_r2:
                best_r2, best_K = r2, K
        self.fit(Z_train, R_train, best_K)
        return best_K

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("Call fit() first.")
        Z_reduced = self.pca_.transform(Z)
        return self.ols_.predict(Z_reduced).astype(np.float32)

    def __repr__(self) -> str:
        return f"PCRModel(K={self.best_K_})"


class PLSModel:
    """Partial Least Squares via SIMPLS algorithm.

    Paper Section 1.4, Equation (12):
        w_j = argmax_w Cov^2(R, Z@w)
              s.t. w'w=1, Cov(Z@w, Z@w_l)=0 for l<j

    PLS maximizes predictive covariance with returns (unlike PCR which
    maximizes variance among predictors alone). Implemented via sklearn's
    PLSRegression which uses NIPALS/SIMPLS.

    Paper reference: Section 1.4, Equations (10)–(12), citing de Jong (1993)
    """

    def __init__(self):
        self.model_: Optional[PLSRegression] = None
        self.best_K_: Optional[int] = None

    def fit(self, Z: np.ndarray, R: np.ndarray, K: int = 3) -> "PLSModel":
        """
        Args:
            Z: [NT, P] features
            R: [NT]    returns (paper: 3-6 components selected)
            K: Number of PLS components
        """
        K = min(K, Z.shape[1], Z.shape[0] - 1)
        self.model_ = PLSRegression(n_components=K, scale=False)
        self.model_.fit(Z, R.reshape(-1, 1))
        self.best_K_ = K
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        K_grid: List[int],
    ) -> int:
        """Select K via validation R²."""
        best_r2, best_K = -np.inf, K_grid[0]
        for K in K_grid:
            self.fit(Z_train, R_train, K)
            r2 = _oos_r2(R_val, self.predict(Z_val))
            if r2 > best_r2:
                best_r2, best_K = r2, K
        self.fit(Z_train, R_train, best_K)
        return best_K

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(Z).squeeze().astype(np.float32)

    def __repr__(self) -> str:
        return f"PLSModel(K={self.best_K_})"


def _oos_r2(R_actual: np.ndarray, R_pred: np.ndarray) -> float:
    """Out-of-sample R² benchmarked against zero forecast (Paper Eq. 22).

    Paper Section 1.8: "A subtle but important aspect of our R² metric is
    that the denominator is the sum of squared excess returns without demeaning."

    R²_oos = 1 - sum(r - r_hat)^2 / sum(r^2)
    """
    ss_res = np.sum((R_actual - R_pred) ** 2)
    ss_tot = np.sum(R_actual ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
