"""
models/glm.py — Generalized linear model with spline basis expansion.

Implements the GLM from Section 1.5 of Gu, Kelly, Xiu (2020):
  g(z; theta, p(·)) = sum_j p(z_j)' @ theta_j

where p(z) = (1, z, (z-c1)^2, ..., (z-c_{K-2})^2) is a spline basis
and the group lasso penalty selects characteristics as groups:
  phi(theta; lambda, K) = lambda * sum_j ||theta_j||_2

Key property: introduces nonlinear univariate transformations but
NO interactions between predictors. This explains why GLM does NOT
outperform linear methods significantly — interactions are missing.

ASSUMED: Spline knot locations not specified. Using quantile-based knots.
         Confidence: 0.62. TODO: verify from paper.

Paper reference: Section 1.5, Equations (13) and (14)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet

from asset_pricing_ml.models.linear import _oos_r2


class GeneralizedLinearModel:
    """Generalized linear model: spline basis + group lasso.

    For each predictor j, expand z_j with K spline basis functions:
        p(z) = (1, z, (z-c1)^2, ..., (z-c_{K-2})^2)

    This multiplies parameters by K but still includes no cross-predictor
    interactions — explaining why it does NOT outperform linear approaches.

    Group lasso selects either all K spline terms for a characteristic or none,
    implementing 'variable selection at the characteristic level'.

    ASSUMED: Knot placement via equally spaced quantiles of training data.
             Confidence: 0.62. TODO: verify from Internet Appendix B.1.

    Paper reference: Section 1.5, Equations (13) and (14)

    Args:
        K: Spline order (number of basis functions per predictor). Default 3.
        lambda_: Group lasso penalty strength.
        knot_strategy: 'quantiles' (assumed) or 'uniform'.
    """

    def __init__(
        self,
        K: int = 3,
        lambda_: float = 0.01,
        # ASSUMED: knot strategy not specified in paper (confidence 0.62)
        # TODO: verify from Internet Appendix B.1
        knot_strategy: str = "quantiles",
    ):
        self.K = K
        self.lambda_ = lambda_
        self.knot_strategy = knot_strategy
        self.knots_: Optional[np.ndarray] = None  # [P, K-2] knot locations per predictor
        self.coef_: Optional[np.ndarray] = None   # [P*K] coefficients
        self.best_K_: int = K
        self.best_lambda_: float = lambda_

    def _compute_knots(self, Z: np.ndarray) -> np.ndarray:
        """Compute spline knot locations from training data.

        ASSUMED: equally spaced quantiles. Confidence 0.62.

        Args:
            Z: [NT, P] training feature matrix
        Returns:
            knots: [P, K-2] knot locations per predictor
        """
        P = Z.shape[1]
        n_knots = max(0, self.K - 2)
        if n_knots == 0:
            return np.zeros((P, 0))

        # K-2 interior knots at equally spaced quantile points
        quantile_pts = np.linspace(0, 100, n_knots + 2)[1:-1]
        knots = np.zeros((P, n_knots))
        for j in range(P):
            knots[j] = np.percentile(Z[:, j], quantile_pts)
        return knots

    def _build_spline_basis(self, Z: np.ndarray) -> np.ndarray:
        """Expand each predictor with spline basis functions.

        Paper Equation (13):
            p(z) = (1, z, (z-c1)^2, ..., (z-c_{K-2})^2)

        Args:
            Z: [N, P] feature matrix (original predictors)
        Returns:
            Z_expanded: [N, P*K] spline-expanded feature matrix
        """
        assert self.knots_ is not None, "Must call _compute_knots first"
        N, P = Z.shape
        n_knots = self.knots_.shape[1]
        K = self.K

        expanded_cols = []
        for j in range(P):
            z_j = Z[:, j]                    # [N]
            basis = [np.ones(N), z_j]         # constant and linear terms
            for k in range(n_knots):
                c_k = self.knots_[j, k]
                # (z - c_k)^2 truncated spline term
                basis.append(np.maximum(0, z_j - c_k) ** 2)
            expanded_cols.append(np.stack(basis[:K], axis=1))  # [N, K]

        return np.concatenate(expanded_cols, axis=1).astype(np.float32)  # [N, P*K]

    def _group_lasso_fit(self, Z_expanded: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Fit with approximate group lasso via elastic net with L1 on group norms.

        Paper Equation (14):
            phi(theta; lambda, K) = lambda * sum_j ( sum_k theta_{j,k}^2 )^(1/2)

        NOTE: True group lasso requires specialized solvers. We approximate with
        scikit-learn ElasticNet applied to spline-expanded features as a practical
        surrogate. For a faithful group lasso, replace with a dedicated implementation.

        TODO: Replace with true group lasso solver for exact paper reproduction.
        """
        # Approximate group lasso with standard L1+L2
        model = ElasticNet(
            alpha=self.lambda_,
            l1_ratio=0.9,  # predominantly L1 to approximate group selection
            fit_intercept=False,
            max_iter=10000,
        )
        model.fit(Z_expanded, R)
        return model.coef_.astype(np.float32)

    def fit(
        self,
        Z: np.ndarray,
        R: np.ndarray,
        K: Optional[int] = None,
        lambda_: Optional[float] = None,
    ) -> "GeneralizedLinearModel":
        """Fit GLM with spline expansion and group lasso.

        Args:
            Z: [NT, P] feature matrix
            R: [NT] excess returns
            K: Spline order (overrides __init__ value if set)
            lambda_: Group lasso strength
        """
        if K is not None:
            self.K = K
        if lambda_ is not None:
            self.lambda_ = lambda_

        # Compute knot locations from training data
        self.knots_ = self._compute_knots(Z)

        # Expand features with spline basis
        Z_expanded = self._build_spline_basis(Z)

        # Fit with (approximate) group lasso
        self.coef_ = self._group_lasso_fit(Z_expanded, R)
        return self

    def tune(
        self,
        Z_train: np.ndarray, R_train: np.ndarray,
        Z_val: np.ndarray,   R_val: np.ndarray,
        K_grid: List[int],
        lambda_grid: List[float],
    ) -> Tuple[int, float]:
        """Grid search over validation sample."""
        best_r2, best_K, best_lam = -np.inf, self.K, self.lambda_
        for K in K_grid:
            for lam in lambda_grid:
                self.fit(Z_train, R_train, K=K, lambda_=lam)
                r2 = _oos_r2(R_val, self.predict(Z_val))
                if r2 > best_r2:
                    best_r2, best_K, best_lam = r2, K, lam
        self.fit(Z_train, R_train, K=best_K, lambda_=best_lam)
        self.best_K_ = best_K
        self.best_lambda_ = best_lam
        return best_K, best_lam

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        Z_expanded = self._build_spline_basis(Z)
        return (Z_expanded @ self.coef_).astype(np.float32)

    def __repr__(self) -> str:
        return f"GeneralizedLinearModel(K={self.best_K_}, lambda={self.best_lambda_:.4f})"
