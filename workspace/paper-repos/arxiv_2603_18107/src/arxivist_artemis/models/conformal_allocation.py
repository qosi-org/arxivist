"""
models/conformal_allocation.py
=================================
Module 4: adaptive conformal prediction (Section 4.4, Appendix A.7) and
optional differentiable Kelly-criterion portfolio QP.

NOTE (architecture_plan.json risk assessment, Low severity): the paper never
confirms whether the Kelly-portfolio layer was actually exercised to produce
any reported Table 2/3 metric (those are all point-prediction metrics: RMSE,
RankIC, DirAcc, Weighted R2). KellyPortfolioLayer is implemented as an
explicitly optional, opt-in module and its output is never fed into any of
the reproduced benchmark metrics.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn


class AdaptiveConformalPredictor:
    """
    Distribution-free prediction intervals via (adaptive) conformal
    prediction (Section 4.4, Appendix A.7.1-A.7.2, Eqs. 15-17).

    Standard (non-adaptive) conformal prediction over a fixed calibration
    set gives the marginal coverage guarantee P(y_test in C(X_test)) >= 1-alpha
    under exchangeability (Eq. 16). Financial data violates exchangeability
    (non-stationarity), so the adaptive rolling-window variant is used
    instead, at the cost of losing the strict finite-sample guarantee (the
    paper explicitly acknowledges this trade-off, citing Gibbs & Candes 2021).

    Args:
        window: rolling window size W. ASSUMED, not specified in paper.
        alpha: target miscoverage rate. ASSUMED=0.1 (90% coverage), not
            specified in paper.
    """

    def __init__(self, window: int = 500, alpha: float = 0.1) -> None:
        self.window = window
        self.alpha = alpha
        self._residuals: Deque[float] = deque(maxlen=window)

    def update(self, residual: float) -> None:
        """Add a new absolute residual |y - y_hat| to the rolling window."""
        self._residuals.append(abs(residual))

    def quantile(self) -> float:
        """Current (1-alpha)-quantile of the rolling residual window (Eq. 17)."""
        if len(self._residuals) == 0:
            return 0.0
        return float(np.quantile(np.array(self._residuals), 1 - self.alpha))

    def predict_interval(self, y_hat: float) -> Tuple[float, float]:
        """[y_hat - q, y_hat + q] (Eq. 17)."""
        q = self.quantile()
        return (y_hat - q, y_hat + q)

    def calibrate(self, residuals: np.ndarray) -> None:
        """Bulk-initialize the rolling window from an array of residuals."""
        for r in residuals[-self.window :]:
            self._residuals.append(abs(float(r)))


class KellyPortfolioLayer(nn.Module):
    """
    Differentiable continuous Kelly-criterion portfolio optimization
    (Appendix A.7.3, Eqs. 18-20):

        max_w  w^T y_hat - (gamma/2) w^T Sigma_hat w
        s.t.   1^T w = 1, w >= 0

    with Sigma_hat approximated as diag(q_p^2) from conformal intervals.
    Solved via a differentiable convex-optimization layer (cvxpylayers,
    citing Agrawal et al. 2019). OPTIONAL -- disabled by default via
    config.model.artemis.use_conformal_allocation controlling interval
    prediction only; this Kelly layer is a separate opt-in (see README).
    """

    def __init__(self, n_assets: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.gamma = gamma
        self._layer = None  # lazily built; requires cvxpylayers + cvxpy

    def _build_layer(self):
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "KellyPortfolioLayer requires 'cvxpy' and 'cvxpylayers'. "
                "Install with: pip install cvxpy cvxpylayers"
            ) from exc

        n = self.n_assets
        y_hat_param = cp.Parameter(n)
        q_sq_param = cp.Parameter(n, nonneg=True)
        w = cp.Variable(n)
        objective = cp.Maximize(y_hat_param @ w - (self.gamma / 2) * cp.sum(cp.multiply(q_sq_param, cp.square(w))))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        self._layer = CvxpyLayer(problem, parameters=[y_hat_param, q_sq_param], variables=[w])

    def forward(self, y_hat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat: [n_assets] point predictions.
            q: [n_assets] conformal interval half-widths (q_p from Eq. 17).
        Returns:
            [n_assets] portfolio weights w* (Eq. 20).
        """
        if self._layer is None:
            self._build_layer()
        (w_star,) = self._layer(y_hat, q ** 2)
        return w_star
