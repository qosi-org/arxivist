"""
evaluation/metrics.py — Evaluation metrics for Gu, Kelly, Xiu (2020).

Implements all performance metrics reported in the paper:
  - R²_oos: out-of-sample R² vs zero forecast (Eq. 22, Section 1.8)
  - Diebold-Mariano test (Section 1.8, Eq. 23-24)
  - Variable importance: R² reduction and SSD (Section 1.9)
  - Sharpe ratio and Campbell-Thompson improvement (Eq. 25)
  - Maximum drawdown and portfolio turnover (Section 2.4.2)
  - Factor model alpha (Section 2.4.2)

Paper reference: Sections 1.8, 1.9, 2.4
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class ReturnMetrics:
    """Predictive performance metrics for return forecasts.

    Paper reference: Section 1.8 (performance evaluation) and
                     Section 1.9 (variable importance).
    """

    @staticmethod
    def oos_r2(r_actual: np.ndarray, r_pred: np.ndarray) -> float:
        """Out-of-sample R² benchmarked against zero forecast.

        Paper Equation (22):
            R²_oos = 1 - sum(r_it+1 - r_hat_it+1)² / sum(r_it+1²)

        CRITICAL: Denominator is sum of squared returns, NOT variance.
        This benchmarks against a constant-zero forecast, NOT historical mean.

        Paper Section 1.8: "A subtle but important aspect of our R² metric
        is that the denominator is the sum of squared excess returns without
        demeaning."

        Args:
            r_actual: [N] realized excess returns (test set)
            r_pred:   [N] predicted excess returns

        Returns:
            R²_oos as a percentage (multiply by 100 to match paper tables).
        """
        assert r_actual.shape == r_pred.shape
        ss_res = np.sum((r_actual - r_pred) ** 2)
        ss_tot = np.sum(r_actual ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def diebold_mariano(
        errors_1: np.ndarray,   # [T, N_t] prediction errors of model 1
        errors_2: np.ndarray,   # [T, N_t] prediction errors of model 2
        n_lags: int = 1,
    ) -> Tuple[float, float]:
        """Modified Diebold-Mariano test for pairwise model comparison.

        Paper Equations (23)-(24):
            d_12,t+1 = (1/n_3,t+1) * sum_i [(e^1_it+1)² - (e^2_it+1)²]
            DM_12 = d_bar_12 / sigma_d_bar_12

        DM ~ N(0,1) under null; positive → model 2 outperforms model 1.
        Paper uses Newey-West standard errors for d_bar.

        Paper reference: Section 1.8, Equations (23)-(24)

        Args:
            errors_1: [T, N_t] prediction errors for model 1
            errors_2: [T, N_t] prediction errors for model 2
            n_lags: Number of lags for Newey-West HAC estimator.

        Returns:
            (dm_stat, p_value) — positive stat means model 2 better than model 1.
        """
        # d_12,t = (1/N_t) * sum_i (e1_it^2 - e2_it^2) — per-period loss difference
        T = errors_1.shape[0]
        d = np.array([
            np.mean(errors_1[t]**2 - errors_2[t]**2)
            for t in range(T)
        ])
        d_bar = np.mean(d)

        # Newey-West HAC standard error for d
        nw_var = _newey_west_variance(d, n_lags)
        se = np.sqrt(nw_var / T)

        if se < 1e-12:
            return 0.0, 1.0

        dm_stat = d_bar / se
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
        return float(dm_stat), float(p_value)

    @staticmethod
    def variable_importance_r2(
        model: Any,
        Z: np.ndarray,
        R: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Variable importance via R² reduction when each predictor is zeroed.

        Paper Section 1.9: "the reduction in panel predictive R² from setting
        all values of predictor j to zero, while holding remaining model
        estimates fixed."

        VI_j = R²_oos(full) - R²_oos(z_j = 0)

        Args:
            model: Fitted model with .predict(Z) method.
            Z: [N, P] feature matrix (test set).
            R: [N] actual returns.
            feature_names: List of P feature names.

        Returns:
            Dict mapping feature name → importance score (higher = more important).
        """
        baseline_r2 = ReturnMetrics.oos_r2(R, model.predict(Z))
        importances = {}
        for j, name in enumerate(feature_names):
            Z_zeroed = Z.copy()
            Z_zeroed[:, j] = 0.0
            r2_j = ReturnMetrics.oos_r2(R, model.predict(Z_zeroed))
            importances[name] = float(baseline_r2 - r2_j)
        # Normalize to sum to 1 (within-model relative importance)
        total = sum(max(0, v) for v in importances.values())
        if total > 0:
            importances = {k: max(0, v) / total for k, v in importances.items()}
        return importances

    @staticmethod
    def variable_importance_ssd(
        model: Any,
        Z: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Variable importance via sum of squared partial derivatives (SSD).

        Paper Section 1.9 (Dimopoulos et al. 1995):
            SSD_j = sum_{i,t in T1} (∂g(z; theta) / ∂z_j |_{z=z_it})^2

        NOTE: Only applicable to differentiable models (NNs, GLMs).
              For tree models, use variable_importance_r2 with mean decrease in impurity.

        Args:
            model: Fitted model with .get_gradients(Z) method (NeuralNetModel).
            Z: [N, P] feature matrix (training set T1).
            feature_names: List of P feature names.

        Returns:
            Dict mapping feature name → SSD importance score (normalized).
        """
        grads = model.get_gradients(Z)  # [N, P]
        ssd = np.sum(grads ** 2, axis=0)  # [P]

        total = ssd.sum()
        if total > 0:
            ssd = ssd / total

        return {name: float(ssd[j]) for j, name in enumerate(feature_names)}


class PortfolioMetrics:
    """Portfolio-level performance metrics.

    Paper reference: Section 2.4
    """

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 12,
    ) -> float:
        """Annualized Sharpe ratio (mean/std, no risk-free adjustment since returns are excess).

        Args:
            returns: [T] monthly excess returns
            annualize: If True, annualize.
            periods_per_year: 12 for monthly data.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(returns) == 0 or returns.std() < 1e-12:
            return 0.0
        sr = returns.mean() / returns.std()
        if annualize:
            sr *= np.sqrt(periods_per_year)
        return float(sr)

    @staticmethod
    def campbell_thompson_sr(sr_bah: float, r2_oos: float) -> float:
        """Sharpe ratio for an investor exploiting predictive forecasts.

        Paper Equation (25) (Campbell & Thompson 2008):
            SR* = sqrt(SR² + R²/(1-R²))

        Translates predictive R²_oos into Sharpe ratio improvement.

        Args:
            sr_bah: Buy-and-hold Sharpe ratio.
            r2_oos: Out-of-sample R² (as a fraction, NOT percentage).

        Returns:
            Improved Sharpe ratio for market-timing investor.
        """
        if r2_oos >= 1.0:
            r2_oos = 0.9999
        return float(np.sqrt(sr_bah**2 + r2_oos / (1.0 - r2_oos)))

    @staticmethod
    def max_drawdown(cum_log_returns: np.ndarray) -> float:
        """Maximum drawdown of cumulative log return series.

        Paper: MaxDD = max_{0 ≤ t1 ≤ t2 ≤ T} (Y_t1 - Y_t2)

        Args:
            cum_log_returns: [T] cumulative log returns (Y_t = sum of log(1+r) up to t)

        Returns:
            Maximum drawdown (positive number).
        """
        running_max = np.maximum.accumulate(cum_log_returns)
        drawdowns = running_max - cum_log_returns
        return float(drawdowns.max())

    @staticmethod
    def monthly_turnover(
        weights: np.ndarray,  # [T, N] portfolio weights each month
        returns: np.ndarray,  # [T, N] stock returns each month
    ) -> float:
        """Average monthly portfolio turnover.

        Paper formula:
            Turnover = (1/T) * sum_t sum_i |w_it+1 - w_it*(1+r_it+1)/sum_j(w_jt*(1+r_jt+1))|

        Args:
            weights: [T, N] portfolio weights at start of each period
            returns: [T, N] realized stock returns

        Returns:
            Average monthly turnover (fraction of portfolio rebalanced).
        """
        T = weights.shape[0]
        turnovers = []
        for t in range(T - 1):
            w_t = weights[t]
            r_t = returns[t]
            # Weight drift after returns
            gross_ret = 1.0 + np.nansum(w_t * r_t)
            w_t_plus = w_t * (1 + r_t) / (gross_ret + 1e-10)
            w_t_next = weights[t + 1]
            turnovers.append(np.nansum(np.abs(w_t_next - w_t_plus)))
        return float(np.mean(turnovers)) if turnovers else 0.0

    @staticmethod
    def factor_alpha(
        portfolio_returns: np.ndarray,  # [T] strategy returns
        factor_returns: np.ndarray,     # [T, K] factor returns (FF5+mom = 6 factors)
    ) -> Tuple[float, float, float]:
        """Regression alpha with respect to a multi-factor model.

        Paper Table 8: "FF5+Mom alpha" using Fama-French 5 factors + momentum.

        Args:
            portfolio_returns: [T] monthly strategy excess returns
            factor_returns: [T, K] factor returns

        Returns:
            (alpha, t_stat, r_squared)
        """
        T = len(portfolio_returns)
        X = np.column_stack([np.ones(T), factor_returns])
        try:
            result = np.linalg.lstsq(X, portfolio_returns, rcond=None)
            coef = result[0]
            alpha = float(coef[0])
            r_hat = X @ coef
            resid = portfolio_returns - r_hat
            sigma2 = np.var(resid, ddof=X.shape[1])
            try:
                xtx_inv = np.linalg.inv(X.T @ X)
                se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
                t_stat = alpha / se_alpha if se_alpha > 1e-10 else 0.0
            except np.linalg.LinAlgError:
                t_stat = 0.0
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((portfolio_returns - portfolio_returns.mean())**2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            return alpha, t_stat, r2
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0

    @staticmethod
    def aggregate_results(episode_r2s: List[float]) -> Dict[str, float]:
        """Aggregate R² statistics across test periods."""
        arr = np.array(episode_r2s)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(arr),
        }


def _newey_west_variance(x: np.ndarray, n_lags: int) -> float:
    """Newey-West HAC variance estimator for a time series.

    Used in Diebold-Mariano test for autocorrelation-robust standard errors.

    Args:
        x: [T] time series
        n_lags: Number of lags.

    Returns:
        HAC variance estimate.
    """
    T = len(x)
    x_centered = x - x.mean()
    gamma_0 = np.sum(x_centered**2) / T
    var = gamma_0
    for h in range(1, n_lags + 1):
        gamma_h = np.sum(x_centered[h:] * x_centered[:-h]) / T
        weight = 1.0 - h / (n_lags + 1)
        var += 2.0 * weight * gamma_h
    return max(0.0, var)
