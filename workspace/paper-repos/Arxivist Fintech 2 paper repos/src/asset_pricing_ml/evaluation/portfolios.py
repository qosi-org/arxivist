"""
evaluation/portfolios.py — Portfolio construction from ML forecasts.

Implements the portfolio analysis from Section 2.4 of Gu, Kelly, Xiu (2020):
  - Bottom-up portfolio forecasts (Section 2.4.1)
  - ML-sorted decile long-short portfolios (Section 2.4.2)
  - Market timing strategy (Campbell & Thompson 2008)

Paper reference: Section 2.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from asset_pricing_ml.evaluation.metrics import PortfolioMetrics


@dataclass
class PortfolioReturns:
    """Monthly returns for a portfolio strategy."""
    name: str
    monthly_returns: np.ndarray  # [T]
    cum_log_returns: np.ndarray  # [T]
    sharpe_ratio: float
    max_drawdown: float

    @classmethod
    def from_returns(cls, name: str, monthly_returns: np.ndarray) -> "PortfolioReturns":
        cum = np.cumsum(np.log1p(monthly_returns))
        sr = PortfolioMetrics.sharpe_ratio(monthly_returns)
        dd = PortfolioMetrics.max_drawdown(cum)
        return cls(name=name, monthly_returns=monthly_returns,
                   cum_log_returns=cum, sharpe_ratio=sr, max_drawdown=dd)


class PortfolioConstructor:
    """Builds investment portfolios from model return forecasts.

    Paper reference: Section 2.4.1 (prespecified portfolios) and
                     Section 2.4.2 (ML-sorted portfolios).
    """

    @staticmethod
    def bottom_up_forecast(
        r_hat: np.ndarray,   # [N] stock-level return predictions
        weights: np.ndarray, # [N] portfolio weights (e.g. S&P 500 value weights)
    ) -> float:
        """Construct portfolio-level return forecast from stock-level predictions.

        Paper Section 2.4.1, Equation (26):
            r_hat_p_t+1 = sum_i w^p_it * r_hat_it+1

        Args:
            r_hat: [N] predicted stock excess returns
            weights: [N] ex-ante portfolio weights (known at forecast time)

        Returns:
            Scalar portfolio return forecast.
        """
        assert r_hat.shape == weights.shape
        w_norm = weights / (weights.sum() + 1e-10)
        return float(np.dot(w_norm, r_hat))

    @staticmethod
    def decile_sort(
        r_hat: np.ndarray,     # [N] predicted returns
        r_actual: np.ndarray,  # [N] realized returns
        mkt_cap: np.ndarray,   # [N] market capitalization
        n_deciles: int = 10,
        weighting: str = "value",
    ) -> Dict[int, Dict]:
        """Sort stocks into deciles by predicted return and compute portfolio stats.

        Paper Section 2.4.2: "sort stocks into deciles based on each model's
        forecast and reconstitute portfolios each month using value weights."

        Args:
            r_hat: [N] predicted returns
            r_actual: [N] realized returns
            mkt_cap: [N] market caps for value weighting
            n_deciles: Number of decile portfolios (default 10)
            weighting: 'value' (value-weighted) or 'equal'

        Returns:
            Dict mapping decile (1=lowest, 10=highest pred) → portfolio stats.
        """
        N = len(r_hat)
        # Assign each stock to a decile based on predicted return rank
        ranks = r_hat.argsort().argsort()  # ordinal rank 0..N-1
        decile_assignments = (ranks * n_deciles // N).clip(0, n_deciles - 1) + 1

        results = {}
        for d in range(1, n_deciles + 1):
            mask = decile_assignments == d
            if mask.sum() == 0:
                continue
            r_d = r_actual[mask]
            r_hat_d = r_hat[mask]
            w_d = mkt_cap[mask] if weighting == "value" else np.ones(mask.sum())
            w_d = w_d / w_d.sum()

            results[d] = {
                "predicted_mean": float(np.dot(w_d, r_hat_d)),
                "realized_mean": float(np.dot(w_d, r_d)),
                "realized_std": float(r_d.std()),
                "n_stocks": int(mask.sum()),
            }
        return results

    @staticmethod
    def long_short_portfolio(
        r_hat_series: List[np.ndarray],   # List of [N_t] predictions per month
        r_actual_series: List[np.ndarray], # List of [N_t] realized returns per month
        mkt_cap_series: List[np.ndarray],  # List of [N_t] market caps per month
        weighting: str = "value",
        n_deciles: int = 10,
    ) -> PortfolioReturns:
        """Long top decile, short bottom decile portfolio over time.

        Paper Section 2.4.2: "a zero-net-investment portfolio that buys the
        highest expected return decile and sells the lowest."

        Best result from paper: NN3 value-weighted H-L Sharpe ratio = 1.35.

        Args:
            r_hat_series: Monthly lists of predicted returns
            r_actual_series: Monthly lists of realized returns
            mkt_cap_series: Monthly lists of market caps
            weighting: 'value' or 'equal'
            n_deciles: Number of deciles

        Returns:
            PortfolioReturns for the long-short H-L decile spread.
        """
        monthly_rets = []
        for r_hat, r_actual, mkt_cap in zip(r_hat_series, r_actual_series, mkt_cap_series):
            deciles = PortfolioConstructor.decile_sort(
                r_hat, r_actual, mkt_cap, n_deciles, weighting
            )
            if n_deciles in deciles and 1 in deciles:
                long_ret = deciles[n_deciles]["realized_mean"]
                short_ret = deciles[1]["realized_mean"]
                monthly_rets.append(long_ret - short_ret)
            else:
                monthly_rets.append(0.0)

        return PortfolioReturns.from_returns("H-L Decile Spread", np.array(monthly_rets))

    @staticmethod
    def market_timing(
        r_hat_series: np.ndarray,   # [T] monthly portfolio return forecasts
        r_actual_series: np.ndarray, # [T] monthly realized portfolio returns
        max_leverage: float = 1.5,
        allow_short: bool = False,
    ) -> PortfolioReturns:
        """Market timing strategy: scale position with forecast.

        Paper Section 2.4.1 (Campbell & Thompson 2008 strategy):
        "scaling up/down positions each month as expected returns rise/fall,
        while imposing a maximum leverage constraint of 50% and excluding
        short sales for long-only portfolios."

        Args:
            r_hat_series: [T] predicted portfolio returns
            r_actual_series: [T] realized portfolio returns
            max_leverage: Maximum position size (1.5 = 150%)
            allow_short: Whether to allow short positions

        Returns:
            PortfolioReturns for the market timing strategy.
        """
        T = len(r_hat_series)
        monthly_rets = np.zeros(T)
        for t in range(T):
            # Scale position by predicted return (normalized to have unit variance)
            r_hat_std = np.std(r_hat_series[:t+1]) if t > 0 else 1.0
            position = r_hat_series[t] / (r_hat_std + 1e-8)
            position = np.clip(position, -max_leverage if allow_short else 0.0, max_leverage)
            monthly_rets[t] = position * r_actual_series[t]

        return PortfolioReturns.from_returns("Market Timing", monthly_rets)

    def __repr__(self) -> str:
        return "PortfolioConstructor()"
