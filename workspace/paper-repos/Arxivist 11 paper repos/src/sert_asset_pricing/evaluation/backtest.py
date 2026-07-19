"""
Backtesting engine.

Implements Section 5.3's trading-strategy backtests:
- Sign-signal (trend-following): long-only, open when sign(pred)==sign(actual)==+1,
  close on sign flip.
- Softmax trading-signal filter: applies softmax to sign signals, filters out the
  worst 50% of monthly signals across all stocks (Section 5.3).
- Equal-weighted (EW) and value-weighted (VW; weights proportional to market cap).
- Static transaction cost (50bps per paper) vs dynamic transaction cost
  (turnover_rate * 20bps for large-cap stocks, Section 5.3).
- Performance metrics: annualized return, Sharpe ratio, Sortino ratio, maximum
  drawdown (MDD), turnover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Backtester:
    """Runs sign-signal / softmax-filtered backtests over predicted vs actual returns.

    Args:
        transaction_cost_static_bps: static per-trade cost in basis points (paper: 50).
        transaction_cost_dynamic_bps: turnover-scaled cost in basis points (paper: 20,
            "large-cap-specific rate").
        softmax_filter_pct: fraction of worst monthly signals to filter out (paper: 0.5).
        periods_per_year: for annualization (monthly data -> 12).
    """

    transaction_cost_static_bps: float = 50.0
    transaction_cost_dynamic_bps: float = 20.0
    softmax_filter_pct: float = 0.5
    periods_per_year: int = 12

    def _sign_signals(self, preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Long-only sign signal: 1 when sign(pred)==sign(actual)==+1, else 0.

        Args:
            preds: [T, N] predicted returns (T months, N stocks).
            actuals: [T, N] actual returns.

        Returns:
            [T, N] binary position indicator.
        """
        pred_sign = np.sign(preds)
        actual_sign = np.sign(actuals)
        long_signal = (pred_sign > 0) & (actual_sign > 0)
        return long_signal.astype(np.float64)

    def _softmax_filter(self, preds: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """Apply softmax over predicted returns cross-sectionally and drop the worst
        `softmax_filter_pct` fraction of signals each month (Section 5.3).

        Args:
            preds: [T, N] predicted returns.
            signals: [T, N] binary sign signals to filter.

        Returns:
            [T, N] filtered signals (zeroed-out where filtered).
        """
        t, n = preds.shape
        filtered = signals.copy()
        keep_n = max(1, int(round(n * (1 - self.softmax_filter_pct))))
        for i in range(t):
            row = preds[i]
            # numerically stable softmax
            exp_row = np.exp(row - np.nanmax(row))
            probs = exp_row / np.nansum(exp_row)
            # keep only the top `keep_n` by softmax probability among currently-active signals
            active_idx = np.where(signals[i] > 0)[0]
            if len(active_idx) == 0:
                continue
            active_probs = probs[active_idx]
            order = np.argsort(-active_probs)
            keep_idx = active_idx[order[:keep_n]]
            drop_idx = np.setdiff1d(active_idx, keep_idx)
            filtered[i, drop_idx] = 0.0
        return filtered

    def _weights(self, signals: np.ndarray, market_caps: np.ndarray | None, scheme: str) -> np.ndarray:
        """Compute portfolio weights given active signals and a weighting scheme.

        Args:
            signals: [T, N] binary/soft active-position indicator.
            market_caps: [T, N] market caps for value-weighting, or None for equal-weight.
            scheme: "equal" or "value".

        Returns:
            [T, N] portfolio weights (rows sum to 1 where any signal is active, else 0).
        """
        t, n = signals.shape
        weights = np.zeros_like(signals, dtype=np.float64)
        for i in range(t):
            active = signals[i] > 0
            if not active.any():
                continue
            if scheme == "equal":
                weights[i, active] = 1.0 / active.sum()
            elif scheme == "value":
                assert market_caps is not None, "market_caps required for value-weighted scheme"
                caps = market_caps[i, active]
                weights[i, active] = caps / caps.sum()
            else:
                raise ValueError(f"Unknown weighting scheme: {scheme!r}")
        return weights

    def _transaction_costs(self, weights: np.ndarray, tc_mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-period transaction cost drag and turnover series.

        Args:
            weights: [T, N] portfolio weights over time.
            tc_mode: "static" (flat bps per period) or "dynamic" (turnover * dynamic bps).

        Returns:
            (cost_drag [T], turnover [T]) both as fractional (not bps) series.
        """
        t = weights.shape[0]
        turnover = np.zeros(t)
        turnover[1:] = np.abs(weights[1:] - weights[:-1]).sum(axis=1) / 2.0

        if tc_mode == "static":
            cost_drag = np.full(t, self.transaction_cost_static_bps / 1e4)
            cost_drag[0] = 0.0
        elif tc_mode == "dynamic":
            cost_drag = turnover * (self.transaction_cost_dynamic_bps / 1e4)
        else:
            raise ValueError(f"Unknown tc_mode: {tc_mode!r}")
        return cost_drag, turnover

    @staticmethod
    def _sharpe(returns: np.ndarray, periods_per_year: int) -> tuple[float, float]:
        if returns.std(ddof=1) == 0 or len(returns) < 2:
            return 0.0, 0.0
        sr = returns.mean() / returns.std(ddof=1)
        return float(sr), float(sr * np.sqrt(periods_per_year))

    @staticmethod
    def _sortino(returns: np.ndarray, periods_per_year: int) -> tuple[float, float]:
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
        if downside_std == 0 or len(returns) < 2:
            return 0.0, 0.0
        so = returns.mean() / downside_std
        return float(so), float(so * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(cum_returns: np.ndarray) -> float:
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return float(np.min(drawdown))

    def run(
        self,
        preds: np.ndarray,
        actuals: np.ndarray,
        weighting: str = "equal",
        tc_mode: str = "static",
        softmax_filter: bool = False,
        market_caps: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run a full backtest and return the paper's performance table columns.

        Args:
            preds: [T, N] predicted returns.
            actuals: [T, N] actual returns.
            weighting: "equal" or "value".
            tc_mode: "static" or "dynamic".
            softmax_filter: whether to apply the softmax trading-signal filter.
            market_caps: [T, N] market caps, required if weighting=="value".

        Returns:
            Dict with keys: AR (annualized return), SR (Sharpe), Ann_SR, SO (Sortino),
            Ann_SO, MDD (max drawdown), turnover (mean turnover), portfolio_returns [T].
        """
        assert preds.shape == actuals.shape, "preds and actuals must have matching shape"

        signals = self._sign_signals(preds, actuals)
        if softmax_filter:
            signals = self._softmax_filter(preds, signals)

        weights = self._weights(signals, market_caps, weighting)
        cost_drag, turnover = self._transaction_costs(weights, tc_mode)

        gross_returns = (weights * actuals).sum(axis=1)
        net_returns = gross_returns - cost_drag

        cum_returns = np.cumprod(1 + net_returns)
        ar = float(cum_returns[-1] ** (self.periods_per_year / len(net_returns)) - 1) if len(net_returns) > 0 else 0.0
        sr, ann_sr = self._sharpe(net_returns, self.periods_per_year)
        so, ann_so = self._sortino(net_returns, self.periods_per_year)
        mdd = self._max_drawdown(cum_returns)

        return {
            "AR": ar,
            "SR": sr,
            "Ann_SR": ann_sr,
            "SO": so,
            "Ann_SO": ann_so,
            "MDD": mdd,
            "turnover": float(turnover.mean()),
            "portfolio_returns": net_returns,
            "cumulative_returns": cum_returns,
        }

    def __repr__(self) -> str:
        return (
            f"Backtester(static_bps={self.transaction_cost_static_bps}, "
            f"dynamic_bps={self.transaction_cost_dynamic_bps})"
        )
