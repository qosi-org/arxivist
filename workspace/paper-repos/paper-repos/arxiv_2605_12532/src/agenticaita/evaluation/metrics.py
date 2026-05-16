"""
evaluation/metrics.py — Trading performance metrics.

Paper: AGENTICAITA (arxiv:2605.12532), Section 5, Tables 3-6
Implements Agentic Friction (Eq. 8), win rate, profit factor, drawdown,
and the transaction cost sensitivity model (Eqs. 12-13).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SessionMetrics:
    """
    Full metric set from Table 5 (Section 5).
    All field names mirror the paper's reported metrics.
    """
    total_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    profit_factor: float
    max_drawdown: float
    mean_win: float
    mean_loss: float
    mean_rr: float
    break_even_wr: float
    agentic_friction_F: float
    n_total_invocations: int
    n_analyst_wait: int
    n_rm_rejected: int
    n_long: int
    n_short: int
    long_rate: float
    short_rate: float

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "AGENTICAITA Session Metrics",
            "=" * 50,
            f"Total trades:         {self.total_trades}",
            f"Win rate:             {self.win_rate:.2%}",
            f"Gross profit:         ${self.gross_profit:.2f}",
            f"Gross loss:           ${self.gross_loss:.2f}",
            f"Net PnL:              ${self.net_pnl:.2f}",
            f"Profit factor:        {self.profit_factor:.3f}",
            f"Max drawdown:         ${self.max_drawdown:.2f}",
            f"Mean win:             ${self.mean_win:.2f}",
            f"Mean loss:            ${self.mean_loss:.2f}",
            f"Mean R:R:             {self.mean_rr:.2f}",
            f"Break-even WR@RR:     {self.break_even_wr:.2%}",
            f"Long signal rate:     {self.long_rate:.2%}",
            f"Short signal rate:    {self.short_rate:.2%}",
            f"",
            f"Agentic Friction F:   {self.agentic_friction_F:.2%}",
            f"  Total invocations:  {self.n_total_invocations}",
            f"  Analyst abstained:  {self.n_analyst_wait} ({self.n_analyst_wait/max(self.n_total_invocations,1):.1%})",
            f"  RM rejected:        {self.n_rm_rejected} ({self.n_rm_rejected/max(self.n_total_invocations,1):.1%})",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_metrics(
    pnl_series: List[float],
    signals: List[str],
    n_total: int,
    n_wait: int,
    n_rejected: int,
    sl_pcts: Optional[List[float]] = None,
    tp_pcts: Optional[List[float]] = None,
) -> SessionMetrics:
    """
    Compute all session metrics from the trades table.

    Args:
        pnl_series: List of realized PnL values per closed trade.
        signals: List of signal types ("long" or "short") per trade.
        n_total: Total pipeline invocations (N in Eq. 8).
        n_wait: Analyst self-abstentions (Nwait in Eq. 8).
        n_rejected: RM hard-gate rejections (Nrej in Eq. 8).
        sl_pcts: Stop-loss percentages per trade for R:R computation.
        tp_pcts: Take-profit percentages per trade for R:R computation.
    """
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = gross_profit - gross_loss
    win_rate = len(wins) / len(pnl_series) if pnl_series else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    mean_win = sum(wins) / len(wins) if wins else 0.0
    mean_loss = sum(losses) / len(losses) if losses else 0.0

    # Max cumulative drawdown (peak-to-trough on equity curve)
    equity = [0.0]
    for p in pnl_series:
        equity.append(equity[-1] + p)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    # R:R and break-even WR
    if sl_pcts and tp_pcts and len(sl_pcts) == len(tp_pcts):
        mean_sl = sum(sl_pcts) / len(sl_pcts)
        mean_tp = sum(tp_pcts) / len(tp_pcts)
        mean_rr = mean_tp / mean_sl if mean_sl > 0 else 0.0
    else:
        mean_rr = 0.0
    break_even_wr = 1.0 / (1.0 + mean_rr) if mean_rr > 0 else 0.5

    # Agentic Friction (Eq. 8)
    friction = (n_rejected + n_wait) / n_total if n_total > 0 else 0.0

    n_long = signals.count("long")
    n_short = signals.count("short")
    total_signals = len(signals)

    return SessionMetrics(
        total_trades=len(pnl_series),
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=net_pnl,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        mean_win=mean_win,
        mean_loss=mean_loss,
        mean_rr=mean_rr,
        break_even_wr=break_even_wr,
        agentic_friction_F=friction,
        n_total_invocations=n_total,
        n_analyst_wait=n_wait,
        n_rm_rejected=n_rejected,
        n_long=n_long,
        n_short=n_short,
        long_rate=n_long / total_signals if total_signals > 0 else 0.0,
        short_rate=n_short / total_signals if total_signals > 0 else 0.0,
    )
