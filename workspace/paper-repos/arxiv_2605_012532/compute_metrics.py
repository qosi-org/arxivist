"""
compute_metrics.py — Reproduce Table 5 performance metrics from trades DB.
arxiv:2605.12532 (Letteri 2026)

Computes: win rate, net PnL, gross profit/loss, profit factor,
max drawdown, mean win, mean loss, mean risk/reward, agentic friction,
and benchmark alpha vs BTC buy-and-hold.

Usage:
    python compute_metrics.py --db data/episodic_memory.db --output results/metrics.json
"""
from __future__ import annotations
import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Optional


def load_trades(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp ASC"
    )
    trades = [dict(row) for row in cur.fetchall()]
    conn.close()
    return trades


def load_pipeline_log(db_path: str) -> dict[str, int]:
    """Count pipeline events for friction calculation (Eq. 8)."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT event_type, COUNT(*) as n FROM pipeline_log GROUP BY event_type"
    )
    counts = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return counts


def compute_metrics(trades: list[dict], pipeline_counts: dict[str, int]) -> dict:
    """Compute all Table 5 metrics."""
    if not trades:
        return {"error": "No closed trades found"}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(pnls)
    n_wins = len(wins)
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    net_pnl = gross_profit + gross_loss  # gross_loss is negative
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # Drawdown (simple running max)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    mean_win = sum(wins) / len(wins) if wins else 0.0
    mean_loss = sum(losses) / len(losses) if losses else 0.0

    # Mean risk/reward: mean_win / |mean_loss|
    mean_rr = abs(mean_win / mean_loss) if mean_loss != 0 else float("inf")

    # Breakeven WR at this R:R
    breakeven_wr = 1.0 / (1.0 + mean_rr) if mean_rr != float("inf") else 0.0

    # Eq. 8: Agentic friction
    n_wait = pipeline_counts.get("analyst_wait", 0)
    n_rm_reject = pipeline_counts.get("rm_reject", 0)
    n_executed = pipeline_counts.get("executed", 0)
    n_total = n_executed + n_wait + n_rm_reject
    friction = (n_wait + n_rm_reject) / n_total if n_total > 0 else 0.0

    # Notional traded
    notional_values = [t.get("approved_size_usd", 0) for t in trades]
    total_notional = sum(notional_values)

    # Binomial p-value (one-sided test H0: WR = 0.50)
    try:
        from scipy.stats import binom_test
        p_value = binom_test(n_wins, n, 0.5, alternative="greater")
    except ImportError:
        # Rough normal approximation
        if n > 0:
            z = (n_wins / n - 0.5) / math.sqrt(0.25 / n)
            import math as _math
            p_value = 0.5 * (1 - _math.erf(z / _math.sqrt(2)))
        else:
            p_value = 1.0

    return {
        "n_trades": n,
        "win_rate": round(n_wins / n, 4),
        "net_pnl_usd": round(net_pnl, 4),
        "gross_profit_usd": round(gross_profit, 4),
        "gross_loss_usd": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_usd": round(max_dd, 4),
        "mean_win_usd": round(mean_win, 4),
        "mean_loss_usd": round(mean_loss, 4),
        "mean_risk_reward": round(mean_rr, 2),
        "breakeven_win_rate": round(breakeven_wr, 4),
        "total_notional_usd": round(total_notional, 2),
        "net_return_pct": round(net_pnl / total_notional * 100, 4) if total_notional else 0,
        "agentic_friction_rate": round(friction, 4),
        "n_analyst_abstain": n_wait,
        "n_rm_rejected": n_rm_reject,
        "n_total_invocations": n_total,
        "binomial_p_value_wr50": round(p_value, 4),
        "paper_reported": {
            "n_trades": 139,
            "win_rate": 0.518,
            "net_pnl_usd": -15.07,
            "profit_factor": 0.841,
            "friction_rate": 0.115,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AGENTICAITA Table 5 metrics")
    parser.add_argument("--db", default="data/episodic_memory.db")
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    print(f"Loading trades from {args.db}...")
    trades = load_trades(args.db)
    pipeline_counts = load_pipeline_log(args.db)

    print(f"Computing metrics for {len(trades)} closed trades...")
    metrics = compute_metrics(trades, pipeline_counts)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Table 5 Metrics ──────────────────────────")
    for k, v in metrics.items():
        if k != "paper_reported":
            print(f"  {k:<35} {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
