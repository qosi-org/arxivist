#!/usr/bin/env python3
"""
scripts/evaluate.py — Generate metrics report from trades database.

Paper: AGENTICAITA (arxiv:2605.12532), Section 5, Tables 3-7
Reproduces all reported metrics from the trades SQLite table, including
transaction cost sensitivity analysis (Table 7).

Usage:
    python scripts/evaluate.py --db data/agenticaita.db
    python scripts/evaluate.py --db data/agenticaita.db --cost-scenario realistic
    python scripts/evaluate.py --db data/agenticaita.db --out results/metrics.json
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiosqlite

from agenticaita.evaluation.cost_model import benchmark_comparison, sensitivity_analysis
from agenticaita.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AGENTICAITA session metrics")
    parser.add_argument("--db", type=str, default="data/agenticaita.db")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--cost-scenario", type=str, default="all",
        choices=["zero", "conservative", "realistic", "adverse", "all"],
        help="Transaction cost scenario for sensitivity analysis",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


async def load_trades(db_path: str) -> tuple:
    """Load all closed trades from the database."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT pnl_usd, signal, confidence, stop_loss, take_profit, entry_price, size_usd "
            "FROM trades WHERE pnl_usd IS NOT NULL ORDER BY timestamp"
        )
        rows = await cursor.fetchall()

        # Pipeline log counts
        cur2 = await db.execute(
            "SELECT COUNT(*) FROM pipeline_log WHERE event_type='pipeline_start'"
        )
        n_total = (await cur2.fetchone())[0]

        cur3 = await db.execute(
            "SELECT COUNT(*) FROM pipeline_log WHERE event_type='analyst_abstain'"
        )
        n_wait = (await cur3.fetchone())[0]

        cur4 = await db.execute(
            "SELECT COUNT(*) FROM pipeline_log WHERE event_type='rm_rejected_hardgate'"
        )
        n_rej = (await cur4.fetchone())[0]

    return rows, n_total, n_wait, n_rej


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-8s %(message)s")

    if not Path(args.db).exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    rows, n_total, n_wait, n_rej = await load_trades(args.db)

    if not rows:
        print("No closed trades found in database.")
        return

    pnl_series = [r[0] for r in rows]
    signals = [r[1] for r in rows]
    sl_pcts = [abs(r[3] - r[4]) / r[4] if r[4] > 0 else 0.0 for r in rows]  # |SL-entry|/entry
    tp_pcts = [abs(r[5] - r[4]) / r[4] if r[4] > 0 else 0.0 for r in rows]  # |TP-entry|/entry
    sizes = [r[6] for r in rows]
    mean_size = sum(sizes) / len(sizes) if sizes else 0.0
    total_notional = sum(sizes)

    metrics = compute_metrics(pnl_series, signals, n_total, n_wait, n_rej, sl_pcts, tp_pcts)
    print(str(metrics))

    # Transaction cost sensitivity (Table 7)
    print("\n=== Transaction Cost Sensitivity (Table 7 reproduction) ===")
    scenarios = sensitivity_analysis(metrics.net_pnl, len(pnl_series), mean_size)
    for name, data in scenarios.items():
        print(
            f"  {name:<15} {data['roundtrip_label']:>7}  "
            f"total cost: ${abs(data['total_cost_usd']):.2f}  "
            f"adj PnL: ${data['adj_net_pnl_usd']:.2f}"
        )

    result = {
        "metrics": {
            "total_trades": metrics.total_trades,
            "win_rate": round(metrics.win_rate, 4),
            "net_pnl_usd": round(metrics.net_pnl, 4),
            "profit_factor": round(metrics.profit_factor, 4),
            "max_drawdown_usd": round(metrics.max_drawdown, 4),
            "agentic_friction_F": round(metrics.agentic_friction_F, 4),
            "long_rate": round(metrics.long_rate, 4),
            "short_rate": round(metrics.short_rate, 4),
            "n_total_invocations": metrics.n_total_invocations,
        },
        "cost_sensitivity": scenarios,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nMetrics saved to: {args.out}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
