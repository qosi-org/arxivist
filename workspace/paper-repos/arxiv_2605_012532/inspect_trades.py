"""
inspect_trades.py — Query and export trades from episodic memory.
arxiv:2605.12532 (Letteri 2026)

Usage:
    python inspect_trades.py --db data/episodic_memory.db --output json
    python inspect_trades.py --db data/episodic_memory.db --asset BTC/USDT:USDT
"""
from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Query AGENTICAITA trade records")
    parser.add_argument("--db", default="data/episodic_memory.db")
    parser.add_argument("--asset", default=None, help="Filter by asset symbol")
    parser.add_argument("--output", choices=["json", "csv", "table"], default="table")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM trades"
    params = []
    if args.asset:
        query += " WHERE asset = ?"
        params.append(args.asset)
    query += f" ORDER BY timestamp DESC LIMIT {args.limit}"

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    if not rows:
        print("No trades found.")
        return

    if args.output == "json":
        print(json.dumps(rows, indent=2))
    elif args.output == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    else:
        # Simple table
        cols = ["id", "asset", "timestamp", "signal", "analyst_confidence", "approved_size_usd", "pnl", "mode"]
        header = " | ".join(f"{c:<20}" for c in cols)
        print(header)
        print("-" * len(header))
        for row in rows:
            line = " | ".join(f"{str(row.get(c, '')):<20}" for c in cols)
            print(line)


if __name__ == "__main__":
    main()
