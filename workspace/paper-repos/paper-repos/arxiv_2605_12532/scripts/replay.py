#!/usr/bin/env python3
"""
scripts/replay.py — Replay pipeline audit log for verification.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.4
'Every execution is written to pipeline_log and trades, providing a complete
replayable audit record.'

Usage:
    python scripts/replay.py --db data/agenticaita.db
    python scripts/replay.py --db data/agenticaita.db --from-ts 2026-04-06T09:00:00
    python scripts/replay.py --db data/agenticaita.db --event-type pipeline_start
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiosqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay AGENTICAITA pipeline audit log")
    parser.add_argument("--db", type=str, default="data/agenticaita.db")
    parser.add_argument("--from-ts", type=str, default=None, help="ISO8601 start timestamp")
    parser.add_argument("--to-ts", type=str, default=None, help="ISO8601 end timestamp")
    parser.add_argument("--asset", type=str, default=None, help="Filter by asset symbol")
    parser.add_argument("--event-type", type=str, default=None, help="Filter by event type")
    parser.add_argument("--limit", type=int, default=100, help="Max events to show")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    conditions = []
    params = []
    if args.from_ts:
        conditions.append("timestamp >= ?")
        params.append(args.from_ts)
    if args.to_ts:
        conditions.append("timestamp <= ?")
        params.append(args.to_ts)
    if args.asset:
        conditions.append("asset = ?")
        params.append(args.asset)
    if args.event_type:
        conditions.append("event_type = ?")
        params.append(args.event_type)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT timestamp, event_type, asset, details FROM pipeline_log {where} ORDER BY timestamp LIMIT ?"
    params.append(args.limit)

    async with aiosqlite.connect(args.db) as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    print(f"{'TIMESTAMP':<25} {'EVENT':<25} {'ASSET':<15} DETAILS")
    print("-" * 90)
    for ts, evt, asset, details in rows:
        detail_str = ""
        if details:
            try:
                d = json.loads(details)
                detail_str = " ".join(f"{k}={v}" for k, v in list(d.items())[:3])
            except Exception:
                detail_str = details[:60]
        print(f"{ts:<25} {evt:<25} {(asset or ''):<15} {detail_str}")

    print(f"\n({len(rows)} events shown)")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
