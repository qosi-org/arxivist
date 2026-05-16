"""
memory.py — Episodic Memory (Section 3, Section 4.2)
Persistent SQLite database (WAL mode) storing:
  - trades: full decision records with agent reasoning traces
  - vol_history: rolling volatility samples for hot-restart
  - pipeline_log: audit trail of all pipeline events
  - ollama_calls: LLM inference telemetry

The reasoning field in trades is retrieved as narrative episodic memory
briefing for future Analyst invocations on the same asset (Section 4.2).
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .schemas import ExecutionRecord, PipelineEvent, OllamaCallRecord

logger = logging.getLogger(__name__)

CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    signal TEXT NOT NULL,
    analyst_confidence REAL,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    requested_size_usd REAL,
    approved_size_usd REAL,
    reasoning TEXT,
    negotiation_summary TEXT,
    mode TEXT NOT NULL,
    order_id TEXT,
    pnl REAL
);
"""

CREATE_VOL_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS vol_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL NOT NULL,
    z_score REAL,
    r_t REAL
);
"""

CREATE_PIPELINE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT
);
"""

CREATE_OLLAMA_CALLS_SQL = """
CREATE TABLE IF NOT EXISTS ollama_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt_len INTEGER,
    user_prompt_len INTEGER,
    response_len INTEGER,
    latency_ms REAL,
    success INTEGER
);
"""


class EpisodicMemory:
    """
    Persistent episodic memory via SQLite WAL-mode.
    Section 3 of arxiv:2605.12532.

    Stores all trade decisions with reasoning traces. The reasoning text
    constitutes narrative episodic memory retrieved for future Analyst
    invocations on the same asset.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize_db(self) -> None:
        """Create tables and enable WAL mode."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(CREATE_TRADES_SQL)
        await self._conn.execute(CREATE_VOL_HISTORY_SQL)
        await self._conn.execute(CREATE_PIPELINE_LOG_SQL)
        await self._conn.execute(CREATE_OLLAMA_CALLS_SQL)
        await self._conn.commit()
        logger.info(f"EpisodicMemory: initialized at {self.db_path} (WAL mode)")

    async def store_trade(self, record: ExecutionRecord) -> None:
        """Persist a completed trade record (Section 4.2)."""
        await self._conn.execute(
            """INSERT INTO trades
               (asset, timestamp, signal, analyst_confidence, entry_price, stop_loss,
                take_profit, requested_size_usd, approved_size_usd, reasoning,
                negotiation_summary, mode, order_id, pnl)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.asset,
                record.timestamp.isoformat(),
                record.signal,
                record.analyst_confidence,
                record.entry_price,
                record.stop_loss,
                record.take_profit,
                record.requested_size_usd,
                record.approved_size_usd,
                record.reasoning,
                record.negotiation_summary,
                record.mode,
                record.order_id,
                record.pnl,
            ),
        )
        await self._conn.commit()
        logger.debug(f"Trade stored: {record.asset} {record.signal}")

    async def get_briefing(self, asset: str, limit: int = 5) -> str:
        """
        Retrieve narrative episodic memory briefing for asset.
        Section 4.2: reasoning field stored verbatim, retrieved for future invocations.
        Returns formatted string injected into Analyst system prompt.
        """
        cursor = await self._conn.execute(
            """SELECT timestamp, signal, analyst_confidence, pnl, reasoning
               FROM trades WHERE asset=? ORDER BY timestamp DESC LIMIT ?""",
            (asset, limit),
        )
        rows = await cursor.fetchall()
        if not rows:
            return f"No prior trades recorded for {asset}."

        lines = [f"Past {len(rows)} trade(s) on {asset}:"]
        for ts, signal, conf, pnl, reasoning in rows:
            pnl_str = f"PnL=${pnl:.2f}" if pnl is not None else "PnL=open"
            lines.append(f"  [{ts[:16]}] {signal} conf={conf:.2f} {pnl_str}")
            lines.append(f"    Reasoning: {reasoning[:200]}...")
        return "\n".join(lines)

    async def store_vol_sample(
        self, asset: str, price: float, z_score: Optional[float], r_t: float
    ) -> None:
        """Persist volatility sample for hot-restart capability."""
        await self._conn.execute(
            "INSERT INTO vol_history (asset, timestamp, price, z_score, r_t) VALUES (?,?,?,?,?)",
            (asset, datetime.utcnow().isoformat(), price, z_score, r_t),
        )
        await self._conn.commit()

    async def get_vol_history(self, asset: str, limit: int = 30) -> list[dict]:
        """Retrieve recent vol samples for AZTE hot-restart."""
        cursor = await self._conn.execute(
            "SELECT price, z_score, r_t FROM vol_history WHERE asset=? ORDER BY id DESC LIMIT ?",
            (asset, limit),
        )
        rows = await cursor.fetchall()
        return [{"price": r[0], "z_score": r[1], "r_t": r[2]} for r in reversed(rows)]

    async def log_pipeline_event(self, event: PipelineEvent) -> None:
        """Write entry to pipeline_log audit trail."""
        await self._conn.execute(
            "INSERT INTO pipeline_log (timestamp, asset, event_type, detail) VALUES (?,?,?,?)",
            (event.timestamp.isoformat(), event.asset, event.event_type, event.detail),
        )
        await self._conn.commit()

    async def log_ollama_call(self, call: OllamaCallRecord) -> None:
        """Write LLM inference telemetry entry."""
        await self._conn.execute(
            """INSERT INTO ollama_calls
               (timestamp, agent, model, system_prompt_len, user_prompt_len,
                response_len, latency_ms, success)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                call.timestamp.isoformat(),
                call.agent,
                call.model,
                call.system_prompt_len,
                call.user_prompt_len,
                call.response_len,
                call.latency_ms,
                1 if call.success else 0,
            ),
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    def __repr__(self) -> str:
        return f"EpisodicMemory(path={self.db_path})"
