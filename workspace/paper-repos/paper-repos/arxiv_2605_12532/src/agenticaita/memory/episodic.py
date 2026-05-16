"""
memory/episodic.py — Episodic Memory (SQLite WAL).

Paper: AGENTICAITA (arxiv:2605.12532), Section 3
Implements the persistent memory layer described in the system architecture:
  - trades: full decision records including agent reasoning traces
  - vol_history: 143,807 volatility samples across 117 monitored assets
  - pipeline_log: audit trail for all pipeline events
  - ollama_calls: LLM inference telemetry

The Analyst agent retrieves past trade reasoning from the same asset as a
'narrative episodic memory briefing' — a form of cross-episode context with
no equivalent in RL-based systems.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from agenticaita.pipeline.contracts import TradeRecord, TriggerEvent

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    signal          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    take_profit     REAL NOT NULL,
    size_usd        REAL NOT NULL,
    analyst_reasoning      TEXT NOT NULL,
    rm_negotiation_summary TEXT NOT NULL,
    execution_mode  TEXT NOT NULL,
    pnl_usd         REAL,
    closed_at       TEXT,
    z_score         REAL,
    omega_cbd       REAL
);

CREATE TABLE IF NOT EXISTS vol_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    return_mag  REAL NOT NULL,
    price       REAL
);

CREATE TABLE IF NOT EXISTS pipeline_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    asset       TEXT,
    details     TEXT
);

CREATE TABLE IF NOT EXISTS ollama_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    agent       TEXT NOT NULL,
    model       TEXT NOT NULL,
    prompt_len  INTEGER,
    response_len INTEGER,
    latency_ms  REAL
);
"""


class EpisodicMemory:
    """
    SQLite WAL-mode episodic memory store.

    Paper: Section 3 — 'Persistent memory is implemented as a SQLite database
    in Write-Ahead Logging (WAL) mode mounted on a Docker volume.'

    Provides cross-episode narrative memory: the Analyst's reasoning strings
    from past trades on the same asset are retrieved and injected into future
    invocations, accumulating experiential context across the session.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None

    def __repr__(self) -> str:
        return f"EpisodicMemory(db={self.db_path}, connected={self._db is not None})"

    async def connect(self) -> None:
        """Open the database connection and initialize schema."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info(f"[Memory] connected to {self.db_path} (WAL mode)")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Trades table
    # ------------------------------------------------------------------

    async def save_trade(self, record: TradeRecord) -> int:
        """Persist a full decision record to the trades table."""
        assert self._db is not None, "Call connect() first"
        cursor = await self._db.execute(
            """
            INSERT INTO trades
              (asset, timestamp, signal, confidence, entry_price, stop_loss, take_profit,
               size_usd, analyst_reasoning, rm_negotiation_summary, execution_mode,
               pnl_usd, closed_at, z_score, omega_cbd)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.asset,
                record.timestamp.isoformat(),
                record.signal.value,
                record.confidence,
                record.entry_price,
                record.stop_loss,
                record.take_profit,
                record.size_usd,
                record.analyst_reasoning,
                record.rm_negotiation_summary,
                record.execution_mode,
                record.pnl_usd,
                record.closed_at.isoformat() if record.closed_at else None,
                record.z_score,
                record.omega_cbd,
            ),
        )
        await self._db.commit()
        trade_id = cursor.lastrowid
        logger.debug(f"[Memory] saved trade id={trade_id} for {record.asset}")
        return trade_id

    async def get_past_trades(self, asset: str, n: int = 5) -> List[dict]:
        """
        Retrieve the n most recent trade reasoning strings for an asset.

        This implements the 'narrative episodic memory briefing' passed to
        the Analyst agent. The reasoning field contains the Analyst's verbatim
        prior deliberation on the same asset, providing cross-episode context.

        Paper: Section 4.2 — 'constituting a form of narrative episodic memory
        that accumulates experiential context across the session.'
        """
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT timestamp, signal, confidence, pnl_usd, analyst_reasoning
            FROM trades
            WHERE asset = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (asset, n),
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "signal": row[1],
                "confidence": row[2],
                "pnl_usd": row[3],
                "reasoning": row[4],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # vol_history table
    # ------------------------------------------------------------------

    async def append_vol(self, asset: str, return_mag: float, price: Optional[float] = None) -> None:
        """Append a return magnitude sample to vol_history."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO vol_history (asset, timestamp, return_mag, price) VALUES (?,?,?,?)",
            (asset, datetime.utcnow().isoformat(), return_mag, price),
        )
        await self._db.commit()

    async def get_prices(self, asset: str, W: int = 30) -> List[float]:
        """Retrieve the W most recent prices for an asset (for CBD correlation)."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT price FROM vol_history WHERE asset=? AND price IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (asset, W),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in reversed(rows)]

    async def get_returns(self, asset: str, W: int = 30) -> List[float]:
        """Retrieve the W most recent return magnitudes for AZTE hot-restart."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT return_mag FROM vol_history WHERE asset=? ORDER BY timestamp DESC LIMIT ?",
            (asset, W),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in reversed(rows)]

    # ------------------------------------------------------------------
    # pipeline_log table
    # ------------------------------------------------------------------

    async def log_event(self, event_type: str, asset: Optional[str] = None, details: Optional[dict] = None) -> None:
        """Write an event to the pipeline audit log."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO pipeline_log (timestamp, event_type, asset, details) VALUES (?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                event_type,
                asset,
                json.dumps(details) if details else None,
            ),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # ollama_calls table
    # ------------------------------------------------------------------

    async def log_ollama_call(
        self,
        agent: str,
        model: str,
        prompt_len: int,
        response_len: int,
        latency_ms: float,
    ) -> None:
        """Record LLM inference telemetry."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO ollama_calls (timestamp, agent, model, prompt_len, response_len, latency_ms) VALUES (?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), agent, model, prompt_len, response_len, latency_ms),
        )
        await self._db.commit()
