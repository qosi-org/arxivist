"""
pipeline/contracts.py — Typed JSON contracts for inter-agent communication.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
Each agent in the SDP communicates via strict typed contracts, validated with Pydantic.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    WAIT = "wait"


class AnalystContract(BaseModel):
    """
    Output contract of the Analyst agent (Section 4.2).

    The Analyst MUST respond ONLY in this JSON format:
    {signal, confidence, entry_price, stop_loss, take_profit, size_usd, reasoning}
    The 'reasoning' field is stored verbatim in episodic memory for future sessions.
    """
    signal: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0, description="[0,1] directional confidence")
    entry_price: float = Field(..., gt=0.0)
    stop_loss: float = Field(..., gt=0.0)
    take_profit: float = Field(..., gt=0.0)
    size_usd: float = Field(..., gt=0.0)
    reasoning: str = Field(..., min_length=10, description="Verbatim reasoning stored in episodic memory")

    @field_validator("signal", mode="before")
    @classmethod
    def normalize_signal(cls, v: str) -> str:
        return v.lower().strip()

    def __repr__(self) -> str:
        return (
            f"AnalystContract(signal={self.signal}, conf={self.confidence:.2f}, "
            f"entry={self.entry_price:.4f}, sl={self.stop_loss:.4f}, tp={self.take_profit:.4f})"
        )


class HardGateResult(BaseModel):
    """Result of the Risk Manager deterministic Layer A hard-gate check."""
    passed: bool
    failed_gate: Optional[Literal["signal", "confidence", "risk_pct", "size_usd"]] = None
    reason: str

    def __repr__(self) -> str:
        return f"HardGateResult(passed={self.passed}, gate={self.failed_gate}, reason={self.reason!r})"


class RMContract(BaseModel):
    """
    Output contract of the Risk Manager agent (Section 4.2).

    After passing Layer A hard gates, the RM produces this typed contract.
    The Executor CANNOT override fields in this contract.
    """
    approved: bool
    size_usd: float = Field(..., ge=0.0)
    negotiation_summary: str

    def __repr__(self) -> str:
        return f"RMContract(approved={self.approved}, size_usd={self.size_usd:.2f})"


class TradeRecord(BaseModel):
    """Full decision record written to the 'trades' SQLite table."""
    id: Optional[int] = None
    asset: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    size_usd: float
    analyst_reasoning: str
    rm_negotiation_summary: str
    execution_mode: str            # DRY_RUN | LIVE
    pnl_usd: Optional[float] = None
    closed_at: Optional[datetime] = None
    z_score: Optional[float] = None
    omega_cbd: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"TradeRecord(asset={self.asset!r}, signal={self.signal}, "
            f"entry={self.entry_price:.4f}, pnl={self.pnl_usd})"
        )


class TriggerEvent(BaseModel):
    """Emitted by AZTE when an asset crosses the trigger threshold (Eq. 3)."""
    asset: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    z_score: float
    return_magnitude: float
    triggered_by: Literal["z_score", "r_floor", "both"]

    def __repr__(self) -> str:
        return (
            f"TriggerEvent(asset={self.asset!r}, z={self.z_score:.3f}, "
            f"r={self.return_magnitude:.4f}, by={self.triggered_by!r})"
        )


class MarketContext(BaseModel):
    """Full market context passed to the Analyst agent."""
    asset: str
    ohlcv_1m: list   # 20 bars of [open, high, low, close, volume]
    l2_orderbook: dict  # {"bids": [[price, size], ...], "asks": [[price, size], ...]}
    funding_rate: Optional[float] = None
    market_snapshot: dict = Field(default_factory=dict)
    omega_cbd: float = 0.0
    episodic_memory: list = Field(default_factory=list)  # Past trade reasoning strings

    def __repr__(self) -> str:
        return f"MarketContext(asset={self.asset!r}, omega={self.omega_cbd:.3f})"
