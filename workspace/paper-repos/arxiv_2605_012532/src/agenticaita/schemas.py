"""
schemas.py — Typed JSON contracts between AGENTICAITA agents.
Implements the inter-agent communication protocol described in Section 4.2.
All fields correspond exactly to the JSON contracts described in the paper.
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Agent I/O Contracts ──────────────────────────────────────────────────────

class AnalystOutput(BaseModel):
    """
    Output contract of the Analyst agent.
    Paper Section 4.2: "Respond ONLY in JSON: {signal, confidence, entry_price,
    stop_loss, take_profit, size_usd, reasoning}"
    """
    signal: Literal["long", "short", "wait"]
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size_usd: Optional[float] = None
    reasoning: str  # Stored verbatim in episodic memory

    def __repr__(self) -> str:
        return f"AnalystOutput(signal={self.signal}, conf={self.confidence:.2f})"


class GateResult(BaseModel):
    """Result of Risk Manager hard-gate Layer A checks (deterministic)."""
    passed: bool
    failed_gate: Optional[str] = None  # Which gate failed if not passed
    reason: Optional[str] = None

    def __repr__(self) -> str:
        return f"GateResult(passed={self.passed}, failed={self.failed_gate})"


class RiskManagerOutput(BaseModel):
    """
    Output contract of the Risk Manager agent (Layer B LLM output).
    Paper Section 4.2: "{approved: bool, size_usd: float, negotiation_summary: string}"
    """
    approved: bool
    size_usd: float
    negotiation_summary: str

    def __repr__(self) -> str:
        return f"RiskManagerOutput(approved={self.approved}, size=${self.size_usd:.2f})"


# ── Market Data Structures ───────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    """One OHLCV bar (1-minute, per Section 4.2)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class L2Level(BaseModel):
    price: float
    size: float


class L2Orderbook(BaseModel):
    """Live L2 orderbook snapshot."""
    asset: str
    timestamp: datetime
    bids: list[L2Level]
    asks: list[L2Level]

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


class MarketContext(BaseModel):
    """Full market context passed to the Analyst agent."""
    asset: str
    timestamp: datetime
    current_price: float
    ohlcv: list[OHLCVBar]       # 20-bar 1-minute OHLCV (Section 4.2)
    l2_orderbook: L2Orderbook
    funding_rate: float
    z_score: float               # Current AZTE z_score
    return_magnitude: float      # Current r_t
    omega: float                 # CBD composite score (Eq. 11)
    memory_briefing: str         # Episodic memory text for this asset


# ── Pipeline Events ──────────────────────────────────────────────────────────

class TriggerEvent(BaseModel):
    """Fired by AZTE when Eq. 3 is satisfied."""
    asset: str
    timestamp: datetime
    z_score: float
    return_magnitude: float
    triggered_by: Literal["z_score", "absolute_floor", "both"]


class ExecutionRecord(BaseModel):
    """Final record written to trades table after Executor runs."""
    asset: str
    timestamp: datetime
    signal: str
    analyst_confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    requested_size_usd: float
    approved_size_usd: float
    reasoning: str
    negotiation_summary: str
    mode: Literal["DRY_RUN", "LIVE"]
    order_id: Optional[str] = None  # None in DRY_RUN
    pnl: Optional[float] = None     # Filled post-close


class PipelineResult(BaseModel):
    """Summary result of a full SDP pipeline run."""
    invocation_id: str
    trigger: TriggerEvent
    outcome: Literal["executed", "analyst_abstain", "rm_rejected", "igp_busy", "error"]
    analyst_output: Optional[AnalystOutput] = None
    gate_result: Optional[GateResult] = None
    rm_output: Optional[RiskManagerOutput] = None
    execution_record: Optional[ExecutionRecord] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0


# ── Database Record Models ───────────────────────────────────────────────────

class PipelineEvent(BaseModel):
    """Entry for pipeline_log table."""
    timestamp: datetime
    asset: str
    event_type: str   # trigger, igp_busy, analyst_wait, rm_reject, executed, error
    detail: str


class OllamaCallRecord(BaseModel):
    """Entry for ollama_calls telemetry table."""
    timestamp: datetime
    agent: Literal["analyst", "risk_manager"]
    model: str
    system_prompt_len: int
    user_prompt_len: int
    response_len: int
    latency_ms: float
    success: bool
