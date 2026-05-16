"""
agents/executor.py — Executor Agent (Section 4.2)
Third and final stage of the Sequential Deliberative Pipeline (SDP).

The only agent with external side effects:
- DRY_RUN: logs decision trace without placing an order
- LIVE: routes through privacy-preserving channel (Tor+VPN, Section 4.7)

Every execution is written to pipeline_log and trades (complete replayable audit record).
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import Literal, Optional

from ..schemas import (
    AnalystOutput, RiskManagerOutput, ExecutionRecord, PipelineEvent
)
from ..exchange import ExchangeAdapter

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    Executor Agent — SDP Stage 3.
    Section 4.2 of arxiv:2605.12532.

    Epistemic mandate: execute approved decisions with full audit trail.
    DRY_RUN mode suppresses order placement (used in proof-of-concept validation).
    """

    def __init__(
        self,
        exchange: Optional[ExchangeAdapter],
        db=None,
        mode: Literal["DRY_RUN", "LIVE"] = "DRY_RUN",
    ) -> None:
        self.exchange = exchange
        self.db = db
        self.mode = mode

    async def execute(
        self,
        asset: str,
        analyst_output: AnalystOutput,
        rm_output: RiskManagerOutput,
    ) -> ExecutionRecord:
        """
        Execute an approved trade decision.
        In DRY_RUN: log without placing order.
        In LIVE: route through exchange adapter (requires Tor safety gate).
        """
        order_id: Optional[str] = None

        if self.mode == "LIVE":
            order_id = await self._place_live_order(asset, analyst_output, rm_output)
        else:
            # DRY_RUN: no external side effects
            logger.info(f"DRY_RUN: would place {analyst_output.signal} {asset} ${rm_output.size_usd:.2f}")

        record = ExecutionRecord(
            asset=asset,
            timestamp=datetime.utcnow(),
            signal=analyst_output.signal,
            analyst_confidence=analyst_output.confidence,
            entry_price=analyst_output.entry_price or 0.0,
            stop_loss=analyst_output.stop_loss or 0.0,
            take_profit=analyst_output.take_profit or 0.0,
            requested_size_usd=analyst_output.size_usd or 0.0,
            approved_size_usd=rm_output.size_usd,
            reasoning=analyst_output.reasoning,
            negotiation_summary=rm_output.negotiation_summary,
            mode=self.mode,
            order_id=order_id,
            pnl=None,  # Filled post-close
        )

        # Persist to episodic memory
        if self.db:
            await self.db.store_trade(record)
            await self.db.log_pipeline_event(PipelineEvent(
                timestamp=datetime.utcnow(),
                asset=asset,
                event_type="executed",
                detail=f"mode={self.mode} signal={analyst_output.signal} size=${rm_output.size_usd:.2f}",
            ))

        return record

    async def _place_live_order(
        self,
        asset: str,
        analyst_output: AnalystOutput,
        rm_output: RiskManagerOutput,
    ) -> Optional[str]:
        """
        LIVE order placement via privacy-preserving exchange adapter.
        Section 4.7: safety gate requires tor_active AND exchange_reachable.
        """
        if self.exchange is None:
            raise RuntimeError("ExchangeAdapter not configured for LIVE mode")

        # Section 4.7: safety gate
        if not await self.exchange.safety_gate_check():
            raise RuntimeError("LIVE order blocked: Tor or exchange not available")

        from ..exchange import OrderSpec
        order = OrderSpec(
            asset=asset,
            side=analyst_output.signal,
            size_usd=rm_output.size_usd,
            entry_price=analyst_output.entry_price,
            stop_loss=analyst_output.stop_loss,
            take_profit=analyst_output.take_profit,
        )
        result = await self.exchange.place_order(order)
        return result.order_id if result else None

    def __repr__(self) -> str:
        return f"ExecutorAgent(mode={self.mode})"
