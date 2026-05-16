"""
agents/executor.py — Executor Agent.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
The Executor is the only agent with external side effects.

  DRY_RUN: logs the full decision trace without placing an order.
  LIVE:     routes through the privacy-preserving dual-channel (Section 4.7).
            Requires Tor SOCKS5h proxy active AND exchange reachable.

Safety gate (Section 4.7):
  safe = tor_active AND exchange_reachable
  Execution is blocked if safe == False.

Every execution is written to pipeline_log and trades, providing a complete
replayable audit record (Section 4.2).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from agenticaita.pipeline.contracts import RMContract, TradeRecord, SignalType

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    Executor agent — third and final stage of the SDP.

    Paper: Section 4.2 and Section 4.7.

    In DRY_RUN mode, the executor suppresses order placement but fully
    executes all other pipeline stages (LLM inference, risk management,
    position monitoring, logging). This enables risk-free live evaluation
    under live market conditions — the methodology used in the 5-day session.

    In LIVE mode, the executor verifies the dual-channel safety gate before
    routing the order via Tor+VPN SOCKS5h.
    """

    def __init__(
        self,
        mode: str = "DRY_RUN",
        tor_socks_host: str = "localhost",
        tor_socks_port: int = 9050,
    ) -> None:
        assert mode in ("DRY_RUN", "LIVE"), f"mode must be DRY_RUN or LIVE, got {mode!r}"
        self.mode = mode
        self.tor_socks_host = tor_socks_host
        self.tor_socks_port = tor_socks_port

    def __repr__(self) -> str:
        return f"ExecutorAgent(mode={self.mode!r})"

    async def execute(
        self,
        rm_contract: RMContract,
        analyst_signal: "AnalystContract",  # noqa: F821 — forward ref
        asset: str,
        z_score: Optional[float] = None,
        omega_cbd: Optional[float] = None,
    ) -> TradeRecord:
        """
        Execute the approved trading decision.

        Paper: Section 4.2 — 'Every execution is written to pipeline_log and
        trades, providing a complete replayable audit record.'

        Args:
            rm_contract: Approved contract from Risk Manager.
            analyst_signal: Original Analyst contract (for entry/sl/tp/reasoning).
            asset: Asset symbol.
            z_score: Z-score that triggered the pipeline.
            omega_cbd: CBD composite score used in Analyst reasoning.

        Returns:
            TradeRecord ready to be saved to the trades table.
        """
        assert rm_contract.approved, "ExecutorAgent.execute called with unapproved contract"

        if self.mode == "LIVE":
            await self._live_execute(rm_contract, analyst_signal, asset)
        else:
            logger.info(
                f"[Executor] DRY_RUN: {asset} {analyst_signal.signal.value.upper()} "
                f"@ {analyst_signal.entry_price:.6f} size=${rm_contract.size_usd:.2f}"
            )

        record = TradeRecord(
            asset=asset,
            timestamp=datetime.utcnow(),
            signal=analyst_signal.signal,
            confidence=analyst_signal.confidence,
            entry_price=analyst_signal.entry_price,
            stop_loss=analyst_signal.stop_loss,
            take_profit=analyst_signal.take_profit,
            size_usd=rm_contract.size_usd,
            analyst_reasoning=analyst_signal.reasoning,
            rm_negotiation_summary=rm_contract.negotiation_summary,
            execution_mode=self.mode,
            pnl_usd=None,  # Populated later when position closes
            z_score=z_score,
            omega_cbd=omega_cbd,
        )
        return record

    async def _live_execute(self, rm_contract: RMContract, analyst_signal, asset: str) -> None:
        """
        LIVE order execution via dual-channel routing (Section 4.7).

        Public:  Agent → direct HTTPS → DEX API  (market data)
        Private: Agent → SOCKS5h → Tor circuit → VPN → DEX API  (order placement)

        Safety gate: safe = tor_active AND exchange_reachable
        Blocks execution if gate is False.

        NOTE: This is a STUB implementation.
        SIR ambiguity: DEX exchange not identified; cannot implement exchange-specific API.
        Replace with your target DEX adapter before enabling LIVE mode.
        """
        # Safety gate (Section 4.7)
        tor_active = await self._check_tor()
        exchange_reachable = await self._check_exchange()
        safe = tor_active and exchange_reachable

        if not safe:
            raise RuntimeError(
                f"[Executor] LIVE safety gate FAILED: tor_active={tor_active}, "
                f"exchange_reachable={exchange_reachable}. Order NOT placed."
            )

        # STUB: exchange-specific order placement
        # TODO: implement DEX REST/WebSocket order API
        # Route through: aiohttp connector with SOCKS5h proxy at tor_socks_host:tor_socks_port
        logger.warning(
            "[Executor] LIVE mode STUB — exchange API not implemented. "
            "See SIR ambiguity: DEX exchange identity unknown."
        )
        raise NotImplementedError(
            "STUB: Implement exchange-specific order placement. "
            "Route via SOCKS5h proxy at "
            f"{self.tor_socks_host}:{self.tor_socks_port}"
        )

    async def _check_tor(self) -> bool:
        """Check if Tor proxy is active and reachable."""
        import asyncio
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.tor_socks_host, self.tor_socks_port),
                timeout=3.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _check_exchange(self) -> bool:
        """Check if the DEX exchange is reachable via direct HTTPS. STUB."""
        # TODO: implement health-check ping to target DEX API
        logger.debug("[Executor] exchange reachability check: STUB — returning True for DRY_RUN")
        return True
