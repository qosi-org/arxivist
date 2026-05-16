"""
pipeline/sdp.py — Sequential Deliberative Pipeline (SDP).

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
The agentic core: orchestrates Analyst → Risk Manager → Executor.

Pipeline flow (from Figure 2):
  AZTE trigger → IGP lock → Analyst (signal or self-abstain)
               → Risk Manager Layer A (hard gates)
               → Risk Manager Layer B (LLM)
               → Executor (DRY_RUN or LIVE)
               → EpisodicMemory

Rates observed in 5-day live session:
  - Analyst self-abstain (wait): 8.3% of invocations (Nwait=13/157)
  - RM hard-gate rejection:      3.2% of all invocations (Nrej=5/157)
  - Agentic Friction F:          11.5% (Eq. 8: (Nrej + Nwait) / N)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from agenticaita.agents.analyst import AnalystAgent
from agenticaita.agents.executor import ExecutorAgent
from agenticaita.agents.risk_manager import RiskManagerAgent
from agenticaita.memory.episodic import EpisodicMemory
from agenticaita.pipeline.contracts import (
    AnalystContract,
    MarketContext,
    SignalType,
    TradeRecord,
    TriggerEvent,
)
from agenticaita.scoring.cbd import CBD

logger = logging.getLogger(__name__)


class PipelineOutcome:
    """Structured result from a single SDP invocation."""
    __slots__ = ("event", "outcome", "trade_record", "rejection_reason", "timestamp")

    def __init__(
        self,
        event: TriggerEvent,
        outcome: str,  # "traded" | "abstained" | "rejected_hard_gate" | "rejected_llm" | "error"
        trade_record: Optional[TradeRecord] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        self.event = event
        self.outcome = outcome
        self.trade_record = trade_record
        self.rejection_reason = rejection_reason
        self.timestamp = datetime.utcnow()

    def __repr__(self) -> str:
        return f"PipelineOutcome(asset={self.event.asset!r}, outcome={self.outcome!r})"


class SDP:
    """
    Sequential Deliberative Pipeline — the agentic core of AGENTICAITA.

    Paper: Section 4.2
    Operationalizes deliberative role separation: rather than asking a single LLM to
    simultaneously analyze, assess risk, and decide on execution (which induces
    confirmation bias and role blending), the decision is decomposed into a chain
    of three agents with distinct epistemic mandates.

    Args:
        analyst: AnalystAgent instance.
        risk_manager: RiskManagerAgent instance.
        executor: ExecutorAgent instance.
        memory: EpisodicMemory instance.
        cbd: CBD scoring module.
        btc_symbol: Symbol used as BTC reference for CBD.
    """

    def __init__(
        self,
        analyst: AnalystAgent,
        risk_manager: RiskManagerAgent,
        executor: ExecutorAgent,
        memory: EpisodicMemory,
        cbd: CBD,
        btc_symbol: str = "BTC",
    ) -> None:
        self.analyst = analyst
        self.risk_manager = risk_manager
        self.executor = executor
        self.memory = memory
        self.cbd = cbd
        self.btc_symbol = btc_symbol

        # Session counters (for Eq. 8: Agentic Friction F)
        self._n_total: int = 0
        self._n_wait: int = 0    # Analyst self-abstentions
        self._n_rej: int = 0     # Risk Manager hard-gate rejections
        self._n_traded: int = 0

    def __repr__(self) -> str:
        return (
            f"SDP(total={self._n_total}, traded={self._n_traded}, "
            f"friction={self.friction_rate:.3f})"
        )

    @property
    def friction_rate(self) -> float:
        """
        Agentic Friction F = (Nrej + Nwait) / N.

        Paper: Section 4.3, Eq. 8.
        F > 0 confirms genuine inter-agent negotiation vs. a naive pass-through system.
        """
        if self._n_total == 0:
            return 0.0
        return (self._n_rej + self._n_wait) / self._n_total

    async def run(
        self,
        trigger: TriggerEvent,
        ohlcv: list,
        l2: dict,
        funding_rate: Optional[float] = None,
    ) -> PipelineOutcome:
        """
        Execute one full SDP invocation for a triggered asset.

        This method should only be called after the IGP lock has been acquired.

        Args:
            trigger: TriggerEvent from AZTE.
            ohlcv: 20-bar 1-minute OHLCV candles.
            l2: Live L2 orderbook snapshot.
            funding_rate: Current perpetual funding rate.

        Returns:
            PipelineOutcome describing the result of this invocation.
        """
        self._n_total += 1
        asset = trigger.asset

        await self.memory.log_event("pipeline_start", asset, {
            "z_score": trigger.z_score,
            "return_mag": trigger.return_magnitude,
        })

        try:
            # --- CBD Score ---
            asset_prices = await self.memory.get_prices(asset, self.cbd.config.window_bars)
            btc_prices = await self.memory.get_prices(self.btc_symbol, self.cbd.config.window_bars)
            omega = self.cbd.score(trigger.z_score, asset_prices, btc_prices)

            # --- Episodic Memory Briefing ---
            past_trades = await self.memory.get_past_trades(asset, n=5)

            # --- Build Market Context ---
            ctx = MarketContext(
                asset=asset,
                ohlcv_1m=ohlcv,
                l2_orderbook=l2,
                funding_rate=funding_rate,
                omega_cbd=omega,
                episodic_memory=past_trades,
            )

            # =================================================================
            # Stage A: Analyst
            # =================================================================
            analyst_output: AnalystContract = await self.analyst.call(ctx, past_trades)

            if analyst_output.signal == SignalType.WAIT:
                # Analyst self-abstention — Nwait++
                self._n_wait += 1
                await self.memory.log_event("analyst_abstain", asset, {"reasoning": analyst_output.reasoning[:200]})
                logger.info(f"[SDP] {asset}: Analyst self-abstained (Nwait={self._n_wait})")
                return PipelineOutcome(trigger, "abstained", rejection_reason="analyst_wait")

            # =================================================================
            # Stage B: Risk Manager — Layer A (deterministic hard gates)
            # =================================================================
            gate_result = self.risk_manager.hard_gates(analyst_output)

            if not gate_result.passed:
                # Hard gate rejection — Nrej++
                self._n_rej += 1
                await self.memory.log_event("rm_rejected_hardgate", asset, {
                    "gate": gate_result.failed_gate,
                    "reason": gate_result.reason,
                })
                logger.info(f"[SDP] {asset}: RM hard-gate rejected ({gate_result.failed_gate}): {gate_result.reason}")
                return PipelineOutcome(
                    trigger, "rejected_hard_gate", rejection_reason=gate_result.reason
                )

            # =================================================================
            # Stage B: Risk Manager — Layer B (LLM contextual validation)
            # =================================================================
            rm_output = await self.risk_manager.llm_validate(analyst_output)

            if not rm_output.approved:
                await self.memory.log_event("rm_rejected_llm", asset, {
                    "summary": rm_output.negotiation_summary[:200]
                })
                logger.info(f"[SDP] {asset}: RM LLM disapproved: {rm_output.negotiation_summary[:100]}")
                return PipelineOutcome(trigger, "rejected_llm", rejection_reason=rm_output.negotiation_summary)

            # =================================================================
            # Stage C: Executor
            # =================================================================
            trade_record = await self.executor.execute(
                rm_output,
                analyst_output,
                asset,
                z_score=trigger.z_score,
                omega_cbd=omega,
            )

            trade_id = await self.memory.save_trade(trade_record)
            await self.memory.log_event("trade_executed", asset, {"trade_id": trade_id, "mode": trade_record.execution_mode})

            self._n_traded += 1
            logger.info(
                f"[SDP] {asset}: EXECUTED {analyst_output.signal.value.upper()} "
                f"@ {analyst_output.entry_price:.6f} size=${rm_output.size_usd:.2f} "
                f"[trade_id={trade_id}]"
            )
            return PipelineOutcome(trigger, "traded", trade_record=trade_record)

        except Exception as exc:
            await self.memory.log_event("pipeline_error", asset, {"error": str(exc)[:300]})
            logger.error(f"[SDP] {asset}: pipeline error: {exc}", exc_info=True)
            return PipelineOutcome(trigger, "error", rejection_reason=str(exc))

    def session_stats(self) -> dict:
        """Return session-level pipeline statistics including Agentic Friction."""
        return {
            "n_total": self._n_total,
            "n_wait": self._n_wait,
            "n_rejected": self._n_rej,
            "n_traded": self._n_traded,
            "agentic_friction_F": round(self.friction_rate, 4),
        }
