"""
pipeline.py — Sequential Deliberative Pipeline (SDP) Orchestrator (Section 4.2)
Coordinates the three-agent chain: Analyst → Risk Manager → Executor.
Enforces the IGP gating protocol (Section 4.4).

This module implements Definition 1 from the paper:
  SDP = (A_analyst, A_rm, A_exec) with typed JSON contracts between stages.
"""
from __future__ import annotations
import logging
import time
import uuid
from datetime import datetime
from typing import Literal, Optional

from .schemas import TriggerEvent, PipelineResult, PipelineEvent
from .igp import InferenceGatingProtocol
from .cbd import CorrelationBreakDiversification
from .memory import EpisodicMemory
from .market_data import MarketDataFeed
from .agents.analyst import AnalystAgent
from .agents.risk_manager import RiskManagerAgent
from .agents.executor import ExecutorAgent
from .config import AgenticAITAConfig

logger = logging.getLogger(__name__)


class SequentialDeliberativePipeline:
    """
    Sequential Deliberative Pipeline (SDP).
    Section 4.2 of arxiv:2605.12532.

    Three-agent chain with strict ordering and typed JSON contracts.
    IGP ensures at most one pipeline executes at a time.
    Agentic friction F (Eq. 8) is tracked via outcome statistics.
    """

    def __init__(
        self,
        cfg: AgenticAITAConfig,
        igp: InferenceGatingProtocol,
        cbd: CorrelationBreakDiversification,
        memory: EpisodicMemory,
        market_data: MarketDataFeed,
        analyst: AnalystAgent,
        risk_manager: RiskManagerAgent,
        executor: ExecutorAgent,
    ) -> None:
        self.cfg = cfg
        self.igp = igp
        self.cbd = cbd
        self.memory = memory
        self.market_data = market_data
        self.analyst = analyst
        self.risk_manager = risk_manager
        self.executor = executor

        # Runtime statistics for friction tracking (Eq. 8)
        self.stats = {
            "total_invocations": 0,
            "analyst_abstain": 0,
            "rm_rejected": 0,
            "executed": 0,
            "igp_busy": 0,
            "errors": 0,
        }

    async def handle_trigger(self, trigger: TriggerEvent) -> PipelineResult:
        """
        Full pipeline run for one trigger event.
        IGP gating is checked first; if busy, returns immediately.
        """
        invocation_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        # IGP: non-blocking acquire (Definition 2)
        acquired = await self.igp.acquire(asset=trigger.asset)
        if not acquired:
            self.stats["igp_busy"] += 1
            await self.memory.log_pipeline_event(PipelineEvent(
                timestamp=datetime.utcnow(),
                asset=trigger.asset,
                event_type="igp_busy",
                detail=f"trigger discarded z={trigger.z_score:.3f}",
            ))
            return PipelineResult(
                invocation_id=invocation_id,
                trigger=trigger,
                outcome="igp_busy",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        self.stats["total_invocations"] += 1

        try:
            result = await self._run_pipeline(invocation_id, trigger, t0)
        except Exception as e:
            logger.error(f"Pipeline error [{invocation_id}]: {e}", exc_info=True)
            self.stats["errors"] += 1
            result = PipelineResult(
                invocation_id=invocation_id,
                trigger=trigger,
                outcome="error",
                error_message=str(e),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        finally:
            await self.igp.release()

        return result

    async def _run_pipeline(
        self, invocation_id: str, trigger: TriggerEvent, t0: float
    ) -> PipelineResult:
        """Inner pipeline: Analyst → RiskManager → Executor."""
        asset = trigger.asset
        logger.info(f"Pipeline [{invocation_id}]: START {asset} z={trigger.z_score:.3f}")

        # ── CBD score ────────────────────────────────────────────────────────
        omega = self.cbd.compute_omega(
            z_score=trigger.z_score,
            asset=asset,
            window=self.cfg.azte.rolling_window,
        )

        # ── Market context ───────────────────────────────────────────────────
        memory_briefing = await self.memory.get_briefing(asset)
        ctx = await self.market_data.build_market_context(
            asset=asset,
            z_score=trigger.z_score,
            r_t=trigger.return_magnitude,
            omega=omega,
            memory_briefing=memory_briefing,
        )

        # ── Stage 1: Analyst ─────────────────────────────────────────────────
        analyst_output = await self.analyst.analyze(ctx, memory_briefing)

        if analyst_output is None:
            self.stats["errors"] += 1
            return PipelineResult(invocation_id=invocation_id, trigger=trigger, outcome="error",
                                  error_message="Analyst returned None (parse failure)",
                                  duration_ms=(time.monotonic() - t0) * 1000)

        if analyst_output.signal == "wait":
            # Analyst self-abstention — contributes to agentic friction F (Eq. 8)
            self.stats["analyst_abstain"] += 1
            logger.info(f"Pipeline [{invocation_id}]: Analyst ABSTAIN ({asset})")
            await self.memory.log_pipeline_event(PipelineEvent(
                timestamp=datetime.utcnow(), asset=asset,
                event_type="analyst_wait", detail=f"conf={analyst_output.confidence:.2f}",
            ))
            return PipelineResult(invocation_id=invocation_id, trigger=trigger,
                                  outcome="analyst_abstain", analyst_output=analyst_output,
                                  duration_ms=(time.monotonic() - t0) * 1000)

        # ── Stage 2a: Risk Manager hard gates (Layer A) ──────────────────────
        gate_result = self.risk_manager.hard_gate_check(analyst_output)

        if not gate_result.passed:
            # Hard gate rejection — contributes to agentic friction F (Eq. 8)
            self.stats["rm_rejected"] += 1
            logger.info(f"Pipeline [{invocation_id}]: RM REJECT {gate_result.failed_gate} ({asset})")
            await self.memory.log_pipeline_event(PipelineEvent(
                timestamp=datetime.utcnow(), asset=asset,
                event_type="rm_reject", detail=f"gate={gate_result.failed_gate}: {gate_result.reason}",
            ))
            return PipelineResult(invocation_id=invocation_id, trigger=trigger,
                                  outcome="rm_rejected", analyst_output=analyst_output,
                                  gate_result=gate_result,
                                  duration_ms=(time.monotonic() - t0) * 1000)

        # ── Stage 2b: Risk Manager LLM validation (Layer B) ─────────────────
        rm_output = await self.risk_manager.llm_validate(analyst_output)

        if rm_output is None or not rm_output.approved:
            self.stats["rm_rejected"] += 1
            reason = "LLM parse failure" if rm_output is None else rm_output.negotiation_summary
            logger.info(f"Pipeline [{invocation_id}]: RM LLM REJECT ({asset}): {reason}")
            await self.memory.log_pipeline_event(PipelineEvent(
                timestamp=datetime.utcnow(), asset=asset,
                event_type="rm_reject", detail=f"llm: {reason}",
            ))
            return PipelineResult(invocation_id=invocation_id, trigger=trigger,
                                  outcome="rm_rejected", analyst_output=analyst_output,
                                  gate_result=gate_result, rm_output=rm_output,
                                  duration_ms=(time.monotonic() - t0) * 1000)

        # ── Stage 3: Executor ────────────────────────────────────────────────
        exec_record = await self.executor.execute(asset, analyst_output, rm_output)
        self.stats["executed"] += 1

        logger.info(f"Pipeline [{invocation_id}]: EXECUTED {asset} {analyst_output.signal} ${rm_output.size_usd:.2f}")
        return PipelineResult(
            invocation_id=invocation_id,
            trigger=trigger,
            outcome="executed",
            analyst_output=analyst_output,
            gate_result=gate_result,
            rm_output=rm_output,
            execution_record=exec_record,
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    def friction_rate(self) -> float:
        """
        Eq. 8: F = (N_rej + N_wait) / N
        Agentic friction rate over all invocations.
        """
        n = self.stats["total_invocations"]
        if n == 0:
            return 0.0
        return (self.stats["rm_rejected"] + self.stats["analyst_abstain"]) / n

    def __repr__(self) -> str:
        return f"SDP(executed={self.stats['executed']}, friction={self.friction_rate():.1%})"
