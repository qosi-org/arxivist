"""
agents/risk_manager.py — Risk Manager Agent (Section 4.2)
Second stage of the Sequential Deliberative Pipeline (SDP).

Two-layer validation:
  Layer A: 4 deterministic hard gates (Eq. 4–7) — architectural guarantee
  Layer B: LLM contextual validation and position size calibration

Hard gates are enforced BEFORE any LLM call. No amount of persuasive
Analyst reasoning can override a gate failure.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

from ..schemas import AnalystOutput, GateResult, RiskManagerOutput
from ..config import RiskManagerConfig
from ..ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Section 4.2 — Risk Manager system prompt (paper excerpt + extension)
RM_SYSTEM_PROMPT = """You are the AgenticAITA Risk Manager. Your goal is Proportional Portfolio Balancing.
Calculate size_usd based on the Analyst's confidence (higher confidence → larger size, but never exceed the approved maximum).
Respond ONLY in JSON: {{"approved": true|false, "size_usd": float, "negotiation_summary": string}}.
Be conservative. Prefer smaller positions when uncertain."""


class RiskManagerAgent:
    """
    Risk Manager Agent — SDP Stage 2.
    Section 4.2 of arxiv:2605.12532.

    Layer A (deterministic hard gates, Eq. 4–7):
      4: signal in {long, short}
      5: confidence >= 0.60
      6: |entry - stop_loss| / entry <= 0.02
      7: size_usd <= 500

    Layer B (LLM): contextual validation + position sizing.
    The output JSON contract cannot be overridden by the Executor.
    """

    def __init__(self, cfg: RiskManagerConfig, llm: OllamaClient) -> None:
        self.cfg = cfg
        self.llm = llm

    def hard_gate_check(self, output: AnalystOutput) -> GateResult:
        """
        Layer A: 4 deterministic hard gates (Equations 4–7).
        Executed BEFORE any LLM call. Architectural guarantee.
        """
        # Eq. 4: signal validity
        if output.signal not in ("long", "short"):
            return GateResult(passed=False, failed_gate="eq4_signal", reason=f"signal={output.signal} not in {{long, short}}")

        # Eq. 5: confidence floor
        if output.confidence < self.cfg.confidence_gate:
            return GateResult(passed=False, failed_gate="eq5_confidence",
                              reason=f"confidence={output.confidence:.3f} < {self.cfg.confidence_gate}")

        # Eq. 6: stop-loss distance
        if output.entry_price is None or output.stop_loss is None:
            return GateResult(passed=False, failed_gate="eq6_stoploss", reason="entry_price or stop_loss is None")

        sl_distance = abs(output.entry_price - output.stop_loss) / output.entry_price
        if sl_distance > self.cfg.max_stop_loss_pct:
            return GateResult(passed=False, failed_gate="eq6_stoploss",
                              reason=f"SL distance={sl_distance:.4f} > {self.cfg.max_stop_loss_pct}")

        # Eq. 7: max position size
        if output.size_usd is None or output.size_usd > self.cfg.max_position_usd:
            return GateResult(passed=False, failed_gate="eq7_size",
                              reason=f"size_usd={output.size_usd} > ${self.cfg.max_position_usd}")

        return GateResult(passed=True)

    async def llm_validate(self, analyst_output: AnalystOutput) -> Optional[RiskManagerOutput]:
        """
        Layer B: LLM contextual validation and position size calibration.
        Only called if hard gates pass.
        """
        user_prompt = f"""Analyst signal for review:
Signal: {analyst_output.signal}
Confidence: {analyst_output.confidence}
Entry: {analyst_output.entry_price}
Stop-loss: {analyst_output.stop_loss}
Take-profit: {analyst_output.take_profit}
Proposed size: ${analyst_output.size_usd}
Analyst reasoning: {analyst_output.reasoning}

Review this signal and calibrate the position size proportionally to confidence.
Maximum allowed size: ${self.cfg.max_position_usd}"""

        try:
            response = await self.llm.complete(
                system_prompt=RM_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent="risk_manager",
            )
        except RuntimeError as e:
            logger.error(f"Risk Manager LLM call failed: {e}")
            return None

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            output = RiskManagerOutput(**data)
        except Exception as e:
            logger.error(f"Risk Manager JSON parse failed: {e}\nResponse: {response[:500]}")
            return None

        # Final size clamp (safety net)
        output.size_usd = min(output.size_usd, self.cfg.max_position_usd)

        logger.info(f"RiskManager: approved={output.approved}, size=${output.size_usd:.2f}")
        return output

    def __repr__(self) -> str:
        return f"RiskManagerAgent(conf_gate={self.cfg.confidence_gate}, max_size=${self.cfg.max_position_usd})"
