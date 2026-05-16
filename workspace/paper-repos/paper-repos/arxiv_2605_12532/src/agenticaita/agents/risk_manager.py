"""
agents/risk_manager.py — Risk Manager Agent (Hybrid Gate).

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
Implements the two-layer validation:
  Layer A: 4 deterministic hard gates (Eqs. 4-7) — architecture guarantee
  Layer B: LLM contextual validation + position size calibration

Key architectural guarantee (from paper):
  'This is an architectural guarantee — no amount of persuasive reasoning by
   the Analyst can override a gate failure.'

Hard gates (Section 4.2):
  signal ∈ {long, short}                     (Eq. 4)
  confidence ≥ 0.60                           (Eq. 5)
  |entry − stop_loss| / entry ≤ 0.02         (Eq. 6)
  size_usd ≤ 500                              (Eq. 7)
"""
from __future__ import annotations

import logging

from agenticaita.agents.base import LLMAgent
from agenticaita.pipeline.contracts import AnalystContract, HardGateResult, RMContract, SignalType
from agenticaita.utils.config import LLMConfig, RiskManagerConfig

logger = logging.getLogger(__name__)

RM_SYSTEM_PROMPT = """You are the AgenticAITA Risk Manager. Your goal is Proportional Portfolio Balancing.
Calculate size_usd based on the Analyst's confidence. Respond ONLY in JSON: {"approved": <bool>, "size_usd": <float>, "negotiation_summary": "<string>"}.
Your role is risk oversight — you may reduce size_usd or disapprove if contextual risk is too high.
Do not include any text outside the JSON object."""


class RiskManagerAgent(LLMAgent):
    """
    Risk Manager agent — second stage of the SDP.

    Paper: Section 4.2 — hybrid gate with deterministic Layer A and LLM Layer B.

    Layer A is executed BEFORE any LLM call. Failure terminates the pipeline
    immediately regardless of what the LLM might say. This ensures deterministic
    safety bounds independent of LLM stochasticity.

    Layer B (LLM) calibrates position size and performs contextual validation.
    The output is a typed JSON contract that the Executor cannot override.

    Observed metrics (5-day session):
      - 3.2% hard-gate rejection rate (of all invocations)
      - 3.5% rejection rate (of invocations reaching the RM)
    """

    def __init__(self, llm_config: LLMConfig, rm_config: RiskManagerConfig) -> None:
        super().__init__(llm_config, RM_SYSTEM_PROMPT, "RiskManager")
        self.rm_config = rm_config

    def hard_gates(self, signal: AnalystContract) -> HardGateResult:
        """
        Layer A: 4 deterministic hard gates.

        Paper: Section 4.2, Eqs. 4-7. Executed before ANY LLM call.
        Failure at any gate triggers immediate rejection.

        Returns HardGateResult with passed=True only if ALL gates pass.
        """
        # Eq. 4: signal ∈ {long, short}
        if signal.signal == SignalType.WAIT:
            return HardGateResult(
                passed=False,
                failed_gate="signal",
                reason=f"signal='wait' must not reach RM (Eq. 4); analyst should self-abstain",
            )

        # Eq. 5: confidence ≥ min_confidence (0.60)
        if signal.confidence < self.rm_config.min_confidence:
            return HardGateResult(
                passed=False,
                failed_gate="confidence",
                reason=f"confidence {signal.confidence:.3f} < {self.rm_config.min_confidence} (Eq. 5)",
            )

        # Eq. 6: |entry − stop_loss| / entry ≤ max_risk_pct (2%)
        risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if risk_pct > self.rm_config.max_risk_pct:
            return HardGateResult(
                passed=False,
                failed_gate="risk_pct",
                reason=f"risk {risk_pct:.4f} > {self.rm_config.max_risk_pct} (Eq. 6)",
            )

        # Eq. 7: size_usd ≤ max_size_usd ($500)
        if signal.size_usd > self.rm_config.max_size_usd:
            return HardGateResult(
                passed=False,
                failed_gate="size_usd",
                reason=f"size_usd {signal.size_usd:.2f} > {self.rm_config.max_size_usd} (Eq. 7)",
            )

        return HardGateResult(
            passed=True,
            failed_gate=None,
            reason=f"all hard gates passed (conf={signal.confidence:.2f}, risk={risk_pct:.4f}, size={signal.size_usd:.2f})",
        )

    async def llm_validate(self, signal: AnalystContract) -> RMContract:
        """
        Layer B: LLM contextual validation and position size calibration.

        Called only if Layer A passes. Calibrates size_usd based on Analyst
        confidence using proportional portfolio balancing logic.

        Paper: 'If Layer A passes, Layer B invokes the LLM for contextual
        validation and position size calibration. The output is a typed JSON
        contract that the Executor cannot override.'
        """
        user_message = self._build_rm_prompt(signal)
        raw, latency_ms = await self._call_ollama(user_message)

        parsed = self._parse_json_response(raw)
        try:
            contract = RMContract(**parsed)
        except Exception as e:
            raise ValueError(f"[RiskManager] contract validation failed: {e}\nParsed: {parsed}")

        # Safety: cap size_usd even if LLM exceeds it
        if contract.size_usd > self.rm_config.max_size_usd:
            logger.warning(
                f"[RiskManager] LLM returned size_usd={contract.size_usd:.2f} > "
                f"max={self.rm_config.max_size_usd}; capping."
            )
            contract = RMContract(
                approved=contract.approved,
                size_usd=self.rm_config.max_size_usd,
                negotiation_summary=contract.negotiation_summary + " [size capped by hard gate]",
            )

        logger.info(
            f"[RiskManager] approved={contract.approved}, "
            f"size={contract.size_usd:.2f}, latency={latency_ms:.0f}ms"
        )
        return contract

    def _build_rm_prompt(self, signal: AnalystContract) -> str:
        return f"""Analyst Signal to review:
  signal: {signal.signal.value}
  confidence: {signal.confidence:.4f}
  entry_price: {signal.entry_price:.6f}
  stop_loss: {signal.stop_loss:.6f}
  take_profit: {signal.take_profit:.6f}
  proposed size_usd: {signal.size_usd:.2f}
  analyst_reasoning: {signal.reasoning[:400]}

Hard gate constraints already verified:
  ✓ signal in {{long, short}}
  ✓ confidence ≥ 0.60
  ✓ risk% ≤ 2%
  ✓ size_usd ≤ $500

Your task: Calibrate size_usd proportionally to confidence and approve/disapprove.
Max allowed size_usd: ${self.rm_config.max_size_usd:.0f}
Respond ONLY with JSON: {{"approved": bool, "size_usd": float, "negotiation_summary": string}}"""
