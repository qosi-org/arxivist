"""
agents/analyst.py — Analyst LLM Agent.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
The Analyst receives full market context and produces a typed JSON trading signal.

System prompt (verbatim from paper Section 4.2):
  'You are the AgenticAITA Analyst. Analyze the market and produce a trading signal.
   Respond ONLY in JSON: {signal: long|short|wait, confidence: float[0,1],
   entry_price, stop_loss, take_profit, size_usd, reasoning: string}.
   Your reasoning MUST cite the composite score, volatility regime, and orderbook
   context explicitly.'

The reasoning field is stored verbatim in the trades table and retrieved as
narrative episodic memory in future invocations on the same asset.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from agenticaita.agents.base import LLMAgent
from agenticaita.pipeline.contracts import AnalystContract, MarketContext
from agenticaita.utils.config import LLMConfig

logger = logging.getLogger(__name__)

# Verbatim from paper Section 4.2 (italicized block quote)
ANALYST_SYSTEM_PROMPT = """You are the AgenticAITA Analyst. Analyze the market and produce a trading signal.
Respond ONLY in JSON: {"signal": "long|short|wait", "confidence": <float 0-1>, "entry_price": <float>, "stop_loss": <float>, "take_profit": <float>, "size_usd": <float>, "reasoning": "<string>"}.
Your reasoning MUST cite the composite score, volatility regime, and orderbook context explicitly.
Do not include any text outside the JSON object."""


class AnalystAgent(LLMAgent):
    """
    Analyst agent — first stage of the SDP.

    Paper: Section 4.2 — receives 20-bar 1-min OHLCV, live L2 orderbook,
    funding rate, market snapshot, CBD score Ω_t, and episodic memory briefing.

    Self-abstention: if the Analyst returns signal='wait', the pipeline logs
    a Nwait event and terminates without invoking the Risk Manager.
    (8.3% self-abstention rate observed in 5-day session)
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, ANALYST_SYSTEM_PROMPT, "Analyst")

    async def call(
        self,
        ctx: MarketContext,
        memory: Optional[List[dict]] = None,
    ) -> AnalystContract:
        """
        Call the Analyst agent with full market context.

        Args:
            ctx: MarketContext with OHLCV, L2, funding, snapshot, omega_cbd.
            memory: Past trade records for this asset (episodic memory briefing).

        Returns:
            AnalystContract (validated Pydantic model).
            Raises ValueError if LLM output fails schema validation.
        """
        user_message = self._build_prompt(ctx, memory or [])
        raw, latency_ms = await self._call_ollama(user_message)

        parsed = self._parse_json_response(raw)
        try:
            contract = AnalystContract(**parsed)
        except Exception as e:
            raise ValueError(f"[Analyst] contract validation failed: {e}\nParsed: {parsed}")

        logger.info(
            f"[Analyst] {ctx.asset}: signal={contract.signal}, "
            f"conf={contract.confidence:.2f}, z̃latency={latency_ms:.0f}ms"
        )
        return contract

    def _build_prompt(self, ctx: MarketContext, memory: List[dict]) -> str:
        """
        Build the user prompt injected into the Analyst's context.

        Includes: OHLCV (last 20 bars), L2 orderbook summary, funding rate,
        CBD composite score Ω, and episodic memory briefing.
        """
        ohlcv_summary = self._summarize_ohlcv(ctx.ohlcv_1m)
        l2_summary = self._summarize_l2(ctx.l2_orderbook)
        memory_str = self._format_memory(memory)

        prompt = f"""Asset: {ctx.asset}

=== Market Context ===
OHLCV (last 20 bars, 1-min):
{ohlcv_summary}

L2 Orderbook:
{l2_summary}

Funding Rate: {ctx.funding_rate if ctx.funding_rate is not None else 'N/A'}

=== Composite Score ===
CBD Ω (composite anomaly + decorrelation score): {ctx.omega_cbd:.4f}
  — Higher Ω = more statistically anomalous AND more decorrelated from BTC.
  — Explicitly cite this score in your reasoning field.

=== Episodic Memory (past trades on {ctx.asset}) ===
{memory_str if memory_str else 'No prior trades on this asset in current session.'}

=== Task ===
Based on the above, produce a trading signal. You MAY return signal='wait' if no clear edge.
Respond ONLY with a JSON object matching the specified schema."""

        return prompt

    def _summarize_ohlcv(self, ohlcv: list) -> str:
        if not ohlcv:
            return "No OHLCV data available."
        lines = []
        for i, bar in enumerate(ohlcv[-5:]):  # Show last 5 bars in prompt
            if isinstance(bar, (list, tuple)) and len(bar) >= 5:
                lines.append(f"  [{i}] O={bar[0]:.4f} H={bar[1]:.4f} L={bar[2]:.4f} C={bar[3]:.4f} V={bar[4]:.2f}")
            else:
                lines.append(f"  [{i}] {bar}")
        return f"(showing last {len(lines)} of {len(ohlcv)} bars)\n" + "\n".join(lines)

    def _summarize_l2(self, l2: dict) -> str:
        if not l2:
            return "No orderbook data."
        bids = l2.get("bids", [])[:3]
        asks = l2.get("asks", [])[:3]
        bid_str = ", ".join(f"{b[0]:.4f}@{b[1]:.2f}" for b in bids) if bids else "none"
        ask_str = ", ".join(f"{a[0]:.4f}@{a[1]:.2f}" for a in asks) if asks else "none"
        return f"  Top bids: {bid_str}\n  Top asks: {ask_str}"

    def _format_memory(self, memory: List[dict]) -> str:
        if not memory:
            return ""
        lines = []
        for m in memory[:3]:  # Cap at 3 most recent
            pnl = f"PnL={m['pnl_usd']:.2f}" if m.get("pnl_usd") is not None else "PnL=pending"
            lines.append(f"  [{m['timestamp'][:16]}] {m['signal']} conf={m['confidence']:.2f} {pnl}")
            lines.append(f"    Reasoning: {m['reasoning'][:200]}...")
        return "\n".join(lines)
