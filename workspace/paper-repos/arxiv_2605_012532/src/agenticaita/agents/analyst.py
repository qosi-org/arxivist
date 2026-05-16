"""
agents/analyst.py — Analyst Agent (Section 4.2)
First stage of the Sequential Deliberative Pipeline (SDP).

Receives full market context and produces a trading signal with structured
JSON output. May self-abstain (signal='wait') — counted as agentic friction.
Reasoning is stored verbatim as narrative episodic memory.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

from ..schemas import AnalystOutput, MarketContext
from ..ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Section 4.2 — exact system prompt excerpt from paper, extended for completeness
ANALYST_SYSTEM_PROMPT = """You are the AgenticAITA Analyst. Analyze the market and produce a trading signal.
Respond ONLY in JSON: {{"signal": "long"|"short"|"wait", "confidence": float[0,1], "entry_price": float, "stop_loss": float, "take_profit": float, "size_usd": float, "reasoning": string}}.
Your reasoning MUST cite the composite score, volatility regime, and orderbook context explicitly.
If you are uncertain or conditions are unclear, respond with signal="wait" and confidence=0.
Do not add any text outside the JSON object."""


def _build_analyst_prompt(ctx: MarketContext, memory_briefing: str) -> str:
    """Construct the Analyst user prompt from market context."""
    recent_closes = [bar.close for bar in ctx.ohlcv[-10:]]
    close_str = ", ".join(f"{c:.4f}" for c in recent_closes)

    best_bid = ctx.l2_orderbook.best_bid
    best_ask = ctx.l2_orderbook.best_ask
    spread = best_ask - best_bid

    prompt = f"""Asset: {ctx.asset}
Current price: {ctx.current_price:.6f}
Z-score: {ctx.z_score:.3f}
Return magnitude: {ctx.return_magnitude:.4f}
CBD Composite Score (Omega): {ctx.omega:.4f}

Last 10 close prices (1m OHLCV): {close_str}

L2 Orderbook:
  Best bid: {best_bid:.6f} | Best ask: {best_ask:.6f} | Spread: {spread:.6f}
  Bid depth (top 5): {[(l.price, l.size) for l in ctx.l2_orderbook.bids[:5]]}
  Ask depth (top 5): {[(l.price, l.size) for l in ctx.l2_orderbook.asks[:5]]}

Funding rate: {ctx.funding_rate:.6f}

Episodic memory for {ctx.asset}:
{memory_briefing}

Produce your trading signal now."""
    return prompt


class AnalystAgent:
    """
    Analyst Agent — SDP Stage 1.
    Section 4.2 of arxiv:2605.12532.

    Epistemic mandate: analyze market, produce directional signal.
    May self-abstain via signal='wait' (counts toward agentic friction F, Eq. 8).
    """

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    async def analyze(
        self,
        ctx: MarketContext,
        memory_briefing: str,
    ) -> Optional[AnalystOutput]:
        """
        Run Analyst LLM inference.
        Returns AnalystOutput, or None if JSON parsing fails entirely.
        Returns AnalystOutput(signal='wait') for self-abstentions.
        """
        user_prompt = _build_analyst_prompt(ctx, memory_briefing)

        try:
            response = await self.llm.complete(
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent="analyst",
            )
        except RuntimeError as e:
            logger.error(f"Analyst LLM call failed: {e}")
            return None

        # Parse JSON response
        try:
            # Strip any surrounding whitespace / markdown fences
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            output = AnalystOutput(**data)
        except Exception as e:
            logger.error(f"Analyst JSON parse failed: {e}\nResponse: {response[:500]}")
            return None

        logger.info(f"Analyst: {ctx.asset} → {output.signal} (conf={output.confidence:.2f})")
        return output

    def __repr__(self) -> str:
        return f"AnalystAgent(llm={self.llm})"
