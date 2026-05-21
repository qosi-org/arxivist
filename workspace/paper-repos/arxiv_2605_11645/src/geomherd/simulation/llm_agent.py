"""
src/geomherd/simulation/llm_agent.py
LLM-driven persona-conditioned agent and rule-based fallback.
Paper: arXiv:2605.11645, Section 2.1

Each financial trader is instantiated by a persona-conditioned LLM call.
LLM persona prompts are INTENTIONALLY WITHHELD by authors (see Risk R3).
This module provides:
  1. PersonaAgent: wraps Anthropic API call with a generic persona template
  2. RuleBasedAgentFallback: deterministic rule-based agents (no LLM required)

Set LLM_MODE=false in config (default) to use rule-based fallback.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


# Action constants matching CWSSubstrate
ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2
ACTION_MAP = {"buy": ACTION_BUY, "hold": ACTION_HOLD, "sell": ACTION_SELL}


class PersonaAgent:
    """
    LLM-driven persona-conditioned financial agent.

    Paper reference: Section 2.1
        Each financial trader instantiated by a separate persona-conditioned LLM call.
        Personas vary in: risk appetite, momentum horizon, herding tendency.

    # ASSUMED: Persona prompt template — actual prompts withheld by authors.
    # The template below is a best-effort approximation.
    # Set ANTHROPIC_API_KEY env variable to enable.

    Args:
        persona: Dict with keys {risk_appetite, momentum_horizon, herding_tendency, agent_id}
        model: Anthropic model ID
        temperature: LLM temperature
    """

    PERSONA_PROMPT_TEMPLATE = """You are a financial trader with the following profile:
- Risk appetite: {risk_appetite:.2f} (0=very conservative, 1=very aggressive)
- Momentum horizon: {momentum_horizon} steps (how far back you look for trends)
- Herding tendency: {herding_tendency:.2f} (0=contrarian, 1=strong follower)

Given the current market state, decide to BUY, HOLD, or SELL.
Market state:
- Current prices: {prices}
- Recent returns: {returns}
- Majority action in the market: {majority_action}

Respond with ONLY one word: BUY, HOLD, or SELL.
"""
    # ASSUMED: The above prompt structure; actual prompts withheld by paper authors.

    def __init__(
        self,
        persona: Optional[Dict] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
    ):
        self.persona = persona or {
            "risk_appetite": 0.5,
            "momentum_horizon": 10,
            "herding_tendency": 0.5,
            "agent_id": 0,
        }
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY not set. Use RuleBasedAgentFallback or set the key."
                    )
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic to use PersonaAgent")
        return self._client

    def decide(self, market_state: Dict) -> int:
        """
        Query LLM for trading decision given market state.

        Args:
            market_state: Dict with keys {prices, returns, majority_action}
        Returns:
            action: int in {ACTION_BUY, ACTION_HOLD, ACTION_SELL}
        """
        prompt = self.PERSONA_PROMPT_TEMPLATE.format(
            risk_appetite=self.persona["risk_appetite"],
            momentum_horizon=self.persona["momentum_horizon"],
            herding_tendency=self.persona["herding_tendency"],
            prices=market_state.get("prices", "unknown"),
            returns=market_state.get("returns", "unknown"),
            majority_action=market_state.get("majority_action", "hold"),
        )
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().lower()
        return ACTION_MAP.get(text, ACTION_HOLD)

    def set_persona(
        self,
        risk_appetite: float,
        momentum_horizon: int,
        herding_tendency: float,
    ) -> None:
        self.persona["risk_appetite"] = risk_appetite
        self.persona["momentum_horizon"] = momentum_horizon
        self.persona["herding_tendency"] = herding_tendency

    def __repr__(self) -> str:
        return (f"PersonaAgent(id={self.persona.get('agent_id')}, "
                f"risk={self.persona['risk_appetite']:.2f}, "
                f"herd={self.persona['herding_tendency']:.2f})")


class RuleBasedAgentFallback:
    """
    Deterministic rule-based agent fallback (no LLM required).
    Implements three trader archetypes controlled by persona parameters.

    Archetypes:
      - Noise trader (herding_tendency > 0.6): follows majority action
      - Fundamental trader (risk_appetite < 0.3): buys when price < fundamental, sells otherwise
      - Momentum trader (otherwise): follows recent return trend

    Paper reference: Section 2.1 — 'richer baseline than ABMs with hardcoded archetypes'
    Note: Rule-based agents are explicitly a weaker substitute; use for geometry validation.
    """

    def __init__(self, persona: Optional[Dict] = None, seed: int = 42):
        self.persona = persona or {
            "risk_appetite": 0.5,
            "momentum_horizon": 10,
            "herding_tendency": 0.5,
            "agent_id": 0,
        }
        self._rng = np.random.default_rng(seed)
        self._return_buffer: List[float] = []

    def decide(self, market_state: Dict) -> int:
        """Rule-based trading decision."""
        risk = self.persona["risk_appetite"]
        herd = self.persona["herding_tendency"]
        horizon = self.persona["momentum_horizon"]
        returns = market_state.get("returns", [0.0])
        majority = market_state.get("majority_action", ACTION_HOLD)
        prices = market_state.get("prices", [100.0])
        fundamental = market_state.get("fundamental", 100.0)

        # Track recent returns
        r = float(np.mean(returns)) if hasattr(returns, "__len__") else float(returns)
        self._return_buffer.append(r)
        if len(self._return_buffer) > horizon:
            self._return_buffer.pop(0)

        if herd > 0.6:
            # Noise / herding trader: follow majority with some noise
            if self._rng.random() < herd:
                return majority
            return int(self._rng.integers(0, 3))
        elif risk < 0.3:
            # Fundamental trader
            price = float(np.mean(prices))
            if price < fundamental * 0.98:
                return ACTION_BUY
            elif price > fundamental * 1.02:
                return ACTION_SELL
            return ACTION_HOLD
        else:
            # Momentum trader
            if not self._return_buffer:
                return ACTION_HOLD
            trend = float(np.mean(self._return_buffer))
            if trend > 0.005:
                return ACTION_BUY
            elif trend < -0.005:
                return ACTION_SELL
            return ACTION_HOLD

    def set_persona(
        self,
        risk_appetite: float,
        momentum_horizon: int,
        herding_tendency: float,
    ) -> None:
        self.persona["risk_appetite"] = risk_appetite
        self.persona["momentum_horizon"] = momentum_horizon
        self.persona["herding_tendency"] = herding_tendency

    def __repr__(self) -> str:
        return (f"RuleBasedAgentFallback(id={self.persona.get('agent_id')}, "
                f"risk={self.persona['risk_appetite']:.2f}, "
                f"herd={self.persona['herding_tendency']:.2f})")


def build_agent_population(
    N: int,
    llm_mode: bool = False,
    model: str = "claude-sonnet-4-20250514",
    seed: int = 42,
) -> List:
    """
    Build a heterogeneous population of N agents with random personas.

    Paper: Section 2.1 — distinct system-prompted personas varying in
    risk appetite, momentum horizon, and herding tendency.

    Args:
        N: Number of agents
        llm_mode: If True, use PersonaAgent (requires ANTHROPIC_API_KEY)
        model: LLM model ID
        seed: RNG seed for persona generation
    Returns:
        List of N agent instances
    """
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(N):
        persona = {
            "agent_id": i,
            "risk_appetite": float(rng.uniform(0.1, 0.9)),
            "momentum_horizon": int(rng.integers(5, 30)),
            "herding_tendency": float(rng.uniform(0.1, 0.9)),
        }
        if llm_mode:
            agent = PersonaAgent(persona=persona, model=model)
        else:
            agent = RuleBasedAgentFallback(persona=persona, seed=int(rng.integers(0, 10000)))
        agents.append(agent)
    return agents
