"""
agents/base.py — Abstract base class for LLM agents.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.2
All three agents (Analyst, Risk Manager, Executor) share a common Ollama interface.
Role-constrained prompting and JSON output enforcement are the primary safety mechanisms.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import aiohttp

from agenticaita.utils.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMAgent(ABC):
    """
    Abstract base class for all LLM agents in the SDP.

    Paper: Section 4.2 — agents communicate through typed JSON contracts
    governed by role-constrained system prompts.

    Args:
        config: LLMConfig with Ollama endpoint and model settings.
        system_prompt: Role-defining system prompt for this agent.
        agent_name: Human-readable name for logging.
    """

    def __init__(self, config: LLMConfig, system_prompt: str, agent_name: str) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self.agent_name = agent_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"

    async def _call_ollama(self, user_message: str) -> str:
        """
        Call the Ollama REST API with role-constrained prompting.

        Returns the raw text response from the LLM.
        Raises RuntimeError on API failure.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"num_predict": self.config.max_tokens},
        }

        url = f"{self.config.ollama_base_url}/api/chat"
        t0 = time.monotonic()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_s),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Ollama API error {resp.status}: {body[:200]}")
                data = await resp.json()

        latency_ms = (time.monotonic() - t0) * 1000
        content = data.get("message", {}).get("content", "")
        logger.debug(f"[{self.agent_name}] LLM latency={latency_ms:.0f}ms, len={len(content)}")
        return content, latency_ms

    def _parse_json_response(self, raw: str) -> dict:
        """
        Parse a JSON response from the LLM, stripping markdown fences if present.

        Raises ValueError if the response cannot be parsed as JSON.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Strip first and last fence lines
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            cleaned = inner.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"[{self.agent_name}] JSON parse failed: {e}\nRaw: {raw[:300]}")

    @abstractmethod
    async def call(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's core reasoning task."""
        ...
