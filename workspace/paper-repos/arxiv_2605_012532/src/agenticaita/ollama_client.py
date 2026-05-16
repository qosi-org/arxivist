"""
ollama_client.py — Async Ollama REST API wrapper.
Handles JSON completion requests and telemetry logging.
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from typing import Optional

import aiohttp

from .config import OllamaConfig
from .schemas import OllamaCallRecord

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin async wrapper around the Ollama /api/chat endpoint.
    Used by AnalystAgent and RiskManagerAgent for LLM inference.
    """

    def __init__(self, cfg: OllamaConfig, db=None) -> None:
        self.cfg = cfg
        self.db = db  # EpisodicMemory for telemetry
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        agent: str = "unknown",
    ) -> str:
        """
        Send a completion request to Ollama.
        Returns the model's response text.
        Logs telemetry to ollama_calls table.
        """
        session = await self._get_session()
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
            "stream": False,
        }

        t0 = time.monotonic()
        try:
            async with session.post(
                f"{self.cfg.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response_text = data["message"]["content"]
                latency_ms = (time.monotonic() - t0) * 1000
                success = True
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            response_text = ""
            success = False
            logger.error(f"Ollama call failed ({agent}): {e}")

        # Telemetry
        if self.db:
            await self.db.log_ollama_call(OllamaCallRecord(
                timestamp=datetime.utcnow(),
                agent=agent,
                model=self.cfg.model,
                system_prompt_len=len(system_prompt),
                user_prompt_len=len(user_prompt),
                response_len=len(response_text),
                latency_ms=latency_ms,
                success=success,
            ))

        if not success:
            raise RuntimeError(f"Ollama inference failed for agent={agent}")

        return response_text

    async def health_check(self) -> bool:
        """Verify Ollama endpoint is reachable."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.cfg.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.cfg.model}, url={self.cfg.base_url})"
