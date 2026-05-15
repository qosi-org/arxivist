"""
evolvemem/llm/client.py

Unified LLM client wrapper supporting OpenAI and Anthropic APIs.
Used for extraction, answer generation, diagnosis, and query decomposition.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around OpenAI / Anthropic chat completion APIs.

    Paper uses GPT-4o for extraction and answer generation, GPT-5.1
    for backbone experiments (Section 4.1, Appendix E).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        """
        Args:
            provider: "openai" or "anthropic".
            model: Model identifier string.
            api_key: API key (falls back to environment variable).
            max_tokens: Maximum response tokens.
        """
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")

        logger.info(f"LLMClient initialized: provider={provider}, model={model}")

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."} dicts.
            system: System prompt string (optional).
            max_tokens: Override max_tokens for this call.

        Returns:
            Response content string.

        Raises:
            Exception: Re-raises API errors after logging.
        """
        tokens = max_tokens or self.max_tokens

        if self.provider == "openai":
            return self._openai_complete(messages, system, tokens)
        else:
            return self._anthropic_complete(messages, system, tokens)

    def _openai_complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """Call OpenAI chat completions API."""
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _anthropic_complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """Call Anthropic messages API."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model})"
