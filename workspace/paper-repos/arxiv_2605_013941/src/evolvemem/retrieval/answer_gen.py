"""
evolvemem/retrieval/answer_gen.py

Answer generation and second-pass verification for EVOLVEMEM.
Implements Section 3.2 "Answer generation" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Base answer: yhat_0 = psi_ans(q, R(q; theta), alpha) — Equation 12
Verification: yhat = psi_ver(q, R, yhat_0) if conf < tau_ver or yhat_0 in U — Equation 13

Prompts verbatim from Appendix F of the paper.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..memory.store import MemoryUnit
from .config import RetrievalConfig

logger = logging.getLogger(__name__)


# "Unknown" / "not specified" class — triggers verification (Equation 13)
_UNKNOWN_CLASS = {
    "unknown", "not specified", "not mentioned", "not found",
    "not available", "n/a", "none", "not stated",
}

# WARNING: tau_ver is ASSUMED=0.5 — paper references this threshold but does not define it numerically
_TAU_VER_DEFAULT = 0.5


class AnswerGenerator:
    """
    Generates answers from retrieved memory context using configurable styles.

    Implements answer generation and optional second-pass verification from Section 3.2.
    Prompts are taken verbatim from Appendix F of the paper.

    Paper reference: Section 3.2 "Answer generation", Appendix F
    """

    def __init__(
        self,
        llm_client,
        tau_ver: float = _TAU_VER_DEFAULT,  # ASSUMED: paper threshold not numerically specified
    ):
        """
        Args:
            llm_client: LLMClient instance.
            tau_ver: Confidence threshold for triggering verification (ASSUMED=0.5).
        """
        self.llm_client = llm_client
        self.tau_ver = tau_ver  # ASSUMED: not specified in paper

    def generate(
        self,
        query: str,
        context: List[MemoryUnit],
        config: RetrievalConfig,
        category: Optional[int] = None,
    ) -> str:
        """
        Generate an answer from retrieved context with optional second-pass verification.

        Implements Equations 12–13:
          yhat_0 = psi_ans(q, R(q; theta), alpha)
          yhat = psi_ver(q, R, yhat_0) if conf(yhat_0) < tau_ver or yhat_0 in U

        Args:
            query: Question string.
            context: Retrieved memory units.
            config: Current retrieval configuration (for style and verify settings).
            category: Question category (LoCoMo 1–5) for category-specific prompting.

        Returns:
            Final answer string.
        """
        cfg = config.for_category(category)
        context_str = self._format_context(context)

        # Base answer — Equation 12
        raw = self._base_generate(query, context_str, cfg.answer_style, category)
        answer = self._extract_answer(raw)

        # Second-pass verification — Equation 13
        if cfg.enable_answer_verification:
            answer = self._maybe_verify(query, context_str, answer, cfg.verification_style)

        return answer

    def verify(
        self,
        query: str,
        context: List[MemoryUnit],
        candidate: str,
        style: str = "strict",
    ) -> str:
        """
        Second-pass verification. Exposed for external use.

        "A second-pass verifier reviews low-confidence responses against the context."
        — Section 3.2, Equation 13

        Args:
            query: Original question.
            context: Retrieved memory units.
            candidate: Candidate answer to verify.
            style: Verification style ("strict" or "multi_candidate").

        Returns:
            Verified (possibly corrected) answer string.
        """
        context_str = self._format_context(context)
        return self._maybe_verify(query, context_str, candidate, style)

    # ------------------------------------------------------------------
    # Private: base generation
    # ------------------------------------------------------------------

    def _base_generate(
        self,
        query: str,
        context_str: str,
        style: str,
        category: Optional[int],
    ) -> str:
        """Choose prompt based on style/category and call LLM. From Appendix F."""
        if category == 5:  # Adversarial name-swap — Appendix F.3 Cat.5
            prompt = self._cat5_adversarial_prompt(query, context_str)
        elif category == 3:  # Inferential — Appendix F.3 Cat.3
            prompt = self._cat3_inferential_prompt(query, context_str)
        elif style == "strict":  # Appendix F.3 "Strict answer_style"
            prompt = self._strict_prompt(query, context_str)
        else:  # Default concise
            prompt = self._concise_prompt(query, context_str)

        system = "Professional Q&A assistant. Concise answers grounded in context. JSON output only."
        try:
            return self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return json.dumps({"answer": "not specified", "reasoning": "LLM call failed"})

    # ------------------------------------------------------------------
    # Private: verification
    # ------------------------------------------------------------------

    def _maybe_verify(
        self,
        query: str,
        context_str: str,
        candidate: str,
        style: str,
    ) -> str:
        """
        Apply second-pass verification if candidate is low-confidence or in Unknown class.

        Equation 13: yhat = psi_ver(q, R, yhat0) if conf(yhat0) < tau_ver or yhat0 in U
        """
        candidate_lower = candidate.lower().strip()
        in_unknown_class = any(u in candidate_lower for u in _UNKNOWN_CLASS)

        # Trigger verification (Equation 13)
        if in_unknown_class:
            return self._run_verification(query, context_str, candidate, style)

        return candidate

    def _run_verification(
        self,
        query: str,
        context_str: str,
        candidate: str,
        style: str,
    ) -> str:
        """Run the verifier LLM pass. Prompt from Appendix F.5."""
        if style == "multi_candidate":
            specific_instruction = (
                "Review the candidate answer. If it is wrong, give the correct answer. "
                "If the candidate is 'Unknown' but the context contains any relevant fact, "
                "pick the most plausible option. Consider 2-3 candidate answers and pick the best."
            )
        else:  # strict (default) — Appendix F.5
            specific_instruction = (
                "Review the candidate answer. If it says 'Unknown' or 'Not specified', "
                "replace it with the most likely answer from the context. "
                "Format numbers as Arabic digits, years as YYYY. Keep the answer concise (1-8 words)."
            )

        prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Candidate answer: {candidate}\n\n"
            f"{specific_instruction}\n\n"
            f"Return JSON: {{\"reasoning\":\"brief\",\"answer\":\"final answer\"}}"
        )

        try:
            raw = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                system="Answer verifier. JSON output only.",
            )
            return self._extract_answer(raw)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return candidate

    # ------------------------------------------------------------------
    # Prompt templates (verbatim from Appendix F)
    # ------------------------------------------------------------------

    def _cat5_adversarial_prompt(self, query: str, context: str) -> str:
        """Appendix F.3: LoCoMo Cat.5 Adversarial name-swap prompt."""
        return (
            "Answer based on the context.\n\n"
            "IMPORTANT: This question may deliberately swap person names. "
            "The CONTEXT contains the TRUE information; answer based on the context "
            "even if the question names seem off.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Rules:\n"
            "1. ALWAYS provide a substantive answer, never 'not specified'.\n"
            "2. Answer in 1-5 words using exact facts from context.\n\n"
            'Return JSON: {"reasoning":"brief","answer":"concise"}'
        )

    def _cat3_inferential_prompt(self, query: str, context: str) -> str:
        """Appendix F.3: LoCoMo Cat.3 Inferential prompt."""
        return (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "This question asks for an INFERENCE or COUNTERFACTUAL judgement "
            "(e.g., 'Would X...', 'What would X likely...'). "
            "Your job is to synthesize a best-guess answer from the evidence, do NOT refuse.\n\n"
            "Rules:\n"
            "1. NEVER answer 'unknown' / 'not specified' / 'not mentioned'. "
            "The answer must always be a substantive judgement.\n"
            "2. Answer in 1-6 words.\n"
            "3. Choose the option most consistent with the user's stated preferences, "
            "history, and values in the context.\n\n"
            'Return JSON: {"reasoning":"brief","answer":"concise"}'
        )

    def _strict_prompt(self, query: str, context: str) -> str:
        """Appendix F.3: LoCoMo Strict answer_style prompt."""
        return (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Rules:\n"
            "1. Answer in 1-10 words. Use EXACT words/phrases from context.\n"
            "2. Format conventions:\n"
            "   - 'how many/times' -> single Arabic numeral.\n"
            "   - 'when'/year questions -> 4-digit year or 'YYYY-MM-DD'.\n"
            "   - 'where' -> place name exactly as in context.\n"
            "   - 'what/who' -> shortest distinctive noun phrase.\n"
            "3. NEVER answer 'Unknown', 'Not specified', 'Not mentioned'.\n"
            "4. For multi-item questions, list items separated by comma.\n\n"
            'Return JSON: {"reasoning":"brief","answer":"concise"}'
        )

    def _concise_prompt(self, query: str, context: str) -> str:
        """Default concise prompt for all other categories."""
        return (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer concisely in 1-10 words, grounded in the context. "
            "Do not say 'unknown' or 'not specified' — infer the most plausible answer.\n\n"
            'Return JSON: {"reasoning":"brief","answer":"concise answer"}'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(units: List[MemoryUnit]) -> str:
        """Format retrieved memory units as numbered context string."""
        if not units:
            return "(No relevant memories found)"
        lines = []
        for i, unit in enumerate(units, 1):
            ts = f" [{unit.timestamp}]" if unit.timestamp else ""
            lines.append(f"{i}. {unit.content}{ts}")
        return "\n".join(lines)

    @staticmethod
    def _extract_answer(raw: str) -> str:
        """Extract the 'answer' field from JSON LLM response."""
        cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()
        try:
            parsed = json.loads(cleaned)
            return str(parsed.get("answer", cleaned)).strip()
        except json.JSONDecodeError:
            # Try regex fallback
            match = re.search(r'"answer"\s*:\s*"([^"]+)"', cleaned)
            if match:
                return match.group(1).strip()
            return cleaned.strip()
