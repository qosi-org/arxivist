"""
LLM-Based Answer Matcher
=========================
Semantic equivalence checking for free-form forecasting answers.

Paper reference: Section 3 (Evaluation), Section 4.1, Appendix E.5
  "We use language model-based answer matching (Chandak et al., 2025)"
  "Agent predictions to each question are evaluated by DeepSeek v3.2 as the answer matcher"

Two prompts (from Appendix E.5):
  1. resolved_answer_equivalence: Does predicted outcome match ground truth?
  2. prediction_clustering_match: Does new prediction match any existing cluster?
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# --- Prompts from Appendix E.5 ---

RESOLVED_EQUIVALENCE_PROMPT = """You are an objective judge of forecasting predictions.

Question: "{question_title}"
Predicted outcome: "{predicted_outcome}"
Ground truth (actual answer): "{ground_truth}"

Does the predicted outcome match the ground truth? Rules:
- YES if predictions are semantically equivalent (same meaning, different wording)
- YES if predicted outcome is MORE SPECIFIC than ground truth (e.g. "David Raya" matches "Raya")
- NO if predicted outcome contains generic text like "Unknown" or "Answer 1" or "Option 1"
- NO if predicted outcome is VAGUER/MORE GENERAL than ground truth (e.g., "a goalkeeper" does NOT match "David Raya")
- NO if they refer to different things
Essentially, you have to grade whether the forecaster correctly predicted the ground truth answer for the question.
Answer strictly "Yes" or "No"."""

CLUSTERING_MATCH_PROMPT = """You are an objective judge of forecasting predictions.

Question: "{question_title}"
New prediction: "{candidate_prediction}"
Existing predictions:
{existing_list}

Does the new prediction match any of the existing predictions semantically?
- Match if they mean the same thing or if new prediction is more specific
- Do NOT match if new prediction is vaguer/more general
If yes, respond with ONLY the number (e.g., "1" or "3").
If no match exists, respond with "None".
Answer:"""


class AnswerMatcher:
    """
    LLM-based semantic answer equivalence for FutureSim scoring.

    Uses the prompts specified in Appendix E.5 with DeepSeek V3.2 (or any
    OpenAI-compatible endpoint).

    Paper reference: Section 3, Section 4.1, Appendix E.5
    """

    def __init__(
        self,
        model: str = "deepseek-chat",   # DeepSeek V3.2 via API
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            model: Model identifier for the answer-matching LLM
            base_url: API base URL (for DeepSeek or other OpenAI-compat endpoints)
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
            cache_path: Optional path to JSON file for caching match results
        """
        self.model = model
        self._cache: dict[str, bool] = {}
        self._cache_path = cache_path

        if cache_path:
            import json, pathlib
            p = pathlib.Path(cache_path)
            if p.exists():
                with open(p) as f:
                    self._cache = json.load(f)

        if not HAS_OPENAI:
            raise ImportError("openai is required: pip install openai")

        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=base_url or "https://api.deepseek.com",
        )

    def __repr__(self) -> str:
        return f"AnswerMatcher(model={self.model}, cache_size={len(self._cache)})"

    def match(
        self,
        question_title: str,
        predicted_outcome: str,
        ground_truth: str,
    ) -> bool:
        """
        Check if a predicted outcome matches the ground truth.

        Uses the resolved_answer_equivalence prompt from Appendix E.5.

        Args:
            question_title: The forecasting question text
            predicted_outcome: One outcome from the agent's prediction dict
            ground_truth: The resolved answer

        Returns:
            True if the predicted outcome semantically matches ground truth.
        """
        cache_key = f"{question_title}|||{predicted_outcome}|||{ground_truth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = RESOLVED_EQUIVALENCE_PROMPT.format(
            question_title=question_title,
            predicted_outcome=predicted_outcome,
            ground_truth=ground_truth,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # ASSUMED: deterministic evaluation; not specified in paper
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        result = answer.startswith("yes")

        self._cache[cache_key] = result
        self._save_cache()
        return result

    def cluster_prediction(
        self,
        question_title: str,
        candidate: str,
        existing_predictions: list[str],
    ) -> Optional[int]:
        """
        Check if a new prediction matches any existing cluster (0-indexed).

        Uses the prediction_clustering_match prompt from Appendix E.5.

        Args:
            question_title: The forecasting question text
            candidate: New prediction string to cluster
            existing_predictions: List of already-seen prediction strings

        Returns:
            Index (0-based) of matching existing prediction, or None if no match.
        """
        if not existing_predictions:
            return None

        existing_list = "\n".join(
            f"{i+1}. {pred}" for i, pred in enumerate(existing_predictions)
        )
        prompt = CLUSTERING_MATCH_PROMPT.format(
            question_title=question_title,
            candidate_prediction=candidate,
            existing_list=existing_list,
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        if raw.lower() == "none":
            return None
        try:
            return int(raw) - 1  # convert 1-indexed to 0-indexed
        except ValueError:
            return None

    def _save_cache(self) -> None:
        """Persist the match cache to disk."""
        if self._cache_path:
            import json
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)
