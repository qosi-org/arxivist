"""
evolvemem/evolution/diagnosis.py

LLM-powered failure diagnosis module.
Implements Section 3.3 "Failure diagnosis" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

The diagnosis module reads per-question failure logs, categorizes root causes,
and proposes targeted configuration adjustments (Delta_theta_r).

Prompt is taken verbatim from Appendix F.6 of the paper.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

from ..retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


class DiagnosisModule:
    """
    LLM-powered diagnosis engine that reads failure logs and proposes config changes.

    "After each evaluation round r, the system writes a per-question raw log...
    The diagnosis module invokes an LLM with a structured rubric covering common
    failure patterns." — Section 3.3

    The rubric is written in terms of failure patterns rather than specific benchmarks,
    so newly discovered configuration dimensions become immediately usable (Section 3.3).

    Paper reference: Section 3.3 "Failure diagnosis", Appendix F.6
    """

    def __init__(self, llm_client, benchmark: str = "locomo"):
        """
        Args:
            llm_client: LLMClient instance for diagnosis calls.
            benchmark: Benchmark name for context in diagnosis prompt ("locomo" or "membench").
        """
        self.llm_client = llm_client
        self.benchmark = benchmark

    def diagnose(
        self,
        raw_log: List[Dict[str, Any]],
        current_config: RetrievalConfig,
        memory_size: int,
    ) -> Dict[str, Any]:
        """
        Analyze per-question failure log and propose configuration adjustments.

        "The diagnosis module invokes an LLM with a structured rubric covering
        common failure patterns (e.g., wrong entity retrieved, insufficient context,
        temporal confusion)." — Section 3.3

        Args:
            raw_log: Per-question results: [{q, pred, ref, score, category, sources}, ...]
            current_config: Current RetrievalConfig theta_r.
            memory_size: Number of active memories in store.

        Returns:
            Dict with keys: root_causes, parameter_suggestions, per_category_proposals,
            missing_topics, priority_actions.
            parameter_suggestions is a flat dict of {field: new_value} to apply to config.
        """
        total_questions = len(raw_log)
        overall_f1 = sum(r.get("score", 0.0) for r in raw_log) / max(total_questions, 1)
        zero_count = sum(1 for r in raw_log if r.get("score", 0.0) == 0.0)

        failure_summary = self._build_failure_summary(raw_log)
        category_breakdown = self._build_category_breakdown(raw_log)
        sample_failures = self._sample_worst_failures(raw_log, n=10)
        todo_checklist = self._build_todo_checklist(current_config, raw_log)

        prompt = self._build_diagnosis_prompt(
            benchmark=self.benchmark,
            total_memories=memory_size,
            total_questions=total_questions,
            overall_f1=overall_f1,
            zero_count=zero_count,
            current_config=json.dumps(current_config.to_dict(), indent=2),
            failure_summary=failure_summary,
            category_breakdown=category_breakdown,
            sample_failures=sample_failures,
            todo_checklist=todo_checklist,
        )

        try:
            raw = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                system="You are the diagnosis engine of a self-evolving memory system. JSON output only.",
            )
            proposal = self._parse_proposal(raw)
            logger.info(f"Diagnosis complete. Proposed {len(proposal.get('parameter_suggestions', {}))} parameter changes.")
            return proposal
        except Exception as e:
            logger.error(f"Diagnosis LLM call failed: {e}")
            return {"parameter_suggestions": {}, "root_causes": {}, "missing_topics": [], "priority_actions": []}

    # ------------------------------------------------------------------
    # Prompt building (Appendix F.6 verbatim structure)
    # ------------------------------------------------------------------

    def _build_diagnosis_prompt(
        self,
        benchmark: str,
        total_memories: int,
        total_questions: int,
        overall_f1: float,
        zero_count: int,
        current_config: str,
        failure_summary: str,
        category_breakdown: str,
        sample_failures: str,
        todo_checklist: str,
    ) -> str:
        """Build diagnosis prompt verbatim from Appendix F.6."""
        return f"""You are the diagnosis engine of a self-evolving memory system. Your job is to turn evaluation failures into a concrete next-round action that moves the system toward SOTA.

## System Info
- Benchmark: {benchmark}
- Total memories: {total_memories}
- Total questions: {total_questions}
- Overall score: {overall_f1:.4f}
- Zero-score count: {zero_count}/{total_questions}
- Current config (JSON): {current_config}

## Failure Analysis Summary
{failure_summary}

## Per-Category Breakdown
{category_breakdown}

## Sample Failures (worst cases)
{sample_failures}

## Tier-1 TODO checklist: levers that are still OFF in the incumbent
{todo_checklist}

(Dynamically generated: lists each disabled lever whose symptom is present in the current failure data. Prefer picking ONE item per round until empty.)

## Decision Rubric
1. If many 'abstention' failures -> raise top_k, widen max_context, consider rrf fusion.
2. If many 'wrong answer' failures with high retrieval -> lower max_context or raise weights for strongest view.
3. If temporal category weakness -> enable time_decay_half_life_days.
4. If adversarial category weakness -> enable_entity_swap=true.
5. If multi-hop weakness -> reflection_rounds >= 1.
6. If ONE category lags -> per_category_overrides (preserve gains elsewhere).
7. Prefer enabling something disabled BEFORE tuning a small int.
8. If residual 'Unknown' or format-mismatch -> enable_answer_verification=true.
9. LoCoMo prompt-surface flags are highest-ROI when their symptom matches; propose them early.

## Output
Return JSON with 'parameter_suggestions' as a flat dict of field -> new value.
Fields MUST match RetrievalConfig field names exactly. Only include fields you want to change.

{{
  "root_causes": {{"extraction_gap": {{}}, "retrieval_miss": {{}}, "answer_error": {{}}}},
  "missing_topics": ["topic1", "topic2"],
  "parameter_suggestions": {{"fusion_mode": "rrf", "semantic_top_k": 15}},
  "extraction_suggestions": {{"window_size": 30}},
  "per_category_proposals": {{"5": {{"enable_entity_swap": true}}}},
  "priority_actions": ["action1", "action2", "action3"]
}}

Return ONLY JSON."""

    # ------------------------------------------------------------------
    # Log analysis helpers
    # ------------------------------------------------------------------

    def _build_failure_summary(self, raw_log: List[Dict[str, Any]]) -> str:
        """Summarize failure patterns across the log."""
        failures = [r for r in raw_log if r.get("score", 1.0) < 0.5]
        if not failures:
            return "No major failures detected."

        abstentions = sum(
            1 for r in failures
            if any(u in str(r.get("pred", "")).lower() for u in ["not specified", "unknown", "not mentioned"])
        )
        wrong_entity = sum(
            1 for r in failures
            if r.get("score", 0.0) == 0.0 and r.get("pred", "").strip()
        )

        lines = [
            f"Total failures (score < 0.5): {len(failures)}/{len(raw_log)}",
            f"Abstentions ('not specified' / 'unknown'): {abstentions}",
            f"Wrong answers (non-empty, zero score): {wrong_entity}",
        ]
        return "\n".join(lines)

    def _build_category_breakdown(self, raw_log: List[Dict[str, Any]]) -> str:
        """Per-category score breakdown."""
        cat_scores: Dict[str, List[float]] = defaultdict(list)
        for r in raw_log:
            cat = str(r.get("category", "unknown"))
            cat_scores[cat].append(float(r.get("score", 0.0)))

        lines = []
        for cat in sorted(cat_scores.keys()):
            scores = cat_scores[cat]
            avg = sum(scores) / len(scores)
            zeros = sum(1 for s in scores if s == 0.0)
            lines.append(f"  Cat {cat}: avg={avg:.3f}, n={len(scores)}, zeros={zeros}")

        return "\n".join(lines) if lines else "No category data available."

    def _sample_worst_failures(
        self,
        raw_log: List[Dict[str, Any]],
        n: int = 10,
    ) -> str:
        """Return n worst failures as formatted string."""
        sorted_log = sorted(raw_log, key=lambda r: r.get("score", 1.0))
        worst = sorted_log[:n]

        lines = []
        for i, r in enumerate(worst, 1):
            q = str(r.get("q", r.get("question", "")))[:100]
            pred = str(r.get("pred", r.get("prediction", "")))[:80]
            ref = str(r.get("ref", r.get("reference", "")))[:80]
            score = r.get("score", 0.0)
            cat = r.get("category", "?")
            lines.append(f"{i}. [Cat{cat}] score={score:.2f} | Q: {q} | Pred: {pred} | Ref: {ref}")

        return "\n".join(lines) if lines else "No failures."

    def _build_todo_checklist(
        self,
        config: RetrievalConfig,
        raw_log: List[Dict[str, Any]],
    ) -> str:
        """
        Build a checklist of disabled levers that may help based on failure patterns.
        This dynamically generates the Tier-1 TODO list from the Appendix F.6 prompt.
        """
        todos = []

        if config.semantic_top_k == 0:
            todos.append("- [ ] Enable semantic_top_k (currently 0) — helps with paraphrased/abstract queries")
        if config.structured_top_k == 0:
            todos.append("- [ ] Enable structured_top_k (currently 0) — helps with entity-specific lookups")
        if not config.enable_entity_swap:
            todos.append("- [ ] Enable enable_entity_swap — helps with adversarial name-swap questions (Cat 5)")
        if not config.enable_query_decomposition:
            todos.append("- [ ] Enable enable_query_decomposition — helps with multi-hop questions (Cat 1/3)")
        if not config.enable_answer_verification:
            todos.append("- [ ] Enable enable_answer_verification — reduces 'Unknown'/'not specified' abstentions")
        if config.fusion_mode == "sum":
            todos.append("- [ ] Try fusion_mode=rrf — more robust to score scale differences across views")
        if config.time_decay_half_life_days is None:
            todos.append("- [ ] Enable time_decay_half_life_days — helps with temporal ordering questions")
        if not config.per_category_overrides:
            todos.append("- [ ] Add per_category_overrides — allows specialization per question type")

        return "\n".join(todos) if todos else "All major levers are enabled. Focus on tuning."

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_proposal(self, raw: str) -> Dict[str, Any]:
        """
        Parse the diagnosis LLM's JSON response into a structured proposal.

        Merges per_category_proposals into parameter_suggestions under
        the per_category_overrides key for direct application to RetrievalConfig.
        """
        cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()

        try:
            proposal = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to extract JSON object from partial response
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    proposal = json.loads(match.group())
                except Exception:
                    return {"parameter_suggestions": {}, "root_causes": {}, "missing_topics": [], "priority_actions": []}
            else:
                return {"parameter_suggestions": {}, "root_causes": {}, "missing_topics": [], "priority_actions": []}

        # Merge per_category_proposals into parameter_suggestions
        per_cat = proposal.pop("per_category_proposals", {})
        if per_cat:
            proposal.setdefault("parameter_suggestions", {})
            proposal["parameter_suggestions"]["per_category_overrides"] = per_cat

        return proposal
