"""
evolvemem/memory/consolidator.py

Memory store consolidation: deduplication, importance decay, entity reinforcement.
Implements Section 3.1 "Consolidation" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Three lightweight passes (Section 3.1):
  1. Deduplication: Jaccard similarity over tokenized content, threshold tau_J=0.80
  2. Importance decay: linear schedule, rate alpha_d=0.05/day, floor iota_min=0.15
  3. Entity reinforcement: increment rho_i by delta_rho on entity co-occurrence with query
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Set

from .store import MemoryStore, MemoryUnit

logger = logging.getLogger(__name__)


class Consolidator:
    """
    Applies three consolidation passes to maintain memory store quality.

    Paper reference: Section 3.1 "Consolidation", Appendix A "Consolidation parameters"
    """

    def __init__(
        self,
        tau_j: float = 0.80,       # Jaccard dedup threshold (Appendix A)
        alpha_d: float = 0.05,     # Importance decay rate per day (Appendix A)
        iota_min: float = 0.15,    # Importance floor (Appendix A)
        delta_rho: float = 0.05,   # Entity reinforcement increment (Appendix A)
        rho_max: float = 0.30,     # Entity reinforcement cap (Appendix A)
        decay_period_days: float = 30.0,  # D=30d decay period (Appendix A)
    ):
        self.tau_j = tau_j
        self.alpha_d = alpha_d
        self.iota_min = iota_min
        self.delta_rho = delta_rho
        self.rho_max = rho_max
        self.decay_period_days = decay_period_days

    # ------------------------------------------------------------------
    # Pass 1: Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        """
        Merge near-duplicate memory units using Jaccard similarity.

        "First, deduplication merges any pair (m_i, m_j) whose Jaccard similarity
        over tokenized content exceeds threshold tau_J, retaining the higher-importance unit."
        — Section 3.1

        Args:
            units: List of memory units to deduplicate.

        Returns:
            Deduplicated list retaining the higher-importance unit from each duplicate pair.
        """
        if len(units) <= 1:
            return units

        tokenized = [self._tokenize(u.content) for u in units]
        keep_mask = [True] * len(units)

        for i in range(len(units)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(units)):
                if not keep_mask[j]:
                    continue
                sim = self._jaccard(tokenized[i], tokenized[j])
                if sim >= self.tau_j:
                    # Retain higher-importance unit (Section 3.1)
                    if units[i].importance >= units[j].importance:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break  # unit i removed; move to next i

        kept = [u for u, keep in zip(units, keep_mask) if keep]
        removed = len(units) - len(kept)
        if removed > 0:
            logger.info(f"Deduplication: removed {removed} near-duplicate units (tau_J={self.tau_j})")
        return kept

    # ------------------------------------------------------------------
    # Pass 2: Importance decay
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        units: List[MemoryUnit],
        current_time: Optional[datetime] = None,
    ) -> List[MemoryUnit]:
        """
        Apply linear importance decay to memory units.

        "Second, importance decay applies a linear schedule that reduces iota_i by
        a fixed rate alpha_d per time unit, with a floor iota_min to prevent useful
        memories from vanishing entirely." — Section 3.1 / Appendix A

        Args:
            units: Memory units to update.
            current_time: Reference time for decay calculation (default: now).

        Returns:
            Units with updated importance scores.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        for unit in units:
            if unit.created_at:
                try:
                    created = datetime.fromisoformat(unit.created_at.replace("Z", "+00:00").replace("+00:00", ""))
                    age_days = (current_time - created).total_seconds() / 86400.0
                    periods = age_days / self.decay_period_days
                    # Linear decay: iota_i -= alpha_d * periods, floor at iota_min
                    new_importance = max(self.iota_min, unit.importance - self.alpha_d * periods)
                    unit.importance = new_importance
                except (ValueError, TypeError):
                    pass  # Malformed timestamp — skip

        return units

    # ------------------------------------------------------------------
    # Pass 3: Entity reinforcement
    # ------------------------------------------------------------------

    def reinforce_entities(
        self,
        units: List[MemoryUnit],
        query: str,
    ) -> List[MemoryUnit]:
        """
        Increment entity-reinforcement score for units whose entities co-occur with query.

        "Third, entity reinforcement increments rho_i by delta_rho each time a memory's
        extracted entities co-occur with a new query, capped at rho_max." — Section 3.1 / Appendix A

        Both iota_i and rho_i are carried forward in metadata and enter retrieval ranking
        via Equation 1.

        Args:
            units: Memory units to update.
            query: Incoming query string.

        Returns:
            Units with updated reinforcement scores.
        """
        query_lower = query.lower()
        for unit in units:
            all_entities = set(
                (unit.entities or []) +
                (unit.persons or []) +
                (unit.locations or [])
            )
            # Check if any entity co-occurs with the query
            if any(e.lower() in query_lower for e in all_entities if e):
                unit.reinforcement_score = min(
                    self.rho_max,
                    unit.reinforcement_score + self.delta_rho,
                )
        return units

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_all(
        self,
        store: MemoryStore,
        query: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> None:
        """
        Run all three consolidation passes on the store.

        Implements Algorithm 2 "Consolidation Phase" from the paper.

        Args:
            store: MemoryStore to consolidate in-place.
            query: Current query for entity reinforcement (optional).
            current_time: Reference time for decay (default: now).
        """
        units = store.get_all()
        if not units:
            return

        # Pass 1: deduplication
        units = self.deduplicate(units)

        # Pass 2: importance decay
        units = self.apply_decay(units, current_time)

        # Pass 3: entity reinforcement (only if query provided)
        if query:
            units = self.reinforce_entities(units, query)

        # Persist updated scores
        for unit in units:
            store.update_unit(
                unit.memory_id,
                importance=unit.importance,
                reinforcement_score=unit.reinforcement_score,
            )

        logger.info(f"Consolidation complete: {len(units)} active units after dedup/decay/reinforce")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple whitespace + lowercase tokenizer for Jaccard similarity."""
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(tokens)

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
