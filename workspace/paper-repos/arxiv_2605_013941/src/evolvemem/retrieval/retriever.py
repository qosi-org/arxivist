"""
evolvemem/retrieval/retriever.py

Multi-view retriever with fusion, entity-swap, and query decomposition.
Implements Section 3.2 "Retrieval as an Evolvable Action Space" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

Retrieval views (Section 3.2):
  - Lexical: BM25 (Equation 2)
  - Semantic: cosine similarity (Equation 3)
  - Structured metadata: entity/person/location filter (Equation 4)

Fusion modes (Section 3.2):
  - SUM, WEIGHTED-SUM, RRF (Equation 5)

Final ranking (Equation 1):
  s(q, m_i; theta) = s_fuse + lambda_iota * iota_i + lambda_r * rec(m_i) + rho_i

Query augmentation:
  - Adversarial entity-swap (Equation 10)
  - Query decomposition via LLM (Equation 11)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from ..memory.store import MemoryStore, MemoryUnit
from .config import RetrievalConfig

logger = logging.getLogger(__name__)


class MultiViewRetriever:
    """
    Multi-view retriever that fuses lexical, semantic, and structured signals.

    Paper reference: Section 3.2, Equations 1–5, 10–11
    """

    def __init__(
        self,
        store: MemoryStore,
        config: RetrievalConfig,
        llm_client=None,                     # Required for query decomposition
        lambda_iota: float = 1.0,            # ASSUMED: weight for importance in Eq. 1
        lambda_r: float = 1.0,               # ASSUMED: weight for recency in Eq. 1
        rrf_k: int = 60,                     # ASSUMED: standard RRF default (paper omits value)
    ):
        """
        Args:
            store: MemoryStore instance.
            config: Current retrieval configuration theta.
            llm_client: LLM client for query decomposition (optional).
            lambda_iota: Weight for importance in final ranking (ASSUMED=1.0).
            lambda_r: Weight for recency in final ranking (ASSUMED=1.0).
            rrf_k: RRF smoothing constant k (ASSUMED=60; paper formula omits value).
        """
        self.store = store
        self.config = config
        self.llm_client = llm_client
        self.lambda_iota = lambda_iota  # ASSUMED: not specified in paper
        self.lambda_r = lambda_r        # ASSUMED: not specified in paper
        self.rrf_k = rrf_k              # ASSUMED: standard default

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        category: Optional[int] = None,
        query_vec=None,
    ) -> List[MemoryUnit]:
        """
        Main retrieval method — runs multi-view search and returns ranked units.

        Implements the full retrieval pipeline from Section 3.2 including:
          - Per-category config override
          - Three retrieval views
          - Fusion (SUM / WEIGHTED-SUM / RRF)
          - Final ranking with Equation 1
          - Optional entity-swap and query decomposition

        Args:
            query: Query string.
            config: Override config (defaults to self.config).
            category: Question category for per-category overrides.
            query_vec: Pre-computed query embedding (optional).

        Returns:
            List of MemoryUnit objects, ranked, capped at config.max_context.
        """
        cfg = (config or self.config).for_category(category)

        # Optional query decomposition (multi-hop) — Equation 11
        if cfg.enable_query_decomposition and self.llm_client:
            units = self._decompose_and_retrieve(query, cfg, query_vec)
        else:
            units = self._retrieve_single(query, cfg, query_vec)

        # Optional entity-swap augmentation — Equation 10
        if cfg.enable_entity_swap:
            swap_units = self._entity_swap_retrieve(query, cfg)
            # Union: add non-duplicate swap results
            existing_ids = {u.memory_id for u in units}
            units.extend(u for u in swap_units if u.memory_id not in existing_ids)

        # Apply final ranking (Equation 1) and cap at max_context
        ranked = self._final_rank(units, cfg)
        return ranked[:cfg.max_context]

    # ------------------------------------------------------------------
    # Single-query retrieval
    # ------------------------------------------------------------------

    def _retrieve_single(
        self,
        query: str,
        cfg: RetrievalConfig,
        query_vec=None,
    ) -> List[MemoryUnit]:
        """Run all three views for a single query and fuse results."""
        view_results: Dict[str, List[Tuple[MemoryUnit, float]]] = {}

        # Lexical view — BM25, Equation 2
        if cfg.keyword_top_k > 0:
            view_results["kw"] = self.store.search_bm25(query, cfg.keyword_top_k)

        # Semantic view — cosine, Equation 3
        if cfg.semantic_top_k > 0:
            view_results["sem"] = self.store.search_semantic(query, cfg.semantic_top_k, query_vec)

        # Structured metadata view — entity filter, Equation 4
        if cfg.structured_top_k > 0:
            view_results["str"] = self.store.search_structured(query, cfg.structured_top_k)

        if not view_results:
            return []

        return self._fuse(view_results, cfg)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse(
        self,
        view_results: Dict[str, List[Tuple[MemoryUnit, float]]],
        cfg: RetrievalConfig,
    ) -> List[MemoryUnit]:
        """
        Fuse multiple view results into a single ranked list.

        Three modes from Section 3.2:
          - SUM: add raw view scores
          - WEIGHTED-SUM: apply per-view weights on normalized scores
          - RRF: reciprocal rank fusion (Equation 5)
        """
        mode = cfg.fusion_mode.lower()

        if mode == "rrf":
            return self._rrf({v: [u for u, _ in pairs] for v, pairs in view_results.items()})

        # Collect all unique units with their view scores
        unit_map: Dict[str, MemoryUnit] = {}
        unit_scores: Dict[str, float] = defaultdict(float)

        for view_name, pairs in view_results.items():
            if not pairs:
                continue

            scores = [s for _, s in pairs]
            max_s = max(scores) if scores else 1.0
            min_s = min(scores) if scores else 0.0
            range_s = max_s - min_s if max_s != min_s else 1.0

            weight = 1.0
            if mode == "weighted_sum":
                weight = {"kw": cfg.w_kw, "sem": cfg.w_sem, "str": cfg.w_str}.get(view_name, 1.0)

            for unit, score in pairs:
                unit_map[unit.memory_id] = unit
                if mode == "weighted_sum":
                    normalized = (score - min_s) / range_s  # normalize to [0,1]
                    unit_scores[unit.memory_id] += weight * normalized
                else:  # SUM
                    unit_scores[unit.memory_id] += score

        ranked = sorted(unit_scores.items(), key=lambda x: x[1], reverse=True)
        return [unit_map[mid] for mid, _ in ranked]

    def _rrf(
        self,
        ranked_lists: Dict[str, List[MemoryUnit]],
    ) -> List[MemoryUnit]:
        """
        Reciprocal Rank Fusion.

        s_fuse(q, m_i; theta) = sum_v 1 / (k + r_v(m_i))
        — Equation 5, Section 3.2

        Args:
            ranked_lists: Dict of view_name -> list of MemoryUnits in ranked order.

        Returns:
            List of MemoryUnits sorted by RRF score descending.
        """
        scores: Dict[str, float] = defaultdict(float)
        unit_map: Dict[str, MemoryUnit] = {}

        for view_units in ranked_lists.values():
            for rank, unit in enumerate(view_units, start=1):
                # Equation 5: 1 / (k + r_v(m_i))
                scores[unit.memory_id] += 1.0 / (self.rrf_k + rank)
                unit_map[unit.memory_id] = unit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [unit_map[mid] for mid, _ in ranked]

    # ------------------------------------------------------------------
    # Final ranking (Equation 1)
    # ------------------------------------------------------------------

    def _final_rank(
        self,
        units: List[MemoryUnit],
        cfg: RetrievalConfig,
    ) -> List[MemoryUnit]:
        """
        Apply final ranking combining fused relevance with memory-intrinsic quality signals.

        s(q, m_i; theta) = s_fuse(q, m_i; theta) + lambda_iota * iota_i + lambda_r * rec(m_i) + rho_i
        — Equation 1, Section 3.2

        Note: s_fuse is approximated by position in the input list (rank-based).
        lambda_iota and lambda_r are ASSUMED=1.0 (not specified in paper).
        """
        now = datetime.utcnow()
        scored = []

        for rank, unit in enumerate(units, start=1):
            # Approximate s_fuse via rank (1.0 for rank 1, diminishing)
            s_fuse = 1.0 / (1.0 + math.log1p(rank))

            # Recency factor rec(m_i): exponential decay based on half-life
            rec = self._recency(unit, now, cfg.time_decay_half_life_days)

            # Equation 1
            total = (
                s_fuse
                + self.lambda_iota * unit.importance   # lambda_iota ASSUMED=1.0
                + self.lambda_r * rec                  # lambda_r ASSUMED=1.0
                + unit.reinforcement_score
            )
            scored.append((unit, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in scored]

    @staticmethod
    def _recency(
        unit: MemoryUnit,
        now: datetime,
        half_life_days: Optional[float],
    ) -> float:
        """
        Compute recency factor rec(m_i) ∈ [0, 1].

        "Non-increasing function of age parameterized by time_decay_half_life_days;
        when null, rec(m_i) is set to constant." — Appendix A
        """
        if half_life_days is None:
            return 1.0  # No recency weighting

        if not unit.created_at:
            return 0.5  # Default for unknown age

        try:
            created = datetime.fromisoformat(unit.created_at.replace("Z", ""))
            age_days = (now - created).total_seconds() / 86400.0
            # Exponential decay with half-life
            return math.exp(-math.log(2) * age_days / half_life_days)
        except (ValueError, TypeError):
            return 0.5

    # ------------------------------------------------------------------
    # Entity-swap augmentation (Equation 10)
    # ------------------------------------------------------------------

    def _entity_swap_retrieve(
        self,
        query: str,
        cfg: RetrievalConfig,
    ) -> List[MemoryUnit]:
        """
        Adversarial entity-swap retrieval.

        "Adversarial entity-swap strips detected person names from the query and
        re-searches by topic, then unions results with the original retrieval set."
        — Section 3.2

        q_swap = q \\ {p : p ∈ persons(q)}
        R_swap(q; theta) = R_fuse(q_swap; theta)
        — Equation 10, Appendix A

        The final set is R(q;theta) = R_fuse(q;theta) ∪ R_swap(q;theta)
        when enable_entity_swap=true.
        """
        # Strip detected person names (capitalized words as approximation)
        persons_in_query = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", query)
        q_swap = query
        for person in persons_in_query:
            q_swap = q_swap.replace(person, "").strip()

        if not q_swap or q_swap == query:
            return []

        swap_cfg = RetrievalConfig(
            keyword_top_k=cfg.entity_swap_keyword_top_k,
            semantic_top_k=cfg.entity_swap_semantic_top_k,
            structured_top_k=0,
            max_context=cfg.max_context,
            fusion_mode=cfg.fusion_mode,
            enable_entity_swap=False,   # No recursion
            enable_query_decomposition=False,
        )
        return self._retrieve_single(q_swap, swap_cfg)

    # ------------------------------------------------------------------
    # Query decomposition (Equation 11)
    # ------------------------------------------------------------------

    def _decompose_and_retrieve(
        self,
        query: str,
        cfg: RetrievalConfig,
        query_vec=None,
    ) -> List[MemoryUnit]:
        """
        Multi-hop query decomposition.

        "Query decomposition uses an LLM to split multi-hop questions into single-hop
        sub-queries and merges the results via RRF." — Section 3.2

        {q_1, ..., q_K} = psi_dec(q), K <= N_sub
        R_dec(q; theta) = union_k R(q_k; theta)
        — Equation 11, Appendix A

        Falls back to single-query retrieval if decomposition fails.
        """
        if self.llm_client is None:
            return self._retrieve_single(query, cfg, query_vec)

        max_n = cfg.decomposition_max_subqs
        prompt = (
            f"Split this question into 1–{max_n} single-hop sub-questions, "
            f"each answerable from one piece of evidence. "
            f"If the original question IS already single-hop, return only the original.\n\n"
            f"Question: {query}\n\n"
            f"Return JSON list of sub-question strings: [\"sub1\", \"sub2\", ...]\n"
            f"Return ONLY the JSON array."
        )

        try:
            raw = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                system="You are a question decomposition assistant. JSON output only.",
            )
            cleaned = re.sub(r"```json\s*|```\s*", "", raw).strip()
            sub_queries = json.loads(cleaned)
            assert isinstance(sub_queries, list) and sub_queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}. Falling back to single query.")
            return self._retrieve_single(query, cfg, query_vec)

        # Retrieve for each sub-query, then RRF-merge
        sub_results: Dict[str, List[MemoryUnit]] = {}
        for i, sq in enumerate(sub_queries[:max_n]):
            units = self._retrieve_single(sq, cfg, query_vec=None)
            sub_results[f"sub_{i}"] = units

        # RRF merge across sub-query results
        return self._rrf(sub_results)
