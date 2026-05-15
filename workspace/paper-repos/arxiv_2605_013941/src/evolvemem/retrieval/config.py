"""
evolvemem/retrieval/config.py

RetrievalConfig: the full evolvable action space theta for EVOLVEMEM.
Implements Section 3.2 "Action space" from:
  "EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents"
  ArXiv: 2605.13941

The complete configuration is:
  theta = (k_sem, k_kw, k_str, B_ctx, mode, {w_v}, alpha, {theta_c}_{c in C})
  — Equation 2, Section 3.2

All parameter ranges are taken from Table 7 in Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import copy


# Parameter ranges from Table 7 (Appendix A)
_RANGES = {
    "semantic_top_k":    (0, 30),
    "keyword_top_k":     (0, 30),
    "structured_top_k":  (0, 30),
    "max_context":       (1, 30),
    "w_sem":             (0.1, 2.5),
    "w_kw":              (0.1, 2.5),
    "w_str":             (0.1, 2.5),
    "reflection_rounds": (0, 3),
}

_FUSION_MODES = {"sum", "weighted_sum", "rrf"}
_ANSWER_STYLES = {"concise", "explanatory", "verifying", "inferential", "strict", "nuanced"}
_VERIFICATION_STYLES = {"strict", "multi_candidate"}


@dataclass
class RetrievalConfig:
    """
    Full retrieval configuration theta — the evolvable action space.

    Every parameter in this dataclass corresponds to a dimension of the action space
    that the Self-Evolution Engine can modify. All values are clamped to safe ranges
    before any proposed adjustment is applied (Algorithm 1, line 12).

    Paper reference: Section 3.2 "Action space", Table 7 (Appendix A)

    Initial configuration theta_0 (Section 4.1):
      - BM25-only fusion (mode=sum, semantic and structured views disabled)
      - keyword_top_k=5, max_context=8
      - entity_swap and query decomposition disabled
    """

    # Retrieval budget (Table 7: ints in [3, 30]; 0 = disabled)
    semantic_top_k: int = 0
    keyword_top_k: int = 5
    structured_top_k: int = 0

    # Context budget (Table 7: int in [6, 30])
    max_context: int = 8

    # Fusion (Table 7: {sum, rrf, weighted_sum})
    fusion_mode: str = "sum"
    w_sem: float = 1.0      # per-view weight for semantic view
    w_kw: float = 1.0       # per-view weight for keyword view
    w_str: float = 1.0      # per-view weight for structured view

    # Entity swap (Table 7: bool + 2 top-ks)
    enable_entity_swap: bool = False
    entity_swap_semantic_top_k: int = 8
    entity_swap_keyword_top_k: int = 8

    # Query decomposition (Table 7: bool + int)
    enable_query_decomposition: bool = False
    decomposition_max_subqs: int = 3

    # Reflection (Table 7: int in [0, 3])
    reflection_rounds: int = 0

    # Answer verification (Table 7: bool + style)
    enable_answer_verification: bool = False
    verification_style: str = "strict"

    # Answer style (Table 7: 6 branches)
    answer_style: str = "concise"

    # Time decay (Table 7: float/null + date/null)
    time_decay_half_life_days: Optional[float] = None
    reference_date: Optional[str] = None

    # Per-category overrides (Table 7: category → sub-config dict)
    # e.g. {"5": {"enable_entity_swap": True, "answer_style": "strict"}}
    per_category_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RetrievalConfig(kw={self.keyword_top_k}, sem={self.semantic_top_k}, "
            f"str={self.structured_top_k}, ctx={self.max_context}, mode={self.fusion_mode}, "
            f"entity_swap={self.enable_entity_swap}, decomp={self.enable_query_decomposition}, "
            f"verify={self.enable_answer_verification})"
        )

    def clamp(self) -> "RetrievalConfig":
        """
        Project all parameters onto their valid ranges.

        Called as clamp_Theta(theta_r ⊕ Delta_theta_r) in the update rule (Equation 4).
        Prevents any proposed adjustment from producing invalid configurations.

        Paper reference: Section 3.3 "Update rule", Algorithm 1 line 12
        """
        cfg = copy.deepcopy(self)

        # Integer ranges
        for attr, (lo, hi) in _RANGES.items():
            val = getattr(cfg, attr, None)
            if val is not None:
                setattr(cfg, attr, max(lo, min(hi, int(val))))

        # Float clamps
        cfg.w_sem = max(0.1, min(2.5, float(cfg.w_sem)))
        cfg.w_kw = max(0.1, min(2.5, float(cfg.w_kw)))
        cfg.w_str = max(0.1, min(2.5, float(cfg.w_str)))

        # Categorical validation
        if cfg.fusion_mode not in _FUSION_MODES:
            cfg.fusion_mode = "sum"
        if cfg.answer_style not in _ANSWER_STYLES:
            cfg.answer_style = "concise"
        if cfg.verification_style not in _VERIFICATION_STYLES:
            cfg.verification_style = "strict"

        return cfg

    def apply_delta(self, delta: dict) -> "RetrievalConfig":
        """
        Apply a proposed delta (from diagnosis module) and return clamped config.

        Implements theta_r ⊕ Delta_theta_r from Algorithm 1 / Equation 4.

        Args:
            delta: Dict of {field_name: new_value} from DiagnosisModule.
                   Fields not in this dict are unchanged.
                   Per-category proposals are merged into per_category_overrides.

        Returns:
            New clamped RetrievalConfig with applied delta.

        Paper reference: Section 3.3, Algorithm 1
        """
        cfg = copy.deepcopy(self)

        per_cat = delta.pop("per_category_overrides", {})

        for key, value in delta.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        # Merge per-category overrides
        if per_cat:
            for cat, overrides in per_cat.items():
                cfg.per_category_overrides[str(cat)] = {
                    **(cfg.per_category_overrides.get(str(cat), {})),
                    **overrides,
                }

        return cfg.clamp()

    def for_category(self, category: Optional[int]) -> "RetrievalConfig":
        """
        Return a config instance with per-category overrides applied.

        Per-category sub-configurations allow each question type to have specialized
        parameters without affecting global defaults (Section 3.2, Table 7).

        Args:
            category: Integer question category (e.g. 1=single-hop, 5=adversarial).

        Returns:
            A new RetrievalConfig with per-category overrides applied if available.
        """
        if category is None or not self.per_category_overrides:
            return self

        overrides = self.per_category_overrides.get(str(category), {})
        if not overrides:
            return self

        cfg = copy.deepcopy(self)
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg.clamp()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalConfig":
        """Deserialize from dict, ignoring unknown keys."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def initial(cls) -> "RetrievalConfig":
        """
        Return the minimal starting configuration theta_0 from Section 4.1.

        "The initial configuration theta_0 uses BM25-only fusion (mode=SUM,
        semantic and structured views disabled), kkw=5, Bctx=8, with entity-swap
        and query decomposition disabled." — Section 4.1
        """
        return cls(
            semantic_top_k=0,
            keyword_top_k=5,
            structured_top_k=0,
            max_context=8,
            fusion_mode="sum",
            enable_entity_swap=False,
            enable_query_decomposition=False,
            enable_answer_verification=False,
        )
