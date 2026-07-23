"""
utils/config.py
---------------
Typed configuration dataclasses for RED-COHORT-2026.
Loaded from configs/config.yaml via PipelineConfig.from_yaml().

Paper: Kamat (2026), Section 4 (detection hyperparameters),
       Section 6 (causal analysis hyperparameters).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import yaml


@dataclass
class DetectionConfig:
    """Hyperparameters for the two-stage detection pipeline (Sections 4.1–4.2)."""
    first_buyer_window: int = 10
    edge_weight_cutoff: int = 3
    max_cohort_size: int = 12
    score_tau: float = 40.0         # ASSUMED: tau undisclosed (SIR confidence 0.55)
    touch_threshold_score: int = 1  # ASSUMED: >=1 for scoring (SIR confidence 0.68)
    touch_threshold_causal: int = 2
    ablation_cutoffs: List[int] = field(default_factory=lambda: [2, 3, 5])

    def __post_init__(self) -> None:
        if self.first_buyer_window < 1:
            raise ValueError(f"first_buyer_window must be >= 1, got {self.first_buyer_window}")
        if self.edge_weight_cutoff < 1:
            raise ValueError(f"edge_weight_cutoff must be >= 1, got {self.edge_weight_cutoff}")
        if self.max_cohort_size < 2:
            raise ValueError(f"max_cohort_size must be >= 2, got {self.max_cohort_size}")
        if self.score_tau < 0:
            raise ValueError(f"score_tau must be >= 0, got {self.score_tau}")
        if self.touch_threshold_score not in (1, 2):
            raise ValueError(f"touch_threshold_score must be 1 or 2, got {self.touch_threshold_score}")
        if self.touch_threshold_causal not in (1, 2):
            raise ValueError(f"touch_threshold_causal must be 1 or 2, got {self.touch_threshold_causal}")


@dataclass
class CausalConfig:
    """Hyperparameters for causal buyer-flow analysis (Section 6)."""
    window_minutes: int = 30
    control_ratio: int = 3
    random_seed: int = 42
    bootstrap_iterations: int = 1000
    bootstrap_ci_level: float = 0.95
    activity_match_tolerance: int = 100
    top_k_exclusion: int = 3

    def __post_init__(self) -> None:
        if self.control_ratio < 1:
            raise ValueError(f"control_ratio must be >= 1, got {self.control_ratio}")
        if not 0 < self.bootstrap_ci_level < 1:
            raise ValueError(f"bootstrap_ci_level must be in (0,1), got {self.bootstrap_ci_level}")
        if self.bootstrap_iterations < 100:
            raise ValueError(f"bootstrap_iterations should be >= 100, got {self.bootstrap_iterations}")


@dataclass
class TierConfig:
    """Tier classification thresholds (Section 5 / Figure 3)."""
    premium_min_launches: int = 20
    high_min_launches: int = 10
    high_min_score: float = 100.0


@dataclass
class DataConfig:
    """File paths (all configurable — never hardcoded)."""
    buyers_path: str = "data/pumpfun_buyers.jsonl"
    launches_path: str = "data/pumpfun_launches.jsonl"
    intra_path: str = "data/sniper_cohorts_intra.jsonl.gz"
    output_dir: str = "results/"


@dataclass
class HardwareConfig:
    """Runtime resource settings."""
    n_workers: int = 4
    chunk_size: int = 100_000


@dataclass
class PipelineConfig:
    """Root config aggregating all sub-configs."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from a YAML file and return a validated PipelineConfig."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        detection = DetectionConfig(**raw.get("detection", {}))
        causal = CausalConfig(**raw.get("causal", {}))
        tier = TierConfig(**raw.get("tier_thresholds", {}))
        data = DataConfig(**raw.get("data", {}))
        hardware = HardwareConfig(**raw.get("hardware", {}))

        return cls(
            detection=detection,
            causal=causal,
            tier=tier,
            data=data,
            hardware=hardware,
        )

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  detection={self.detection},\n"
            f"  causal={self.causal},\n"
            f"  tier={self.tier},\n"
            f"  data={self.data},\n"
            f"  hardware={self.hardware}\n"
            f")"
        )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy for reproducibility. (No PyTorch — pipeline is CPU-only.)"""
    random.seed(seed)
    np.random.seed(seed)
