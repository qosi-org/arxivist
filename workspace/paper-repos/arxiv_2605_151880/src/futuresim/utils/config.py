"""
Configuration and Seed Utilities
==================================
Manages FutureSim configuration loading and reproducibility.

Paper reference: Section 4.1 (Experimental Setup), Appendix B.2 (Simulation Logic)
  "We evaluate all models in their recommended harness at maximum reasoning effort over 3 seeds."
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class SimulationConfig:
    start_date: str = "2025-12-24"
    end_date: str = "2026-03-28"
    timegap_days: int = 1
    max_outcomes_per_question: int = 5   # Paper Section 4.1: "≤ 5 per question"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class CorpusConfig:
    ccnews_path: str = "data/ccnews/"
    index_path: str = "data/lancedb_index/"
    chunk_size: int = 512        # Paper Section 4.1: "5 chunks of 512 tokens"
    chunks_per_query: int = 5   # Paper Section 4.1
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"  # Paper Section 4.1
    # ASSUMED: embedding_dim=4096 — not stated in paper


@dataclass
class QuestionsConfig:
    questions_path: str = "data/questions.csv"
    source: str = "Al Jazeera / CCNews"
    answer_matcher_model: str = "deepseek-chat"   # DeepSeek V3.2 (Section 4.1)
    answer_matcher_base_url: str = "https://api.deepseek.com"
    match_cache_path: str = "data/match_cache.json"


@dataclass
class ScoringConfig:
    metric: str = "brier_skill_score"
    also_compute: list[str] = field(default_factory=lambda: ["accuracy", "time_weighted"])


@dataclass
class SandboxConfig:
    use_bwrap: bool = True
    block_network: bool = True
    workspace_path: str = "agent_workspace/"
    # ASSUMED: bwrap args for namespace isolation; not fully specified in paper (Appendix B.3)


@dataclass
class HarnessConfig:
    type: str = "native"     # "native" or "custom"
    max_actions: int = 200   # ASSUMED: not specified
    max_total_tokens: int = 200_000  # ASSUMED: context window management


@dataclass
class HardwareConfig:
    device: str = "cuda"
    embedding_gpu: bool = True
    num_workers: int = 4


@dataclass
class SimConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    questions: QuestionsConfig = field(default_factory=QuestionsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def load_config(config_path: str) -> SimConfig:
    """
    Load simulation configuration from a YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Populated SimConfig dataclass
    """
    path = Path(config_path)
    assert path.exists(), f"Config file not found: {path}"
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = SimConfig()
    if "simulation" in raw:
        for k, v in raw["simulation"].items():
            if hasattr(cfg.simulation, k):
                setattr(cfg.simulation, k, v)
            else:
                raise ValueError(f"Unknown simulation config key: {k!r}")
    if "corpus" in raw:
        for k, v in raw["corpus"].items():
            if hasattr(cfg.corpus, k):
                setattr(cfg.corpus, k, v)
    if "questions" in raw:
        for k, v in raw["questions"].items():
            if hasattr(cfg.questions, k):
                setattr(cfg.questions, k, v)
    if "sandbox" in raw:
        for k, v in raw["sandbox"].items():
            if hasattr(cfg.sandbox, k):
                setattr(cfg.sandbox, k, v)
    if "harness" in raw:
        for k, v in raw["harness"].items():
            if hasattr(cfg.harness, k):
                setattr(cfg.harness, k, v)
    if "hardware" in raw:
        for k, v in raw["hardware"].items():
            if hasattr(cfg.hardware, k):
                setattr(cfg.hardware, k, v)
    return cfg


def set_seed(seed: int) -> None:
    """
    Set random seed for full reproducibility across Python, NumPy.

    Paper reference: Section 4.1 — "evaluated over 3 seeds"

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
