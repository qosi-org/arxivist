"""
src/geomherd/utils/config.py
Config loading utilities for GeomHerd.
Paper: arXiv:2605.11645 — Section 2 (hyperparameters throughout)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch optional for geometry-only pipeline


@dataclass
class GraphConfig:
    Tw: int = 100
    w0: float = 0.5
    delta_t: int = 10
    overlap_pct: float = 0.5
    action_alphabet_size: int = 3


@dataclass
class CurvatureConfig:
    alpha: float = 0.5
    kappa_plus_thresh: float = 0.1
    kappa_minus_thresh: float = -0.1
    use_lp_wasserstein: bool = True


@dataclass
class RicciFlowConfig:
    step_size: float = 0.01          # ASSUMED
    max_iter: int = 1000             # ASSUMED
    flow_variant: str = "multiplicative"  # ASSUMED; see Risk R1
    neckpinch_threshold: float = -50.0   # ASSUMED proxy for kappa -> -inf


@dataclass
class OperatingPoint:
    k_sigma: float = 2.0
    h_sigma: float = 4.0
    recall_super: float = 0.04
    far_sub: float = 0.07
    median_lead_steps: int = 178


@dataclass
class DetectionConfig:
    baseline_window: int = 35
    skip_initial: int = 50
    operating_point: str = "precision"  # recall | precision
    recall_oriented: OperatingPoint = field(default_factory=lambda: OperatingPoint(
        k_sigma=0.5, h_sigma=4.0, recall_super=0.52, far_sub=0.76, median_lead_steps=272))
    precision_oriented: OperatingPoint = field(default_factory=OperatingPoint)
    beta_minus_k_sigma: float = 0.5     # ASSUMED
    beta_minus_h_sigma: float = 4.0     # ASSUMED
    kendall_tau_thresh: float = -0.4    # ASSUMED; inferred from Table 3 label
    kendall_window: int = 20            # ASSUMED
    herding_event_threshold: float = 0.5
    geom_alarm_threshold: float = 0.30

    def active_operating_point(self) -> OperatingPoint:
        if self.operating_point == "recall":
            return self.recall_oriented
        return self.precision_oriented


@dataclass
class VocabularyConfig:
    codebook_dims: int = 3
    levels_per_dim: int = 4
    K: int = 64  # = levels_per_dim^codebook_dims


@dataclass
class CWSConfig:
    N_agents: int = 66
    N_assets: int = 4
    kappa_values: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.2, 1.8, 2.5])
    seeds_per_kappa: int = 80
    T_steps: int = 1200    # ASSUMED from Figure 3
    sbase: float = 0.6
    spost: float = 1.6


@dataclass
class VicsekConfig:
    N_particles: int = 600
    eta_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.6, 2.0, 2.5])
    eta_critical: float = 1.6
    seeds_per_eta: int = 20
    T_steps: int = 1000
    snapshot_stride: int = 50
    knn_k: int = 10
    speed: float = 0.3      # ASSUMED
    radius: float = 1.0     # ASSUMED
    polarisation_threshold: float = 0.5


@dataclass
class SimulationConfig:
    llm_mode: bool = False
    llm_model: str = "claude-sonnet-4-20250514"  # ASSUMED; paper used Opus 4.6
    llm_temperature: float = 0.7  # ASSUMED
    cws: CWSConfig = field(default_factory=CWSConfig)
    vicsek: VicsekConfig = field(default_factory=VicsekConfig)


@dataclass
class KronosConfig:
    # STUB: architecture not specified in paper (Risk R2)
    d_model: int = 64           # ASSUMED
    n_layers: int = 2           # ASSUMED
    n_heads: int = 4            # ASSUMED
    tokeniser_codebook_size: int = 512   # ASSUMED
    context_len: int = 64       # ASSUMED
    train_epochs: int = 50      # ASSUMED
    lr: float = 1e-4            # ASSUMED
    batch_size: int = 32        # ASSUMED
    cascade_window_steps: int = 100  # ASSUMED


@dataclass
class EvaluationConfig:
    n_boot: int = 5000
    alpha_level: float = 0.05


@dataclass
class HardwareConfig:
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42


@dataclass
class PathConfig:
    data_dir: str = "data/"
    output_dir: str = "results/"
    checkpoint_dir: str = "results/checkpoints/"
    log_dir: str = "results/logs/"


@dataclass
class GeomHerdConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    ricci_flow: RicciFlowConfig = field(default_factory=RicciFlowConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "GeomHerdConfig":
        """Load config from YAML, merging with dataclass defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "graph" in raw:
            for k, v in raw["graph"].items():
                setattr(cfg.graph, k, v)
        if "curvature" in raw:
            for k, v in raw["curvature"].items():
                setattr(cfg.curvature, k, v)
        if "ricci_flow" in raw:
            for k, v in raw["ricci_flow"].items():
                setattr(cfg.ricci_flow, k, v)
        if "detection" in raw:
            d = raw["detection"]
            for k, v in d.items():
                if k not in ("recall_oriented", "precision_oriented", "beta_minus"):
                    setattr(cfg.detection, k, v)
        if "vocabulary" in raw:
            for k, v in raw["vocabulary"].items():
                setattr(cfg.vocabulary, k, v)
        if "hardware" in raw:
            for k, v in raw["hardware"].items():
                setattr(cfg.hardware, k, v)
        if "paths" in raw:
            for k, v in raw["paths"].items():
                setattr(cfg.paths, k, v)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.graph.Tw <= 0:
            raise ValueError(f"Tw must be positive, got {self.graph.Tw}")
        if not 0 < self.graph.w0 < 1:
            raise ValueError(f"w0 must be in (0,1), got {self.graph.w0}")
        if self.curvature.alpha <= 0 or self.curvature.alpha >= 1:
            raise ValueError(f"alpha must be in (0,1), got {self.curvature.alpha}")
        if self.vocabulary.K != self.vocabulary.levels_per_dim ** self.vocabulary.codebook_dims:
            raise ValueError("K must equal levels_per_dim^codebook_dims")
        assert self.detection.operating_point in ("recall", "precision"), \
            f"operating_point must be 'recall' or 'precision', got {self.detection.operating_point}"

    def to_yaml(self, path: str) -> None:
        """Serialize config back to YAML."""
        import dataclasses
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
