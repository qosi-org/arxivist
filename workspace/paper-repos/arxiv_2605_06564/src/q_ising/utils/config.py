"""
Configuration utilities for Q-Ising.
Loads YAML configs into typed dataclasses and provides reproducibility seeding.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Seed utility (required by all entrypoints)
# ---------------------------------------------------------------------------

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable PyTorch deterministic mode (may slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IsingConfig:
    """Hyperparameters for the Dynamic Ising Model (Stage 1).
    See Section 3.1 and Appendix E of arXiv:2605.06564.
    """
    estimation_method: str = "emvs"   # "emvs" | "mcmc"
    v0: float = 0.01                  # Spike variance (Section 3.1)
    v1: float = 10.0                  # Slab variance (Section 3.1)
    c: float = 1.0                    # Inclusion scale (Section 3.1)
    tau_sq: float = 10.0              # Beta prior variance (Section 3.1)
    emvs_n_iter: int = 10             # Max EMVS iterations
    mcmc_n_draws: int = 200           # HMC posterior draws (Appendix E.3)
    mcmc_n_tune: int = 300            # HMC tuning iterations (Appendix E.3)

    def __post_init__(self):
        if self.estimation_method not in ("emvs", "mcmc"):
            raise ValueError(f"estimation_method must be 'emvs' or 'mcmc', got {self.estimation_method!r}")
        if self.v0 >= self.v1:
            raise ValueError(f"Spike variance v0={self.v0} must be < slab variance v1={self.v1}")


@dataclass
class CQLConfig:
    """Hyperparameters for Conservative Q-Learning (Stage 2).
    See Appendix E.2 of arXiv:2605.06564.
    """
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4       # Appendix E.2
    batch_size: int = 64              # Appendix E.2
    max_steps: int = 30_000           # Appendix E.2
    steps_per_epoch: int = 1_000      # Appendix E.2
    early_stopping_patience: int = 10 # Appendix E.2
    early_stopping_min_delta: float = 1e-4  # Appendix E.2
    dropout_rate: float = 0.3         # Appendix E.2
    batch_norm: bool = True           # Appendix E.2
    alpha: float = 0.1                # Conservative penalty (Appendix E.2)
    discount: float = 0.8             # Discount factor psi (Appendix E.2)
    # ASSUMED: activation=relu — standard for CQL/d3rlpy (confidence: 0.80)
    activation: str = "relu"
    n_ensemble_agents: int = 20       # Section 3.3


@dataclass
class SISConfig:
    """SIS dynamics parameters for synthetic data generation.
    See Appendix E.1 and Section 5 of arXiv:2605.06564.
    """
    # SBM experiment (Section 5.1)
    n_per_block: List[int] = field(default_factory=lambda: [187, 187, 63, 63])
    p_in: float = 0.1
    p_out: float = 0.01
    spread_rates: List[float] = field(default_factory=lambda: [0.010, 0.012, 0.1, 0.12])
    churn_rates: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2, 0.2])


@dataclass
class NetworkConfig:
    """Network and binning configuration."""
    N: Optional[int] = None           # Number of nodes (inferred for villages)
    K: Optional[int] = None           # Number of bins (inferred for villages)
    bin_method: str = "spectral"      # "spectral" | "edge_betweenness" | "manual"
    min_community_size: int = 10      # Appendix E.3


@dataclass
class TrainingConfig:
    """Training and evaluation protocol."""
    T_train: int = 100                # Training panel length
    H_test: int = 25                  # Test horizon (Section 5)
    n_test_runs: int = 50             # Independent test runs (Section 5)
    village_ids: Optional[List[int]] = None  # None = all villages


@dataclass
class PathConfig:
    """File system paths."""
    data_dir: str = "data/"
    output_dir: str = "results/"
    checkpoint_dir: str = "results/checkpoints/"
    log_every_steps: int = 500
    checkpoint_every_steps: int = 5000


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    experiment: str = "sbm"           # "sbm" | "village"
    seed: int = 42
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sis: SISConfig = field(default_factory=SISConfig)
    ising: IsingConfig = field(default_factory=IsingConfig)
    cql: CQLConfig = field(default_factory=CQLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self):
        if self.experiment not in ("sbm", "village"):
            raise ValueError(f"experiment must be 'sbm' or 'village', got {self.experiment!r}")


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Args:
        path: Path to config YAML file.

    Returns:
        Fully validated ExperimentConfig dataclass.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def _get(d, key, default=None):
        return d.get(key, default) if d else default

    ising_raw = _get(raw, "ising", {})
    cql_raw = _get(raw, "cql", {})
    sis_raw = _get(raw, "sis", {})
    network_raw = _get(raw, "network", {})
    training_raw = _get(raw, "training", {})
    paths_raw = _get(raw, "paths", {})

    return ExperimentConfig(
        experiment=raw.get("experiment", "sbm"),
        seed=raw.get("seed", 42),
        network=NetworkConfig(**{k: v for k, v in network_raw.items() if k in NetworkConfig.__dataclass_fields__}),
        sis=SISConfig(**{k: v for k, v in sis_raw.items() if k in SISConfig.__dataclass_fields__}),
        ising=IsingConfig(**{k: v for k, v in ising_raw.items() if k in IsingConfig.__dataclass_fields__}),
        cql=CQLConfig(**{k: v for k, v in cql_raw.items() if k in CQLConfig.__dataclass_fields__}),
        training=TrainingConfig(**{k: v for k, v in training_raw.items() if k in TrainingConfig.__dataclass_fields__}),
        paths=PathConfig(**{k: v for k, v in paths_raw.items() if k in PathConfig.__dataclass_fields__}),
    )
