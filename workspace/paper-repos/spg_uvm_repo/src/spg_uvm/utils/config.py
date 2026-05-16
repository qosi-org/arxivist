"""
Configuration utilities for SPG-UVM.

Implements UVMConfig dataclass with YAML loading/saving and seed management.
All hyperparameters traceable to the paper:
  Abbas-Turki et al. (2026), "Stochastic Policy Gradient Methods in the
  Uncertain Volatility Model", arXiv:2605.06670.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d: int = 2                        # Number of risky assets
    hidden_units: int = 32            # Hidden layer size (Section 4.1.3)
    policy_type: str = "continuous"   # "continuous" or "bangbang"


@dataclass
class UVMParamsConfig:
    x0: float = 100.0
    sigma_min: List[float] = field(default_factory=lambda: [0.1, 0.1])
    sigma_max: List[float] = field(default_factory=lambda: [0.2, 0.2])
    rho_min: float = -0.5
    rho_max: float = 0.5
    T: float = 1.0
    r: float = 0.0
    N: int = 128


@dataclass
class PayoffConfig:
    name: str = "geo_outperformer"
    K1: float = 90.0
    K2: float = 110.0


@dataclass
class TrainingConfig:
    M: int = 32768              # MC samples per epoch = 2^15, Section 4.1.3
    minibatch_size: int = 1024  # = 2^10, Section 4.1.3
    E_first: int = 500          # Epochs for n=N-1, Section 4.1.3
    E_subsequent: int = 10      # Epochs for n<=N-2 (transfer learning), Section 4.1.3
    lr_initial: float = 5e-3   # Section 4.1.3
    lr_final: float = 1e-4     # Section 4.1.3
    ppo_epsilon: float = 0.2   # PPO clipping, Section 3.1 / 4.1.3
    normalize_advantages: bool = True   # Standard PPO practice, Section 4.1.3
    antithetic_variates: bool = True    # Section 4.1.3
    transfer_learning: bool = True      # Section 4.1.3
    log_every: int = 10
    checkpoint_every: int = 100


@dataclass
class ExplorationConfig:
    lambda_initial: float = 1.0    # Section 4.1.2
    lambda_final: float = 0.01     # Section 4.1.2
    gamma_initial: float = 0.01    # Section 4.1.2 (bang-bang entropy coeff)
    gamma_final: float = 0.0       # Section 4.1.2
    # ASSUMED: sigmoid_steepness not given explicitly (Figure 1 reference)
    sigmoid_steepness: float = 0.15


@dataclass
class PenaltyConfig:
    beta: float = 10       # Correlation penalty weight, Section 4.1.1
    delta: float = 0.05    # Huber threshold, Section 4.1.1


@dataclass
class EvaluationConfig:
    n_paths_actor_price: int = 524288   # = 2^19, Section 4.1.3
    confidence_level: float = 0.95
    reference_price: Optional[float] = None


@dataclass
class HardwareConfig:
    device: str = "cuda"
    precision: str = "float32"
    seed: int = 42
    deterministic: bool = False


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


@dataclass
class UVMConfig:
    """
    Master configuration for SPG-UVM.

    Load from YAML via UVMConfig.from_yaml(path).
    All fields correspond to hyperparameters in arXiv:2605.06670.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    uvm_params: UVMParamsConfig = field(default_factory=UVMParamsConfig)
    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "UVMConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if "uvm_params" in raw:
            for k, v in raw["uvm_params"].items():
                setattr(cfg.uvm_params, k, v)
        if "payoff" in raw:
            for k, v in raw["payoff"].items():
                setattr(cfg.payoff, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                setattr(cfg.training, k, v)
        if "exploration" in raw:
            for k, v in raw["exploration"].items():
                setattr(cfg.exploration, k, v)
        if "penalty" in raw:
            for k, v in raw["penalty"].items():
                setattr(cfg.penalty, k, v)
        if "evaluation" in raw:
            for k, v in raw["evaluation"].items():
                setattr(cfg.evaluation, k, v)
        if "hardware" in raw:
            for k, v in raw["hardware"].items():
                setattr(cfg.hardware, k, v)
        if "output" in raw:
            for k, v in raw["output"].items():
                setattr(cfg.output, k, v)

        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Validate config consistency."""
        d = self.model.d
        if len(self.uvm_params.sigma_min) != d:
            # Broadcast scalar to all assets
            if len(self.uvm_params.sigma_min) == 1:
                self.uvm_params.sigma_min = self.uvm_params.sigma_min * d
            else:
                raise ValueError(
                    f"sigma_min length {len(self.uvm_params.sigma_min)} != d={d}"
                )
        if len(self.uvm_params.sigma_max) != d:
            if len(self.uvm_params.sigma_max) == 1:
                self.uvm_params.sigma_max = self.uvm_params.sigma_max * d
            else:
                raise ValueError(
                    f"sigma_max length {len(self.uvm_params.sigma_max)} != d={d}"
                )
        if self.model.policy_type not in ("continuous", "bangbang"):
            raise ValueError(
                f"policy_type must be 'continuous' or 'bangbang', got '{self.model.policy_type}'"
            )
        if self.hardware.device.startswith("cuda") and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            self.hardware.device = "cpu"

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import dataclasses
        d = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    def __repr__(self) -> str:
        return (
            f"UVMConfig(d={self.model.d}, policy={self.model.policy_type}, "
            f"N={self.uvm_params.N}, payoff={self.payoff.name})"
        )


# ---------------------------------------------------------------------------
# Seed utility (required for reproducibility)
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables PyTorch deterministic mode (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: may need CUBLAS_WORKSPACE_CONFIG=:4096:8 env var
