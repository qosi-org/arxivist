"""
utils/config.py
───────────────
Configuration loading (YAML → dataclasses) and reproducibility utilities.
All random-state seeding is centralised here so every entrypoint calls a
single function to guarantee full reproducibility.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    signature_truncation_N: int = 3
    primary_process: str = "heston_variance"
    S0: float = 100.0
    r: float = 0.0

    def __post_init__(self):
        valid = {"heston_variance", "fbm_raw", "fbm_exp", "fbm_shifted_exp"}
        if self.primary_process not in valid:
            raise ValueError(
                f"primary_process must be one of {valid}, got '{self.primary_process}'"
            )
        if self.signature_truncation_N < 1 or self.signature_truncation_N > 5:
            raise ValueError(
                f"signature_truncation_N should be 1–5, got {self.signature_truncation_N}. "
                "N=3 is recommended (paper default). N≥4 is very expensive."
            )


@dataclass
class HestonPrimaryConfig:
    X0: float = 0.1
    x0_is_variance: bool = True   # ASSUMED; see SIR ambiguity R4
    nu: float = 0.2
    kappa: float = 2.0
    theta: float = 0.15
    rho_asset_vol: float = 0.0


@dataclass
class FBMPrimaryConfig:
    H: float = 0.2
    X0: float = 0.1
    rho_asset_vol: float = -0.6

    def __post_init__(self):
        if not (0.0 < self.H < 1.0):
            raise ValueError(f"H must be in (0,1), got {self.H}")


@dataclass
class HestonMarketConfig:
    sigma0: float = 0.2
    nu: float = 0.3
    kappa: float = 3.0
    theta: float = 0.09
    rho: float = 0.0


@dataclass
class RBergomiMarketConfig:
    sigma0: float = 0.2
    H: float = 0.1
    eta: float = 0.5
    rho: float = -0.7


@dataclass
class SimulationConfig:
    nMC: int = 800_000              # paper: Section 4.3
    T_steps_per_unit: int = 252     # ASSUMED: daily steps
    seed: int = 42                  # ASSUMED
    interpolation: str = "linear"   # paper: Section 4.3
    fbm_method: str = "cholesky"    # ASSUMED
    cholesky_reg_eps: float = 1e-8  # ASSUMED: Cholesky regularisation for -Q


@dataclass
class CalibrationConfig:
    optimizer: str = "L-BFGS-B"                    # paper: Section 4.3
    tolerance: float = 1e-8                         # paper: Section 4.3
    l0_init: str = "zeros"                          # ASSUMED
    box_bounds: List[float] = field(default_factory=lambda: [-10.0, 10.0])  # ASSUMED
    maturities: List[float] = field(default_factory=lambda: [0.1, 0.6, 1.1, 1.6])
    strikes: List[float] = field(default_factory=lambda: [90.0, 95.0, 100.0, 105.0, 110.0])
    weight_scheme: str = "inverse_vega"             # paper: Section 5.1
    max_iter: int = 10_000                          # ASSUMED

    def __post_init__(self):
        if len(self.box_bounds) != 2:
            raise ValueError("box_bounds must be [lower, upper]")
        if self.box_bounds[0] >= self.box_bounds[1]:
            raise ValueError("box_bounds[0] must be < box_bounds[1]")


@dataclass
class VIXCalibrationConfig:
    Delta_trading_days: int = 30    # paper: Section 2.2
    T1: float = 0.1                 # ASSUMED
    T2: float = 0.5                 # ASSUMED


@dataclass
class HardwareConfig:
    device: str = "cuda"
    num_workers: int = 4            # ASSUMED
    use_float64: bool = True        # ASSUMED: financial precision

    def __post_init__(self):
        if self.device not in {"cuda", "cpu"}:
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'")


@dataclass
class PathsConfig:
    output_dir: str = "results/"
    checkpoint_dir: str = "results/checkpoints/"
    data_dir: str = "data/"


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Root configuration dataclass for the volsig pipeline.
    All fields correspond 1-to-1 with sections in configs/config.yaml.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    heston_primary: HestonPrimaryConfig = field(default_factory=HestonPrimaryConfig)
    fbm_primary: FBMPrimaryConfig = field(default_factory=FBMPrimaryConfig)
    heston_market: HestonMarketConfig = field(default_factory=HestonMarketConfig)
    rbergomi_market: RBergomiMarketConfig = field(default_factory=RBergomiMarketConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    vix_calibration: VIXCalibrationConfig = field(default_factory=VIXCalibrationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file, merging with dataclass defaults."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            heston_primary=HestonPrimaryConfig(**raw.get("heston_primary", {})),
            fbm_primary=FBMPrimaryConfig(**raw.get("fbm_primary", {})),
            heston_market=HestonMarketConfig(**raw.get("heston_market", {})),
            rbergomi_market=RBergomiMarketConfig(**raw.get("rbergomi_market", {})),
            simulation=SimulationConfig(**raw.get("simulation", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
            vix_calibration=VIXCalibrationConfig(**raw.get("vix_calibration", {})),
            hardware=HardwareConfig(**raw.get("hardware", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )

    def __repr__(self) -> str:
        return f"Config(primary_process={self.model.primary_process}, nMC={self.simulation.nMC}, N={self.model.signature_truncation_N})"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable PyTorch deterministic mode.
            Note: may slow down GPU operations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch optional for CPU-only runs


def ensure_dirs(cfg: Config) -> None:
    """Create output directories specified in config if they don't exist."""
    for d in [cfg.paths.output_dir, cfg.paths.checkpoint_dir, cfg.paths.data_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
