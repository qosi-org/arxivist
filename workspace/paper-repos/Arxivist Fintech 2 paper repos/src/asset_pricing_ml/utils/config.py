"""
utils/config.py — Configuration loading, validation, and seed management.

Paper: Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning"
       Review of Financial Studies 33, 2223-2273. doi:10.1093/rfs/hhaa009
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer random seed.
        deterministic: If True, enables PyTorch deterministic mode (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass
class DataConfig:
    crsp_path: str = "data/raw/crsp_returns.parquet"
    chars_path: str = "data/raw/characteristics.parquet"
    macro_path: str = "data/raw/macro_predictors.csv"
    train_end_year: int = 1974
    val_end_year: int = 1986
    test_end_year: int = 2016
    refit_frequency: str = "annual"
    val_window_years: int = 12
    missing_fill: str = "cross_sectional_median"
    char_lag_monthly_months: int = 1
    char_lag_quarterly_months: int = 4
    char_lag_annual_months: int = 6
    n_characteristics: int = 94
    n_macro_predictors: int = 8
    n_industry_dummies: int = 74
    total_features: int = 920


@dataclass
class ModelConfig:
    variant: str = "NN3"
    nn_architectures: Dict[str, List[int]] = field(default_factory=lambda: {
        "NN1": [32], "NN2": [32, 16], "NN3": [32, 16, 8],
        "NN4": [32, 16, 8, 4], "NN5": [32, 16, 8, 4, 2]
    })
    activation: str = "relu"
    use_batch_norm: bool = True
    use_huber_loss: bool = True
    # ASSUMED: ensemble seed count not stated in paper (confidence 0.50)
    # TODO: verify from Internet Appendix B.3
    n_ensemble_seeds: int = 10

    def get_nn_layers(self, variant: Optional[str] = None) -> List[int]:
        """Return hidden layer sizes for a given NN variant."""
        v = variant or self.variant
        if v not in self.nn_architectures:
            raise ValueError(f"Unknown NN variant '{v}'. Choose from: {list(self.nn_architectures)}")
        return self.nn_architectures[v]


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    # ASSUMED: lr not stated (confidence 0.65) — TODO: verify from Internet Appendix B.3
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    l1_lambda_grid: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1])
    huber_xi_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    # ASSUMED: batch size not stated (confidence 0.45) — TODO: verify from Internet Appendix B.3
    batch_size: int = 512
    # ASSUMED: early stopping patience not stated (confidence 0.55)
    early_stopping_patience: int = 5
    max_epochs: int = 100
    log_every_n_steps: int = 100
    save_checkpoint: bool = True
    checkpoint_dir: str = "models/"


@dataclass
class EvalConfig:
    output_dir: str = "results/"
    figures_dir: str = "results/figures/"
    compute_variable_importance: bool = True
    importance_method: str = "r2_reduction"
    n_portfolio_deciles: int = 10
    portfolio_weighting: str = "value"
    factor_model: str = "FF5_plus_momentum"
    market_timing_max_leverage: float = 1.5
    market_timing_allow_short: bool = False


@dataclass
class HardwareConfig:
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    deterministic: bool = False


@dataclass
class Config:
    """Top-level config for Gu, Kelly, Xiu (2020) reproduction."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        _apply(cfg.data, raw.get("data", {}))
        _apply(cfg.model, raw.get("model", {}))
        _apply(cfg.training, raw.get("training", {}))
        _apply(cfg.eval, raw.get("evaluation", {}))
        _apply(cfg.hardware, raw.get("hardware", {}))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate config values. Raises ValueError on invalid config."""
        valid_variants = {"OLS", "OLS3", "ENet", "PCR", "PLS", "GLM",
                          "RF", "GBRT", "NN1", "NN2", "NN3", "NN4", "NN5"}
        if self.model.variant not in valid_variants:
            raise ValueError(f"model.variant must be one of {valid_variants}")
        if self.data.train_end_year >= self.data.val_end_year:
            raise ValueError("train_end_year must be < val_end_year")
        if self.data.val_end_year >= self.data.test_end_year:
            raise ValueError("val_end_year must be < test_end_year")
        if self.data.total_features != (
            self.data.n_characteristics * (self.data.n_macro_predictors + 1)
            + self.data.n_industry_dummies
        ):
            raise ValueError(
                f"total_features mismatch: expected "
                f"{self.data.n_characteristics}*({self.data.n_macro_predictors}+1)"
                f"+{self.data.n_industry_dummies} = "
                f"{self.data.n_characteristics*(self.data.n_macro_predictors+1)+self.data.n_industry_dummies}, "
                f"got {self.data.total_features}"
            )

    def __repr__(self) -> str:
        return (
            f"Config(variant={self.model.variant}, "
            f"train={self.data.train_end_year}, val={self.data.val_end_year}, "
            f"test={self.data.test_end_year}, features={self.data.total_features})"
        )


def _apply(obj: object, d: dict) -> None:
    """Apply a flat dict of values to a dataclass instance."""
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
