"""
utils/config.py
===============
Configuration loading and parameter dataclasses.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ContractParams:
    S0: float = 100.0
    K: float  = 100.0
    H: float  = 130.0
    L: float  = 70.0
    T: float  = 1.0
    r: float  = 0.03
    q: float  = 0.0


@dataclass
class CIRParams:
    v0:    float = 0.18
    kappa: float = 0.6
    theta: float = 0.20
    xi:    float = 0.4


@dataclass
class SquaredOUParams:
    Y0:    float = 0.4243
    alpha: float = 0.6
    sigma: float = 0.490


@dataclass
class ClockParams:
    family: str = "cir"
    rho:    float = 0.0
    cir:    CIRParams    = field(default_factory=CIRParams)
    sq_ou:  SquaredOUParams = field(default_factory=SquaredOUParams)


@dataclass
class SingleBarrierConfig:
    n_quad:   int   = 200
    quad_tol: float = 1e-10


@dataclass
class DoubleBarrierConfig:
    n_series_terms: int = 100


@dataclass
class LeverageConfig:
    expansion_order: int = 5
    pade_type:       str = "[3/2]"
    n_paths_duhamel: int = 100_000


@dataclass
class MCConfig:
    n_paths:           int  = 1_000_000
    n_steps_per_year:  int  = 2080
    bridge_correction: bool = True
    seed:              int  = 42


@dataclass
class CalibrationConfig:
    optimizer:            str   = "L-BFGS-B"
    vanilla_weight:       float = 1.0
    barrier_weight:       float = 0.5
    regularization_lambda: float = 1e-3


@dataclass
class PricingConfig:
    contract:    ContractParams    = field(default_factory=ContractParams)
    clock:       ClockParams       = field(default_factory=ClockParams)
    single_barrier: SingleBarrierConfig = field(default_factory=SingleBarrierConfig)
    double_barrier: DoubleBarrierConfig = field(default_factory=DoubleBarrierConfig)
    leverage:    LeverageConfig    = field(default_factory=LeverageConfig)
    monte_carlo: MCConfig          = field(default_factory=MCConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


def load_config(path: str | Path) -> PricingConfig:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = PricingConfig()

    if "contract" in raw:
        c = raw["contract"]
        cfg.contract = ContractParams(**{k: v for k, v in c.items()
                                         if k in ContractParams.__dataclass_fields__})

    if "clock" in raw:
        ck = raw["clock"]
        cfg.clock.family = ck.get("family", "cir")
        cfg.clock.rho    = ck.get("rho", 0.0)
        if "cir" in ck:
            cfg.clock.cir = CIRParams(**ck["cir"])
        if "sq_ou" in ck:
            cfg.clock.sq_ou = SquaredOUParams(**ck["sq_ou"])

    if "monte_carlo" in raw:
        mc = raw["monte_carlo"]
        cfg.monte_carlo = MCConfig(**{k: v for k, v in mc.items()
                                       if k in MCConfig.__dataclass_fields__})

    if "calibration" in raw:
        cal = raw["calibration"]
        cfg.calibration = CalibrationConfig(**{k: v for k, v in cal.items()
                                                if k in CalibrationConfig.__dataclass_fields__})
    return cfg


def build_clock_from_config(cfg: PricingConfig):
    """Instantiate a clock object from PricingConfig."""
    from ..models.cir_clock import CIRClock
    from ..models.sq_ou_clock import SquaredOUClock

    if cfg.clock.family == "cir":
        p = cfg.clock.cir
        return CIRClock(kappa=p.kappa, theta=p.theta, xi=p.xi, v0=p.v0)
    elif cfg.clock.family == "sq_ou":
        p = cfg.clock.sq_ou
        return SquaredOUClock(alpha=p.alpha, sigma=p.sigma, Y0=p.Y0)
    else:
        raise ValueError(f"Unknown clock family: {cfg.clock.family}")
