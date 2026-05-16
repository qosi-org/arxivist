"""
utils/config.py — Configuration loader and reproducibility utilities.

Paper: AGENTICAITA (arxiv:2605.12532)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Dataclasses — mirror of configs/default.yaml
# ---------------------------------------------------------------------------

@dataclass
class PollingConfig:
    interval_s: int = 60
    assets: List[str] = field(default_factory=list)


@dataclass
class TriggerConfig:
    """AZTE parameters — Section 4.1, Table 2."""
    z_threshold: float = 2.0      # Eq. 3: z_t >= 2.0
    r_floor: float = 0.003        # Eq. 3: r_t >= 0.003
    window_bars: int = 30         # W=30 rolling window
    per_asset_cooldown_s: int = 300


@dataclass
class IGPConfig:
    """Inference Gating Protocol — Section 4.4, Table 2."""
    global_cooldown_s: int = 1800


@dataclass
class RiskManagerConfig:
    """Risk Manager hard-gate thresholds — Section 4.2, Eqs. 4-7."""
    min_confidence: float = 0.60   # Eq. 5
    max_risk_pct: float = 0.02     # Eq. 6
    max_size_usd: float = 500.0    # Eq. 7


@dataclass
class CBDConfig:
    """Correlation-Break Diversification — Section 4.5, Eqs. 9-11."""
    alpha: float = 0.5   # Eq. 11 — equal weight between anomaly and decorrelation
    kappa: float = 0.5   # Eq. 10 — exponential saturation rate
    window_bars: int = 30


@dataclass
class LLMConfig:
    ollama_base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    max_tokens: int = 512
    timeout_s: int = 30


@dataclass
class ExecutionConfig:
    mode: str = "DRY_RUN"         # DRY_RUN | LIVE
    tor_socks_host: str = "localhost"
    tor_socks_port: int = 9050


@dataclass
class CostModelConfig:
    """Transaction cost model — Section 4.6, Eqs. 12-13."""
    f_taker: float = 0.0005       # ASSUMED: 0.05% taker fee
    lambda_impact: float = 0.8    # Explicit: calibrated to crypto perpetuals


@dataclass
class DatabaseConfig:
    path: str = "data/agenticaita.db"
    wal_mode: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class Config:
    polling: PollingConfig = field(default_factory=PollingConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    igp: IGPConfig = field(default_factory=IGPConfig)
    risk_manager: RiskManagerConfig = field(default_factory=RiskManagerConfig)
    cbd: CBDConfig = field(default_factory=CBDConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

    def validate(self) -> None:
        """Validate config values at load time."""
        if self.trigger.z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {self.trigger.z_threshold}")
        if self.trigger.window_bars < 2:
            raise ValueError(f"window_bars must be >= 2, got {self.trigger.window_bars}")
        if not 0 < self.risk_manager.min_confidence <= 1:
            raise ValueError(f"min_confidence must be in (0,1], got {self.risk_manager.min_confidence}")
        if self.risk_manager.max_risk_pct <= 0 or self.risk_manager.max_risk_pct > 1:
            raise ValueError(f"max_risk_pct must be in (0,1], got {self.risk_manager.max_risk_pct}")
        if self.execution.mode not in ("DRY_RUN", "LIVE"):
            raise ValueError(f"execution.mode must be DRY_RUN or LIVE, got {self.execution.mode}")
        if not 0 < self.cbd.alpha < 1:
            raise ValueError(f"cbd.alpha must be in (0,1), got {self.cbd.alpha}")


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        polling=PollingConfig(**raw.get("polling", {})),
        trigger=TriggerConfig(**raw.get("trigger", {})),
        igp=IGPConfig(**raw.get("igp", {})),
        risk_manager=RiskManagerConfig(**raw.get("risk_manager", {})),
        cbd=CBDConfig(**raw.get("cbd", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        cost_model=CostModelConfig(**raw.get("cost_model", {})),
        database=DatabaseConfig(**raw.get("database", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        seed=raw.get("seed", 42),
    )
    cfg.validate()
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy for reproducibility. (No PyTorch — training-free system.)"""
    random.seed(seed)
    np.random.seed(seed)
