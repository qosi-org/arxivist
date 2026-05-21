"""
config.py — Pydantic v2 settings model for AGENTICAITA.
Loads from configs/config.yaml; validates all values at startup.
"""
from __future__ import annotations
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class AZTEConfig(BaseModel):
    polling_interval_s: int = 60
    z_score_threshold: float = 2.0
    rolling_window: int = 30
    absolute_return_floor: float = 0.003


class IGPConfig(BaseModel):
    global_cooldown_s: int = 1800
    per_asset_cooldown_s: int = 300


class RiskManagerConfig(BaseModel):
    confidence_gate: float = 0.60
    max_stop_loss_pct: float = 0.02
    max_position_usd: float = 500.0

    @field_validator("confidence_gate")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"confidence_gate must be in (0, 1), got {v}")
        return v


class CBDConfig(BaseModel):
    alpha: float = 0.5   # Eq. 11 weight
    kappa: float = 0.5   # Eq. 10 saturation rate
    correlation_method: Literal["pearson", "spearman"] = "pearson"  # ASSUMED


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    temperature: float = 0.0   # ASSUMED: 0 for determinism
    max_tokens: int = 512


class DatabaseConfig(BaseModel):
    path: str = "data/episodic_memory.db"
    wal_mode: bool = True


class ExchangeConfig(BaseModel):
    adapter: str = "stub"
    api_key_env: str = "DEX_API_KEY"
    api_secret_env: str = "DEX_API_SECRET"


class PrivacyConfig(BaseModel):
    tor_socks_host: str = "127.0.0.1"
    tor_socks_port: int = 9050
    require_tor_for_live: bool = True


class MarketDataConfig(BaseModel):
    ohlcv_limit: int = 20
    l2_depth: int = 20


class AssetsConfig(BaseModel):
    monitor_list_path: str = "configs/assets.txt"


class SystemConfig(BaseModel):
    mode: Literal["DRY_RUN", "LIVE"] = "DRY_RUN"
    log_level: str = "INFO"


class AgenticAITAConfig(BaseModel):
    system: SystemConfig = SystemConfig()
    azte: AZTEConfig = AZTEConfig()
    igp: IGPConfig = IGPConfig()
    risk_manager: RiskManagerConfig = RiskManagerConfig()
    cbd: CBDConfig = CBDConfig()
    ollama: OllamaConfig = OllamaConfig()
    database: DatabaseConfig = DatabaseConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    market_data: MarketDataConfig = MarketDataConfig()
    assets: AssetsConfig = AssetsConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "AgenticAITAConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def __repr__(self) -> str:
        return f"AgenticAITAConfig(mode={self.system.mode}, model={self.ollama.model})"
