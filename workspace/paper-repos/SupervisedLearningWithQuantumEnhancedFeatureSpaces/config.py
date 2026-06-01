"""
config.py — Configuration management for the QSVM reproduction.

Loads configs/default.yaml (or user-specified YAML) into typed dataclasses.
All paths and hyperparameters flow through Config; nothing is hardcoded elsewhere.

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FeatureMapConfig:
    n_qubits: int = 2
    reps: int = 2


@dataclass
class DataConfig:
    n_per_label: int = 20
    gap: float = 0.3
    domain_min: float = 0.0001
    domain_max: float = 6.2832
    n_test_sets: int = 10
    V_seed: int = 42


@dataclass
class SPSAConfig:
    n_iter: int = 250
    shots_cost: int = 200
    shots_eval: int = 2000
    shots_classify: int = 10000
    # ASSUMED: Spall canonical defaults — conf=0.55
    a: float = 0.628
    c: float = 0.1
    A: float = 100.0
    alpha_spsa: float = 0.602
    gamma_spsa: float = 0.101


@dataclass
class QVCConfig:
    depths: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_datasets: int = 3
    spsa: SPSAConfig = field(default_factory=SPSAConfig)
    optimizer: str = "spsa"
    bias_init: float = 0.0
    bias_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class SVMConfig:
    C: float = 1.0
    psd_epsilon: float = 1e-10


@dataclass
class QKEConfig:
    shots_per_entry: int = 50000
    use_statevector: bool = True
    n_test_sets: int = 10
    svm: SVMConfig = field(default_factory=SVMConfig)


@dataclass
class ErrorMitigationConfig:
    enabled: bool = False
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 1.5])
    depolarizing_error_rate: float = 0.01


@dataclass
class BackendConfig:
    name: str = "statevector_simulator"
    device: str = "cpu"
    noise_model: Optional[str] = None


@dataclass
class OutputConfig:
    results_dir: str = "results/"
    save_kernel_matrix: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150
    verbose: bool = True


@dataclass
class ExperimentConfig:
    name: str = "havlicek2018_reproduction"
    protocol: str = "both"
    output_dir: str = "results/"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Root configuration for the Havlicek 2018 QSVM reproduction.

    All fields default to paper-reported values (or clearly marked ASSUMED
    values where the paper does not specify).
    """
    seed: int = 42
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    feature_map: FeatureMapConfig = field(default_factory=FeatureMapConfig)
    data: DataConfig = field(default_factory=DataConfig)
    qvc: QVCConfig = field(default_factory=QVCConfig)
    qke: QKEConfig = field(default_factory=QKEConfig)
    error_mitigation: ErrorMitigationConfig = field(default_factory=ErrorMitigationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate configuration values at load time."""
        if self.feature_map.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.feature_map.reps != 2:
            raise ValueError("reps must be 2 (paper uses double U_Phi structure)")
        if not (0.0 < self.data.gap <= 1.0):
            raise ValueError(f"gap must be in (0,1], got {self.data.gap}")
        if self.qvc.optimizer not in ("spsa", "adam", "cobyla"):
            raise ValueError(f"optimizer must be 'spsa'|'adam'|'cobyla', got {self.qvc.optimizer}")
        if self.experiment.protocol not in ("qvc", "qke", "both"):
            raise ValueError(f"protocol must be 'qvc'|'qke'|'both', got {self.experiment.protocol}")

    def __repr__(self) -> str:
        return (
            f"Config(seed={self.seed}, protocol={self.experiment.protocol}, "
            f"n_qubits={self.feature_map.n_qubits}, "
            f"qvc_depths={self.qvc.depths}, "
            f"qke_statevector={self.qke.use_statevector})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> Config:
    """
    Load a YAML config file and return a validated Config object.

    Parameters
    ----------
    path : str or Path
        Path to a YAML config file.

    Returns
    -------
    Config
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    # Unpack nested dicts into dataclasses
    if "seed" in raw:
        cfg.seed = int(raw["seed"])
    if "experiment" in raw:
        e = raw["experiment"]
        cfg.experiment = ExperimentConfig(
            name=e.get("name", cfg.experiment.name),
            protocol=e.get("protocol", cfg.experiment.protocol),
            output_dir=e.get("output_dir", cfg.experiment.output_dir),
        )
    if "feature_map" in raw:
        fm = raw["feature_map"]
        cfg.feature_map = FeatureMapConfig(
            n_qubits=int(fm.get("n_qubits", cfg.feature_map.n_qubits)),
            reps=int(fm.get("reps", cfg.feature_map.reps)),
        )
    if "data" in raw:
        d = raw["data"]
        cfg.data = DataConfig(
            n_per_label=int(d.get("n_per_label", cfg.data.n_per_label)),
            gap=float(d.get("gap", cfg.data.gap)),
            domain_min=float(d.get("domain_min", cfg.data.domain_min)),
            domain_max=float(d.get("domain_max", cfg.data.domain_max)),
            n_test_sets=int(d.get("n_test_sets", cfg.data.n_test_sets)),
            V_seed=int(d.get("V_seed", cfg.data.V_seed)),
        )
    if "qvc" in raw:
        q = raw["qvc"]
        sp = q.get("spsa", {})
        cfg.qvc = QVCConfig(
            depths=list(q.get("depths", cfg.qvc.depths)),
            n_datasets=int(q.get("n_datasets", cfg.qvc.n_datasets)),
            optimizer=q.get("optimizer", cfg.qvc.optimizer),
            bias_init=float(q.get("bias_init", cfg.qvc.bias_init)),
            bias_range=list(q.get("bias_range", cfg.qvc.bias_range)),
            spsa=SPSAConfig(
                n_iter=int(sp.get("n_iter", 250)),
                shots_cost=int(sp.get("shots_cost", 200)),
                shots_eval=int(sp.get("shots_eval", 2000)),
                shots_classify=int(sp.get("shots_classify", 10000)),
                a=float(sp.get("a", 0.628)),
                c=float(sp.get("c", 0.1)),
                A=float(sp.get("A", 100.0)),
                alpha_spsa=float(sp.get("alpha_spsa", 0.602)),
                gamma_spsa=float(sp.get("gamma_spsa", 0.101)),
            ),
        )
    if "qke" in raw:
        q = raw["qke"]
        sv = q.get("svm", {})
        cfg.qke = QKEConfig(
            shots_per_entry=int(q.get("shots_per_entry", 50000)),
            use_statevector=bool(q.get("use_statevector", True)),
            n_test_sets=int(q.get("n_test_sets", 10)),
            svm=SVMConfig(
                C=float(sv.get("C", 1.0)),
                psd_epsilon=float(sv.get("psd_epsilon", 1e-10)),
            ),
        )
    if "error_mitigation" in raw:
        em = raw["error_mitigation"]
        cfg.error_mitigation = ErrorMitigationConfig(
            enabled=bool(em.get("enabled", False)),
            scale_factors=list(em.get("scale_factors", [1.0, 1.5])),
            depolarizing_error_rate=float(em.get("depolarizing_error_rate", 0.01)),
        )
    if "backend" in raw:
        b = raw["backend"]
        cfg.backend = BackendConfig(
            name=b.get("name", "statevector_simulator"),
            device=b.get("device", "cpu"),
            noise_model=b.get("noise_model", None),
        )
    if "output" in raw:
        o = raw["output"]
        cfg.output = OutputConfig(
            results_dir=o.get("results_dir", "results/"),
            save_kernel_matrix=bool(o.get("save_kernel_matrix", True)),
            save_plots=bool(o.get("save_plots", True)),
            plot_format=o.get("plot_format", "png"),
            plot_dpi=int(o.get("plot_dpi", 150)),
            verbose=bool(o.get("verbose", True)),
        )

    cfg.validate()
    return cfg


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and any relevant libraries.

    Parameters
    ----------
    seed : int
        Master random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Qiskit uses numpy internally; no separate seed needed
