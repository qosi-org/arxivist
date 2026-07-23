"""
Configuration loading and validation utilities.

Loads YAML config files into typed dataclasses.
All hyperparameters flow through here; nothing is hardcoded elsewhere.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# CatNat core config
# ---------------------------------------------------------------------------

@dataclass
class CatNatConfig:
    """Configuration for the catnat parameterization.

    Paper: Section 4.2, Eq. 12.
    """
    K: int = 8
    activation: str = "natural"          # "natural" | "sigmoid"
    natural_activation_C: float = 0.0
    natural_activation_A: float = 2 * math.pi

    def __post_init__(self) -> None:
        if self.activation not in ("natural", "sigmoid"):
            raise ValueError(f"activation must be 'natural' or 'sigmoid', got '{self.activation}'")
        if self.K < 2:
            raise ValueError(f"K must be >= 2, got {self.K}")
        if not (self.K & (self.K - 1) == 0):
            raise ValueError(
                f"K={self.K} is not a power of 2. "
                "catnat requires K to be a power of 2 (see RISK-01 in architecture_plan.json). "
                "Consider setting K to the next power of 2 and using pad_to_power_of_2=True."
            )


@dataclass
class HardwareConfig:
    device: str = "cuda"
    deterministic: bool = False
    num_workers: int = 4
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# GSL experiment config
# ---------------------------------------------------------------------------

@dataclass
class GSLModelConfig:
    n_nodes: int = 15
    n_communities: int = 4
    d_in: int = 1
    d_hidden: int = 64
    d_out: int = 1
    n_gcn_layers: int = 2


@dataclass
class GSLDataConfig:
    n_samples: int = 10000
    sigma_x: float = 1.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    theta_star_values: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    theta_star: float = 0.5
    data_dir: str = "data/gsl/"
    seed: int = 42


@dataclass
class GSLTrainingConfig:
    optimizer: str = "adam"
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 40
    M_samples: int = 32
    gradient_estimator: str = "reinforce_loo"
    score_init_low: float = 0.0
    score_init_high: float = 0.1
    n_seeds: int = 10
    lr: float = 0.022
    lr_coarse_grid: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    )
    lr_refined_natural: List[float] = field(
        default_factory=lambda: [0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.041]
    )
    lr_refined_sigmoid: List[float] = field(
        default_factory=lambda: [0.025, 0.03, 0.037, 0.045, 0.055, 0.067, 0.082]
    )
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 5


@dataclass
class GSLOutputConfig:
    results_dir: str = "results/gsl/"
    checkpoint_dir: str = "checkpoints/gsl/"


@dataclass
class GSLConfig:
    experiment: str = "gsl"
    parameterization: str = "natural"
    catnat: CatNatConfig = field(default_factory=lambda: CatNatConfig(K=2))
    model: GSLModelConfig = field(default_factory=GSLModelConfig)
    data: GSLDataConfig = field(default_factory=GSLDataConfig)
    training: GSLTrainingConfig = field(default_factory=GSLTrainingConfig)
    output: GSLOutputConfig = field(default_factory=GSLOutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# ---------------------------------------------------------------------------
# VAE experiment config
# ---------------------------------------------------------------------------

@dataclass
class VAEModelConfig:
    N: int = 20
    K: int = 16
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    encoder_kernel_size: int = 4
    encoder_stride: int = 2
    encoder_fc_hidden: int = 512
    decoder_fc_hidden: int = 512


@dataclass
class VAEDataConfig:
    dataset: str = "mnist"
    binarize: bool = False
    data_dir: str = "data/mnist/"
    batch_size: int = 128


@dataclass
class VAETrainingConfig:
    optimizer: str = "adam"
    n_seeds: int = 5
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.5
    gumbel_tau_anneal_rate: float = 3e-5
    gumbel_injection_point: str = "leaf_logprobs"   # WARNING: RISK-02
    straight_through: bool = True
    lr: float = 0.01
    lr_coarse_grid: List[float] = field(
        default_factory=lambda: [0.0003, 0.001, 0.003, 0.01, 0.03]
    )
    lr_refined_grid: List[float] = field(
        default_factory=lambda: [0.0045, 0.0056, 0.0069, 0.0085, 0.01, 0.013, 0.016, 0.02]
    )
    eval_importance_samples: int = 512
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10


@dataclass
class VAEOutputConfig:
    results_dir: str = "results/vae/"
    checkpoint_dir: str = "checkpoints/vae/"


@dataclass
class VAEConfig:
    experiment: str = "vae"
    parameterization: str = "natural"
    catnat: CatNatConfig = field(default_factory=lambda: CatNatConfig(K=16))
    model: VAEModelConfig = field(default_factory=VAEModelConfig)
    data: VAEDataConfig = field(default_factory=VAEDataConfig)
    training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    output: VAEOutputConfig = field(default_factory=VAEOutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# ---------------------------------------------------------------------------
# RL experiment config
# ---------------------------------------------------------------------------

@dataclass
class RLEnvConfig:
    name: str = "BreakoutNoFrameskip-v4"
    frame_stack: int = 4
    frame_size: int = 84
    reward_clip: float = 1.0
    grayscale: bool = True


@dataclass
class RLModelConfig:
    conv1_filters: int = 32
    conv1_kernel: int = 8
    conv1_stride: int = 4
    conv2_filters: int = 64
    conv2_kernel: int = 4
    conv2_stride: int = 2
    conv3_filters: int = 64
    conv3_kernel: int = 3
    conv3_stride: int = 1
    fc_hidden: int = 512
    init: str = "orthogonal"


@dataclass
class RLTrainingConfig:
    algorithm: str = "ppo"
    optimizer: str = "adam"
    adam_epsilon: float = 1e-5
    total_timesteps: int = 8_000_000
    lr_schedule: str = "linear_anneal"
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_envs: int = 8
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    n_seeds: int = 10
    hparam_search_trials: int = 160
    hparam_top_k: int = 10
    log_every_n_steps: int = 1000
    save_every_n_steps: int = 100_000


@dataclass
class RLOutputConfig:
    results_dir: str = "results/rl/"
    checkpoint_dir: str = "checkpoints/rl/"
    hparam_db: str = "results/rl/hparam_search.db"


@dataclass
class RLConfig:
    experiment: str = "rl"
    parameterization: str = "natural"
    catnat: CatNatConfig = field(default_factory=lambda: CatNatConfig(K=4))
    env: RLEnvConfig = field(default_factory=RLEnvConfig)
    model: RLModelConfig = field(default_factory=RLModelConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    output: RLOutputConfig = field(default_factory=RLOutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load a YAML config file and return the appropriate typed config object.

    Args:
        path: Path to a YAML config file.

    Returns:
        One of GSLConfig, VAEConfig, or RLConfig depending on the 'experiment' key.

    Raises:
        ValueError: If the 'experiment' key is missing or unrecognised.
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    experiment = raw.get("experiment")
    if experiment is None:
        raise ValueError("Config file must contain an 'experiment' key.")

    if experiment == "gsl":
        return _build_gsl_config(raw)
    elif experiment == "vae":
        return _build_vae_config(raw)
    elif experiment == "rl":
        return _build_rl_config(raw)
    else:
        raise ValueError(f"Unknown experiment '{experiment}'. Must be one of: gsl, vae, rl.")


def _build_gsl_config(raw: dict) -> GSLConfig:
    catnat_raw = raw.get("catnat", {})
    return GSLConfig(
        experiment=raw.get("experiment", "gsl"),
        parameterization=raw.get("parameterization", "natural"),
        catnat=CatNatConfig(**catnat_raw),
        model=GSLModelConfig(**raw.get("model", {})),
        data=GSLDataConfig(**raw.get("data", {})),
        training=GSLTrainingConfig(**raw.get("training", {})),
        output=GSLOutputConfig(**raw.get("output", {})),
        hardware=HardwareConfig(**raw.get("hardware", {})),
    )


def _build_vae_config(raw: dict) -> VAEConfig:
    catnat_raw = raw.get("catnat", {})
    return VAEConfig(
        experiment=raw.get("experiment", "vae"),
        parameterization=raw.get("parameterization", "natural"),
        catnat=CatNatConfig(**catnat_raw),
        model=VAEModelConfig(**raw.get("model", {})),
        data=VAEDataConfig(**raw.get("data", {})),
        training=VAETrainingConfig(**raw.get("training", {})),
        output=VAEOutputConfig(**raw.get("output", {})),
        hardware=HardwareConfig(**raw.get("hardware", {})),
    )


def _build_rl_config(raw: dict) -> RLConfig:
    catnat_raw = raw.get("catnat", {})
    return RLConfig(
        experiment=raw.get("experiment", "rl"),
        parameterization=raw.get("parameterization", "natural"),
        catnat=CatNatConfig(**catnat_raw),
        env=RLEnvConfig(**raw.get("env", {})),
        model=RLModelConfig(**raw.get("model", {})),
        training=RLTrainingConfig(**raw.get("training", {})),
        output=RLOutputConfig(**raw.get("output", {})),
        hardware=HardwareConfig(**raw.get("hardware", {})),
    )
