"""
utils/config.py — Configuration loading, validation, seed utilities.

Implements config management for the QSVM fraud detection pipeline.
All hyperparameters are loaded from YAML; no values are hardcoded elsewhere.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


logger = logging.getLogger(__name__)


class Config:
    """
    Loads and validates YAML configuration for the QSVM fraud detection pipeline.

    Args:
        path: Path to the YAML config file.

    Example:
        >>> cfg = Config.load("configs/config.yaml")
        >>> cfg["model"]["n_qubits"]
        10
    """

    REQUIRED_KEYS = {
        "model": ["n_qubits", "backend"],
        "data": ["csv_path", "n_features"],
        "evaluation": ["metrics"],
        "hardware": ["random_seed"],
    }

    VALID_BACKENDS = {"GPU", "qasm_simulator", "aer_simulator"}
    VALID_SCORE_FUNCS = {"f_classif", "chi2", "mutual_info_classif"}
    VALID_ENTANGLEMENTS = {"full", "linear", "circular", "sca"}

    @classmethod
    def load(cls, path: str) -> dict[str, Any]:
        """
        Load and validate config from YAML file.

        Args:
            path: Path to config YAML.

        Returns:
            Validated config dictionary.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If required keys are missing or values are invalid.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        cls._validate(config)
        logger.info("Config loaded from %s", path)
        return config

    @classmethod
    def _validate(cls, config: dict[str, Any]) -> None:
        """Validate required keys and value ranges."""
        for section, keys in cls.REQUIRED_KEYS.items():
            if section not in config:
                raise ValueError(f"Config missing required section: '{section}'")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(
                        f"Config missing required key: '{section}.{key}'"
                    )

        n_qubits = config["model"]["n_qubits"]
        if n_qubits not in (4, 8, 10):
            raise ValueError(
                f"model.n_qubits must be 4, 8, or 10 (paper ablation values); got {n_qubits}"
            )

        backend = config["model"]["backend"]
        if backend not in cls.VALID_BACKENDS:
            raise ValueError(
                f"model.backend '{backend}' not in supported backends: {cls.VALID_BACKENDS}"
            )

        n_features = config["data"]["n_features"]
        if n_features != config["model"]["n_qubits"]:
            raise ValueError(
                f"data.n_features ({n_features}) must equal model.n_qubits ({n_qubits}) "
                f"— in QSVM, qubit count equals feature dimension."
            )

        score_func = config["data"].get("score_func", "f_classif")
        if score_func not in cls.VALID_SCORE_FUNCS:
            raise ValueError(
                f"data.score_func '{score_func}' not in {cls.VALID_SCORE_FUNCS}"
            )

        entanglement = config["model"].get("entanglement", "full")
        if entanglement not in cls.VALID_ENTANGLEMENTS:
            raise ValueError(
                f"model.entanglement '{entanglement}' not in {cls.VALID_ENTANGLEMENTS}"
            )

        C = config["model"].get("C", 1.0)
        if C <= 0:
            raise ValueError(f"model.C must be > 0; got {C}")

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set all random seeds for reproducibility.

        Seeds: Python stdlib random, NumPy. Qiskit uses its own RNG seeded
        via algorithm_globals (set in relevant modules).

        Args:
            seed: Integer random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        logger.debug("Random seed set to %d", seed)

    @staticmethod
    def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
        """Configure root logger with console + optional file handler."""
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
