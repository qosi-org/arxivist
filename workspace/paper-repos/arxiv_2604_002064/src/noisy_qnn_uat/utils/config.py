"""Configuration loading and reproducibility utilities.

Implements: Architecture Plan module `utils/config.py`.
Paper section: Section 4.1 (experimental setup); reproducibility requirements are
an ArXivist Code Generator standard, not paper-specific.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

_REQUIRED_TOP_LEVEL_KEYS = ("model", "training", "data", "evaluation", "hardware")


@dataclass
class ConfigLoader:
    """Loads and validates the YAML configuration files used throughout this repo.

    Args:
        strict: if True, raise ValueError on missing required top-level sections.
    """

    strict: bool = True

    def load(self, path: str) -> dict[str, Any]:
        """Load a config YAML file and perform basic structural validation.

        Args:
            path: filesystem path to a config YAML file (e.g. configs/config.yaml).

        Returns:
            The parsed configuration as a nested dict.

        Raises:
            ValueError: if a required top-level section is missing and `strict` is True.
            FileNotFoundError: if `path` does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Config file at '{path}' is empty or invalid YAML.")

        missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in config]
        if missing and self.strict:
            raise ValueError(
                f"Config file '{path}' is missing required top-level section(s): {missing}. "
                f"Expected all of: {_REQUIRED_TOP_LEVEL_KEYS}."
            )

        self._validate_model_section(config.get("model", {}), path)
        return config

    @staticmethod
    def _validate_model_section(model_cfg: dict[str, Any], path: str) -> None:
        """Sanity-check that n_qubits is consistent with n_accuracy_blocks and n0.

        Per SIR Statement 2.1 / Section 4.1: qubit count n satisfies
        n = ceil(log2(4*n_accuracy_blocks + n0)).
        """
        import math

        n_blocks = model_cfg.get("n_accuracy_blocks")
        n_qubits = model_cfg.get("n_qubits")
        n0 = model_cfg.get("n0", 0)
        if n_blocks is None or n_qubits is None:
            return  # nothing to validate yet (e.g. partial/template config)

        expected_n_qubits = math.ceil(math.log2(4 * n_blocks + n0))
        if expected_n_qubits != n_qubits:
            raise ValueError(
                f"Config '{path}': n_qubits={n_qubits} is inconsistent with "
                f"n_accuracy_blocks={n_blocks}, n0={n0}. Expected "
                f"ceil(log2(4*{n_blocks}+{n0})) = {expected_n_qubits}. "
                "This mismatch would silently invalidate all reported error bounds "
                "(see architecture_plan.json risk_assessment, 'n0 padding value')."
            )

    def __repr__(self) -> str:
        return f"ConfigLoader(strict={self.strict})"


@dataclass
class SeedManager:
    """Seeds Python, NumPy, PyTorch (and notes on Qiskit simulator RNG) for reproducibility.

    NOTE: a fixed seed is an ASSUMED reproducibility practice (SIR
    implementation_assumptions[3], confidence 0.3) -- the paper does not state
    whether / which seed was used.
    """

    def seed_everything(self, seed: int, deterministic: bool = False) -> None:
        """Seed all RNGs used across this repository.

        Args:
            seed: the random seed to apply everywhere.
            deterministic: if True, also request deterministic (possibly slower)
                behaviour from PyTorch. Qiskit's AerSimulator seed must be passed
                explicitly per-run via `seed_simulator=seed` (see
                `models/qnn_circuit.py::QNNCircuitBuilder`); it is not a global RNG.
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
            # torch is an optional dependency for Methods A/B (scipy-only); skip if absent.
            pass

    def __repr__(self) -> str:
        return "SeedManager()"
