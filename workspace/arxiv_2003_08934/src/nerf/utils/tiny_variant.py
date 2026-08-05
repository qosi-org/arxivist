"""
utils/tiny_variant.py — Optional TinyNeRF-inspired consumer-hardware preset.

This module implements SIR implementation_assumptions[3] (confidence 0.5):
an explicitly swappable, opt-in configuration override inspired by the
official TinyNeRF demo notebook (bmild/nerf: tiny_nerf.ipynb), NOT a
modification of the paper's own reported architecture or results. It is only
applied when the user passes `--tiny` to train.py, and always layers on top
of (never silently replaces) whatever base config was loaded.

Per architecture_plan.json risk_assessment: TinyNeRF's exact published
hyperparameters could not be verified purely from the NeRF paper text; the
overrides below mirror configs/tiny_consumer.yaml and are documented as
`# ASSUMED` there.
"""

from __future__ import annotations

import copy
from typing import Any


class TinyNeRFPreset:
    """Applies TinyNeRF-style consumer-hardware overrides to a base config dict."""

    #: Overrides mirror configs/tiny_consumer.yaml; kept here in code so
    #: `--tiny` works even against a config file that isn't tiny_consumer.yaml
    #: (e.g. `--config configs/config.yaml --tiny`).
    _OVERRIDES: dict[str, dict[str, Any]] = {
        "model": {
            "pos_enc_freqs_x": 6,
            "trunk_depth": 4,           # ASSUMED: single shallow MLP, TinyNeRF-style
            "trunk_width": 128,
            "skip_layers": [2],         # must stay < trunk_depth; re-centered for the shrunk 4-layer trunk
            "color_hidden_width": 64,
            "num_coarse_samples": 64,
            "num_fine_samples": 0,
            "use_hierarchical_sampling": False,
        },
        "model_variant": {"name": "no_hierarchical_sampling"},
        "training": {
            "ray_batch_size": 1024,
            "num_training_steps": 5000,
            "checkpoint_every": 500,
            "log_every": 25,
            "eval_every": 500,
        },
        "data": {"half_res": True},
        "hardware": {"tiny_hardware_mode": True},
    }

    def apply(self, base_config: dict) -> dict:
        """Return a new config dict with TinyNeRF-style overrides applied.

        Args:
            base_config: a validated config dict (`NeRFConfig.raw`).

        Returns:
            A new dict (the input is not mutated) with the tiny-hardware
            overrides merged in per top-level section.
        """
        merged = copy.deepcopy(base_config)
        for section, values in self._OVERRIDES.items():
            merged.setdefault(section, {})
            merged[section].update(values)
        return merged

    def __repr__(self) -> str:  # noqa: D105
        return "TinyNeRFPreset()"
