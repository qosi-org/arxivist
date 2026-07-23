"""
Per-stock parameter sampling (Section 2.3) and counterfactual scenario
overrides (Section 4.3, Table 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StockConfig:
    """One independently-parameterised stock's configuration (Section 2.3)."""

    stock_id: int
    seed: int
    initial_price: float
    n_hft: int
    n_retail: int
    pareto_xi: float
    # Scenario overrides (all default to baseline / disabled):
    hft_active: bool = True
    splitting_rule: str | None = None  # None -> use model config default (dirichlet)
    price_limits_enabled: bool = False
    momentum_enabled: bool = False
    force_single_child: bool = False

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return {k: v for k, v in d.items() if v is not None}


class StockParameterSampler:
    """Draws the random per-stock configuration used to build the 2000-stock
    (or 20-stock, for counterfactuals) cross-sectional dataset (Section 2.3).
    """

    def __init__(self, model_cfg: dict, sim_cfg: dict):
        self.model_cfg = model_cfg
        self.sim_cfg = sim_cfg

    def sample(self, n_stocks: int, seed: int) -> list[StockConfig]:
        """Sample n_stocks independent StockConfigs.

        Args:
            n_stocks: Number of stocks to generate.
            seed: Seed for the sampler's own RNG (each stock also gets its
                own independent downstream seed, per the paper: "Stocks are
                independent (separate random seeds)").

        Returns:
            List of StockConfig, one per stock.
        """
        rng = np.random.default_rng(seed)
        sim = self.sim_cfg
        m = self.model_cfg

        configs = []
        for i in range(n_stocks):
            log_p0 = rng.uniform(
                np.log10(sim["initial_price_log_min"]),
                np.log10(sim["initial_price_log_max"]),
            )
            configs.append(
                StockConfig(
                    stock_id=i,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    initial_price=float(10**log_p0),
                    n_hft=int(rng.integers(m["n_hft_min"], m["n_hft_max"] + 1)),
                    n_retail=int(rng.integers(m["n_retail_min"], m["n_retail_max"] + 1)),
                    pareto_xi=float(rng.uniform(
                        m["institutional"]["pareto_xi_min"],
                        m["institutional"]["pareto_xi_max"],
                    )),
                )
            )
        return configs


# Scenario name -> StockConfig field overrides (Section 4.3, Table 1).
# ASSUMED numeric parameters for price_limits / momentum / low_liquidity are
# read from config.yaml (model.price_limits.band_pct, model.momentum.*);
# see SIR ambiguities[2] (confidence 0.4).
SCENARIO_OVERRIDES: dict[str, dict] = {
    "baseline": {},
    "no_splitting": {"splitting_rule": "uniform", "force_single_child": True},
    "no_hft": {"hft_active": False},
    "price_limits": {"price_limits_enabled": True},
    "low_liquidity": {"_reduce_retail_fraction": 0.5},
    "momentum": {"momentum_enabled": True},
    "uniform_split": {"splitting_rule": "uniform"},
    "front_loaded": {"splitting_rule": "front_loaded"},
}


def apply_scenario(stock_config: StockConfig, scenario: str) -> StockConfig:
    """Return a copy of stock_config with the named scenario's overrides applied.

    "no_splitting" is special-cased in the simulation builder (a single
    unsplit child order per metaorder) since it is a structural change, not
    just a parameter override -- see market.py.
    """
    if scenario not in SCENARIO_OVERRIDES:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Must be one of {list(SCENARIO_OVERRIDES)}"
        )
    overrides = SCENARIO_OVERRIDES[scenario]
    import copy

    new_cfg = copy.deepcopy(stock_config)
    if "hft_active" in overrides:
        new_cfg.hft_active = overrides["hft_active"]
    if "splitting_rule" in overrides:
        new_cfg.splitting_rule = overrides["splitting_rule"]
    if "price_limits_enabled" in overrides:
        new_cfg.price_limits_enabled = overrides["price_limits_enabled"]
    if "momentum_enabled" in overrides:
        new_cfg.momentum_enabled = overrides["momentum_enabled"]
    if "force_single_child" in overrides:
        new_cfg.force_single_child = overrides["force_single_child"]
    if "_reduce_retail_fraction" in overrides:
        new_cfg.n_retail = max(1, int(new_cfg.n_retail * overrides["_reduce_retail_fraction"]))
    return new_cfg
