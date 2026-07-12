"""
Batch simulation runner (Sections 2.3-2.4, 4.3).

Repurposes the ArXivist template's "trainer" slot: instead of a gradient
training loop, this iterates over N independently-parameterised stocks,
running each StockMarketSimulation to completion (embarrassingly parallel
across stocks, as noted in Section 2.4: "each stock is embarrassingly
parallel").
"""

from __future__ import annotations

from dataclasses import dataclass

from joblib import Parallel, delayed
from tqdm import tqdm

from sqrt_law_abm.data.dataset import StockConfig, apply_scenario
from sqrt_law_abm.models.market import SimulationResult, StockMarketSimulation


def _run_single_stock(
    stock_config: StockConfig, model_cfg: dict, sim_cfg: dict, n_steps: int, warmup_steps: int
) -> SimulationResult:
    sim = StockMarketSimulation(stock_config.to_dict(), model_cfg, sim_cfg)
    return sim.run(n_steps=n_steps, warmup_steps=warmup_steps)


class BatchSimulationRunner:
    """Runs StockMarketSimulation across many stocks, optionally in parallel.

    Args:
        model_cfg: The `model:` section of the global config.
        sim_cfg: The `simulation:` section of the global config.
    """

    def __init__(self, model_cfg: dict, sim_cfg: dict):
        self.model_cfg = model_cfg
        self.sim_cfg = sim_cfg

    def run_all_stocks(
        self,
        stock_configs: list[StockConfig],
        scenario: str = "baseline",
        n_jobs: int = -1,
        n_steps: int | None = None,
        warmup_steps: int | None = None,
    ) -> list[SimulationResult]:
        """Run all given stocks under a single named scenario.

        Args:
            stock_configs: Baseline per-stock configs (from StockParameterSampler).
            scenario: One of the keys in dataset.SCENARIO_OVERRIDES.
            n_jobs: Parallel workers (-1 = all cores, 1 = sequential, useful
                for debugging/small runs).
            n_steps: Override simulation.n_steps (used by --debug).
            warmup_steps: Override simulation.warmup_steps (used by --debug).

        Returns:
            One SimulationResult per stock, in the same order as stock_configs.
        """
        n_steps = n_steps if n_steps is not None else self.sim_cfg["n_steps"]
        warmup_steps = warmup_steps if warmup_steps is not None else self.sim_cfg["warmup_steps"]

        scenario_configs = [apply_scenario(sc, scenario) for sc in stock_configs]

        if n_jobs == 1:
            return [
                _run_single_stock(sc, self.model_cfg, self.sim_cfg, n_steps, warmup_steps)
                for sc in tqdm(scenario_configs, desc=f"scenario={scenario}")
            ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_stock)(sc, self.model_cfg, self.sim_cfg, n_steps, warmup_steps)
            for sc in tqdm(scenario_configs, desc=f"scenario={scenario}")
        )
        return results

    def run_counterfactual_suite(
        self,
        stock_configs: list[StockConfig],
        scenarios: list[str],
        n_jobs: int = -1,
        n_steps: int | None = None,
        warmup_steps: int | None = None,
    ) -> dict[str, list[SimulationResult]]:
        """Run the full counterfactual ablation suite (Section 4.3, Table 1).

        Args:
            stock_configs: The (typically 20) stocks used for all scenarios.
            scenarios: List of scenario names to run.
            n_jobs: Parallel workers per scenario.
            n_steps: Override simulation.n_steps (used by --debug).
            warmup_steps: Override simulation.warmup_steps (used by --debug).

        Returns:
            Dict mapping scenario name -> list of SimulationResult.
        """
        return {
            scenario: self.run_all_stocks(
                stock_configs, scenario, n_jobs=n_jobs, n_steps=n_steps, warmup_steps=warmup_steps
            )
            for scenario in scenarios
        }
