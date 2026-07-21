"""
Migration-race model: Mosca's inequality as a decision rule across
start-time/adoption scenarios.

Implements Section 7.4 and Figure 4 of arXiv:2606.14484. Mosca's inequality
(Mosca, 2018) states that assets are at risk whenever:

    time_to_start_migrating + time_to_migrate > time_until_CRQC_arrives

The paper sweeps three start-time scenarios (prompt 2026, delayed 2030,
severe-delay 2033) against three CRQC arrival estimates (aggressive 2035,
survey-median 2040, conservative 2050), finding that only a severely delayed
start against an early machine is at risk.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


class MigrationRaceModel:
    """Implements Mosca's inequality as a migration-race decision rule."""

    def __repr__(self) -> str:  # noqa: D105
        return "MigrationRaceModel()"

    def is_at_risk(self, start_year: float, migration_duration_years: float, crqc_arrival_year: float) -> bool:
        """Mosca's inequality: at risk iff (start_year + duration) > crqc_arrival_year.

        Args:
            start_year: year migration begins.
            migration_duration_years: years required to complete migration.
            crqc_arrival_year: assumed CRQC arrival year.

        Returns:
            True if the asset is at risk under this scenario.
        """
        return (start_year + migration_duration_years) > crqc_arrival_year

    def run_scenarios(
        self, scenarios: Dict[str, Dict], crqc_estimates: Dict[str, float]
    ) -> pd.DataFrame:
        """Run all (start-time scenario) x (CRQC estimate) combinations,
        reproducing Figure 4.

        Args:
            scenarios: dict of scenario_name -> {'start_year': float,
                'migration_duration_years': float}. Migration duration is
                derived from the paper's reported finish years
                (finish_year - start_year) if not given directly.
            crqc_estimates: dict of estimate_name -> crqc_arrival_year
                (paper: {'aggressive': 2035, 'survey_median': 2040,
                'conservative': 2050}).

        Returns:
            DataFrame with columns: scenario, start_year, migration_duration_years,
            finish_year, crqc_estimate_name, crqc_arrival_year, at_risk.
        """
        rows = []
        for scenario_name, scenario in scenarios.items():
            start_year = scenario["start_year"]
            duration = scenario["migration_duration_years"]
            finish_year = start_year + duration

            for crqc_name, crqc_year in crqc_estimates.items():
                rows.append(
                    {
                        "scenario": scenario_name,
                        "start_year": start_year,
                        "migration_duration_years": duration,
                        "finish_year": finish_year,
                        "crqc_estimate_name": crqc_name,
                        "crqc_arrival_year": crqc_year,
                        "at_risk": self.is_at_risk(start_year, duration, crqc_year),
                    }
                )
        return pd.DataFrame(rows)
