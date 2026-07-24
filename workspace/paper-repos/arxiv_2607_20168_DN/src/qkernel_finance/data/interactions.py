"""
Twelve literature-catalogued pairwise interaction characteristics (Sec 3.3, Table 2).

Each is the daily cross-sectional product of its components' z-scores,
re-winsorized and re-standardized. Signs are not imposed from the source
literature; the window-level IC screen orients them like base characteristics.
"""
from __future__ import annotations

import pandas as pd

# (interaction_name, component_a, component_b, source_citation)
INTERACTION_CATALOG = [
    ("value_x_momentum", "book_to_price", "mom_12_1", "Asness (1997)"),
    ("value_x_profitability", "book_to_price", "roa", "Piotroski (2000)"),
    ("value_x_gross_margin", "book_to_price", "gross_margin", "Novy-Marx (2013)"),
    ("gross_profitability", "gross_margin", "asset_turnover", "Novy-Marx (2013)"),  # uses 'turnover' proxy
    ("beta_x_lottery", "market_beta", "lottery_max", "Bali et al. (2017)"),
    ("momentum_x_revisions", "mom_12_1", "analyst_revisions", "Chan et al. (1996)"),
    ("momentum_x_surprise", "mom_12_1", "sue", "Chan et al. (1996)"),
    ("lottery_x_reversal", "lottery_max", "reversal_1m", "Nartea et al. (2017); Bali et al. (2011)"),
    ("momentum_x_uncertainty", "mom_12_1", "eps_vol", "Zhang (2006)"),
    ("value_x_sales_growth", "book_to_price", "sales_growth", "Lakonishok et al. (1994)"),
    ("surprise_x_uncertainty", "sue", "eps_vol", "Francis et al. (2007)"),
    ("value_x_cashflow_quality", "book_to_price", "cfo_to_ni", "Piotroski (2000); Sloan (1996)"),
]


class InteractionCatalog:
    """Builds the twelve interaction characteristics from an already-standardized panel."""

    def build(self, standardized_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            standardized_panel: DataFrame with a `date` column and one
                (already cross-sectionally standardized) column per base
                characteristic, including any needed by INTERACTION_CATALOG
                (some -- analyst_revisions, eps_vol -- are only available in
                the 31-characteristic diagnostic-study set, Sec 3.1).

        Returns:
            DataFrame with `date` plus one column per interaction, each
            re-winsorized (+/-3 sigma) and re-standardized cross-sectionally.
        """
        from qkernel_finance.data.characteristics import CharacteristicBuilder

        out = standardized_panel[["date"]].copy()
        missing = []
        for name, comp_a, comp_b, _source in INTERACTION_CATALOG:
            if comp_a not in standardized_panel.columns or comp_b not in standardized_panel.columns:
                missing.append((name, comp_a, comp_b))
                continue
            out[name] = standardized_panel[comp_a] * standardized_panel[comp_b]

        if missing:
            names = ", ".join(m[0] for m in missing)
            raise KeyError(
                f"Cannot build interaction(s) {names}: missing component column(s) in the "
                f"input panel. All twelve interactions require both components present -- "
                f"check that data/README_data.md's 31-characteristic diagnostic-study schema "
                f"(which includes analyst_revisions, eps_vol) was used, not the 27-characteristic set."
            )

        builder = CharacteristicBuilder()
        return builder.standardize(out, date_col="date")

    def __repr__(self) -> str:  # noqa: D105
        return f"InteractionCatalog(n_interactions={len(INTERACTION_CATALOG)})"
