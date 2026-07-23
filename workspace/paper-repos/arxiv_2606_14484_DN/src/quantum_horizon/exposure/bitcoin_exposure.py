"""
Bitcoin exposure reconciliation model.

Implements Section 4.1-4.2 and Figure 3 of arXiv:2606.14484: reconciles
three independent 2025-26 on-chain measurements (Glassnode 6.04M/30.2%,
Coinbase ~6.9M, CoinDesk ~7M) and Deloitte's 2020 baseline (~25%) into a
single decomposition of Bitcoin's exposed supply into irreducible (dormant/
lost/Satoshi-era, ~2.3M BTC), migratable (exposed but spendable, ~3.7M BTC),
and protected (fresh hash-based addresses, ~65-70%).

SIR reference: implementation_assumptions[3], ambiguities[2] (confidence 0.5)
-- the exact reconciliation formula combining the raw source measurements is
not given in the paper; a simple mean is used here (ASSUMED, config.yaml
`reconciliation_method`).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class BitcoinExposureModel:
    """Reconciles Bitcoin on-chain exposure measurements and decomposes
    exposed supply into irreducible/migratable/protected categories.
    """

    def __repr__(self) -> str:  # noqa: D105
        return "BitcoinExposureModel()"

    def reconcile_sources(self, measurements: Dict[str, float], method: str = "simple_mean") -> float:
        """Reconcile multiple named source measurements of exposed-supply
        fraction into a single figure.

        Args:
            measurements: dict of source_name -> exposed_fraction (e.g.
                {'glassnode': 0.302, 'coinbase': 0.345, 'coindesk': 0.35,
                'deloitte_2020': 0.25}). Values expressed as fractions of
                total supply (convert BTC-count sources to fractions before
                calling, e.g. coinbase_btc / total_supply_btc).
            method: reconciliation strategy. Only "simple_mean" is
                implemented (ASSUMED default; paper does not specify the
                exact reconciliation formula -- see module docstring).

        Returns:
            Reconciled exposed-supply fraction.

        Raises:
            ValueError: if an unsupported method is requested.
        """
        if method != "simple_mean":
            raise ValueError(
                f"Unsupported reconciliation method '{method}'. The paper does not "
                f"specify its exact reconciliation formula (SIR confidence 0.5); "
                f"only 'simple_mean' is implemented as a documented default. Extend "
                f"this method to add e.g. a source-quality-weighted average."
            )
        return float(np.mean(list(measurements.values())))

    def decompose(
        self,
        total_supply_btc_millions: float,
        irreducible_btc_millions: float,
        migratable_btc_millions: float,
    ) -> Dict[str, float]:
        """Decompose total supply into irreducible / migratable / protected
        categories (Figure 3).

        Args:
            total_supply_btc_millions: total circulating BTC supply, in millions.
            irreducible_btc_millions: dormant/lost/Satoshi-era BTC, in millions
                (paper: ~2.3).
            migratable_btc_millions: exposed-but-spendable BTC, in millions
                (paper: ~3.7).

        Returns:
            Dict with keys: irreducible_btc, migratable_btc, protected_btc
            (all in millions), and irreducible_frac, migratable_frac,
            protected_frac (fractions of total_supply_btc_millions).
        """
        protected_btc = total_supply_btc_millions - irreducible_btc_millions - migratable_btc_millions
        return {
            "irreducible_btc": irreducible_btc_millions,
            "migratable_btc": migratable_btc_millions,
            "protected_btc": protected_btc,
            "irreducible_frac": irreducible_btc_millions / total_supply_btc_millions,
            "migratable_frac": migratable_btc_millions / total_supply_btc_millions,
            "protected_frac": protected_btc / total_supply_btc_millions,
        }
