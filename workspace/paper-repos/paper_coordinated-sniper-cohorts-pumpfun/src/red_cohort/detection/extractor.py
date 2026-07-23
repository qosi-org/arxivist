"""
detection/extractor.py
-----------------------
Stage 1 — IntraLaunchExtractor.

Filters the full buyer event stream to the first-N buyers per launch,
producing the per-launch index that feeds the co-occurrence graph.

Paper: Kamat (2026), Section 4.1 — "Intra-launch first-buyer window extraction".
"""
from __future__ import annotations

from typing import List

import pandas as pd


class IntraLaunchExtractor:
    """
    For each token launch, extract the ordered list of the first *window_size*
    buyers within the bonding-curve window.

    Args:
        window_size: Number of early buyers to retain per launch (default: 10).
                     Paper uses N=10 (Section 4.1).

    Paper reference:
        Section 4.1 — "For each launch L, we extract the ordered list of the
        first ten buyers within the bonding-curve window."

    Output schema per row:
        mint (str), wallet (str), rank (int), block_time (int), sol_committed (float)
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size

    def extract(self, buyers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter *buyers_df* to rows where rank <= window_size.

        Buyer rank is computed from blockTime ASC, tx_sig ASC within each mint
        (tie-breaking convention — see loader.py TODO note, SIR confidence 0.65).

        Args:
            buyers_df: Raw buyer events DataFrame from DataLoader.load_buyers().

        Returns:
            intra_index: DataFrame with columns
                {mint, wallet, rank, block_time, sol_committed}
                containing only buyers with rank in [1, window_size].
        """
        # Recompute rank from blockTime + tx_sig to guarantee deterministic ordering
        buyers_df = buyers_df.copy()
        buyers_df["_sort_key"] = buyers_df["blockTime"].astype(str) + "_" + buyers_df["tx_sig"]
        buyers_df = buyers_df.sort_values(["mint", "blockTime", "tx_sig"])
        buyers_df["rank"] = buyers_df.groupby("mint").cumcount() + 1

        # Filter to first-window_size buyers per mint
        intra = buyers_df[buyers_df["rank"] <= self.window_size].copy()

        # Rename and select output columns
        intra = intra.rename(columns={"blockTime": "block_time", "sol_in": "sol_committed"})
        intra = intra[["mint", "wallet", "rank", "block_time", "sol_committed"]].reset_index(drop=True)
        return intra

    def get_qualifying_mints(self, buyers_df: pd.DataFrame) -> List[str]:
        """
        Return the list of mints that have at least one buyer in [1, window_size].
        Paper reports 166,098 qualifying mints (Section 3.1).
        """
        intra = self.extract(buyers_df)
        return intra["mint"].unique().tolist()

    def __repr__(self) -> str:
        return f"IntraLaunchExtractor(window_size={self.window_size})"
