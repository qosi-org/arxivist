"""
Top-20 cryptocurrency quantum-readiness survey.

Implements Section 6, Table 3, and Figure 5 of arXiv:2606.14484. This is
explicitly NOT a computed model -- the paper's own Appendix A states "the
market survey of Section 6 is a sourced field assessment, not a model."
Ratings are therefore loaded from a transcribed CSV (data/table3_readiness_ratings.csv)
rather than derived from any formula.

SIR reference: architecture.modules "Top-20 cryptocurrency quantum-readiness
survey" (confidence 0.5) -- the precise point-scoring rubric distinguishing
e.g. a 2.5 from a 3 is illustrated by example in the paper's Table 3 but not
given as a formal scoring formula.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


class QuantumReadinessSurvey:
    """Loads and summarises the top-20 cryptocurrency quantum-readiness
    ratings transcribed from the paper's Table 3."""

    def __repr__(self) -> str:  # noqa: D105
        return "QuantumReadinessSurvey()"

    def load_ratings(self, csv_path: str) -> pd.DataFrame:
        """Load the transcribed Table 3 ratings.

        Args:
            csv_path: path to a CSV with columns: coin, signature_scheme,
                exposure_model, pq_status, rating.

        Returns:
            DataFrame with those columns.

        Raises:
            FileNotFoundError: if csv_path does not exist.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Table 3 ratings CSV not found at {csv_path}. This data is "
                f"transcribed directly from the paper (not derivable from any "
                f"formula, since Section 6 is explicitly a qualitative survey, "
                f"not a model) -- see data/table3_readiness_ratings.csv."
            )
        return pd.read_csv(path)

    def summary_stats(self, ratings_df: pd.DataFrame) -> Dict:
        """Compute summary statistics matching the paper's stated findings.

        Args:
            ratings_df: DataFrame from load_ratings().

        Returns:
            Dict with mean_rating, median_rating, max_rating,
            no_coin_reaches_5 (bool, should be True per the paper),
            and rating_counts (dict of rating -> count).
        """
        ratings = ratings_df["rating"]
        return {
            "mean_rating": float(ratings.mean()),
            "median_rating": float(ratings.median()),
            "max_rating": float(ratings.max()),
            "no_coin_reaches_5": bool(ratings.max() < 5),
            "rating_counts": ratings.value_counts().sort_index(ascending=False).to_dict(),
        }
