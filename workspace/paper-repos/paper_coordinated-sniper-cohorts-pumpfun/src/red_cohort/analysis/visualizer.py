"""
analysis/visualizer.py
-----------------------
Produces Figures 1-3 from the paper in SVG format.

Paper: Kamat (2026), Figures 1, 2, 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


class Visualizer:
    """
    Reproduces the three paper figures as SVG files.

    Paper reference:
        Figure 1 — Cohort size distribution bar chart (Section 5).
        Figure 2 — Lorenz curve of cohort activity concentration (Section 5).
        Figure 3 — Cohort score vs launches hit scatter (Section 5 / Figure 3).
    """

    TIER_COLORS = {
        "premium": "#d62728",   # red
        "high":    "#ff7f0e",   # gold/orange
        "standard": "#aaaaaa",  # grey
    }

    def fig1_size_distribution(
        self,
        cohorts_df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """
        Figure 1: Bar chart of cohort size distribution.
        Paper: pairs (size=2) → 571 cohorts (56.4%).
        """
        counts = cohorts_df["size"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(counts.index, counts.values, color="#4878d0", edgecolor="white")
        ax.set_xlabel("Cohort size (wallets per cohort)", fontsize=11)
        ax.set_ylabel("Number of cohorts", fontsize=11)
        ax.set_title("Fig 1. Cohort size distribution", fontsize=12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Label bars with counts
        for bar, (sz, cnt) in zip(bars, counts.items()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt),
                ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="svg")
        plt.close()
        print(f"[Visualizer] Fig1 saved to {output_path}")

    def fig2_lorenz_curve(
        self,
        lorenz_data: Tuple[List[float], List[float]],
        output_path: str,
    ) -> None:
        """
        Figure 2: Lorenz curve of cohort activity concentration.
        Paper: top 1% cohorts account for >9% of cohort-touch events.
        """
        x_vals, y_vals = lorenz_data

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_vals, y_vals, color="#4878d0", linewidth=2, label="Cohort activity")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect equality")
        ax.set_xlabel("Cumulative share of cohorts (sorted by launches hit, descending)", fontsize=10)
        ax.set_ylabel("Cumulative share of launches hit", fontsize=10)
        ax.set_title("Fig 2. Lorenz curve of cohort activity concentration", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="svg")
        plt.close()
        print(f"[Visualizer] Fig2 saved to {output_path}")

    def fig3_score_vs_launches(
        self,
        cohorts_df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """
        Figure 3: Scatter plot of cohort score vs launches hit.
        Point size encodes cohort wallet count; color encodes tier.
        Paper: identifies premium tier (22 cohorts, n_launches >= 20).
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        tier_col = "tier" if "tier" in cohorts_df.columns else None

        for tier in ["standard", "high", "premium"]:
            if tier_col:
                subset = cohorts_df[cohorts_df["tier"] == tier]
            else:
                subset = cohorts_df

            label_map = {
                "standard": "standard tier",
                "high": "high tier (n_launches>=10)",
                "premium": "red: premium tier (n_launches>=20)",
            }
            ax.scatter(
                subset["n_launches_hit"],
                subset["score"],
                s=subset["size"] * 20,
                color=self.TIER_COLORS[tier],
                alpha=0.6,
                label=label_map[tier],
                edgecolors="none",
            )
            if tier_col is None:
                break

        ax.set_xlabel("Number of launches hit", fontsize=11)
        ax.set_ylabel("Cohort score", fontsize=11)
        ax.set_title("Fig 3. Cohort score vs launches hit (color: tier)", fontsize=11)
        if tier_col:
            ax.legend(fontsize=8, loc="upper left")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="svg")
        plt.close()
        print(f"[Visualizer] Fig3 saved to {output_path}")

    def __repr__(self) -> str:
        return "Visualizer()"
