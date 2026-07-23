"""
Plotting utilities reproducing Figures 1-5 of arXiv:2606.14484.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure1_hashrate(
    gate_speeds_ghz: np.ndarray,
    hashrates_ths: np.ndarray,
    asic_hashrate_ths: float,
    network_hashrate_ehs: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 1: effective quantum hashrate vs gate speed, compared
    against one ASIC and the full 2026 network.

    Args:
        gate_speeds_ghz: swept gate speeds, shape [N].
        hashrates_ths: corresponding effective hashrates in TH/s, shape [N].
        asic_hashrate_ths: single-ASIC hashrate for the reference line.
        network_hashrate_ehs: 2026 network hashrate in EH/s for the reference line.
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(gate_speeds_ghz, hashrates_ths, label="1 quantum machine", color="tab:blue", linewidth=2)
    ax.axhline(asic_hashrate_ths, color="tab:orange", linestyle="--", label=f"one ASIC ({asic_hashrate_ths:.0f} TH/s)")
    ax.axhline(
        network_hashrate_ehs * 1e6, color="tab:red", linestyle=":",
        label=f"2026 network (~{network_hashrate_ehs:.0f} EH/s)",
    )
    ax.set_xlabel("gate speed (GHz)")
    ax.set_ylabel("effective hashrate (TH/s)")
    ax.set_title("Effective quantum 'hashrate' vs gate speed (cf. paper Figure 1)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure2_forecast(
    survey_samples: np.ndarray,
    physics_samples: np.ndarray,
    combined_samples: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 2: the two estimators and the bimodal combined
    distribution.

    Args:
        survey_samples: survey-estimator-only samples.
        physics_samples: physics-estimator-only samples.
        combined_samples: the blended forecast samples.
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(2025, 2065, 41)

    ax.hist(survey_samples, bins=bins, alpha=0.4, color="tab:orange", density=True, label="expert-survey estimator")
    ax.hist(physics_samples, bins=bins, alpha=0.4, color="tab:blue", density=True, label="physics estimator")
    ax.hist(
        combined_samples, bins=bins, histtype="step", color="tab:red", density=True,
        linewidth=2, label="combined (bimodal)",
    )
    median = np.median(combined_samples)
    ax.axvline(median, color="black", linestyle="--", label=f"median {median:.0f}")

    ax.set_xlabel("break-year")
    ax.set_ylabel("probability density")
    ax.set_title("Systemic break-year forecast (cf. paper Figure 2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure3_exposure_pie(decomposition: Dict[str, float], save_path: Optional[str] = None) -> plt.Figure:
    """Reproduce Figure 3: Bitcoin's supply split into permanently-at-risk,
    migratable, and protected coins.

    Args:
        decomposition: dict from BitcoinExposureModel.decompose() with keys
            irreducible_btc, migratable_btc, protected_btc.
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = [
        f"irreducible/dormant\n~{decomposition['irreducible_btc']:.1f}M "
        f"({decomposition['irreducible_frac']*100:.0f}%)",
        f"migratable\n~{decomposition['migratable_btc']:.1f}M "
        f"({decomposition['migratable_frac']*100:.0f}%)",
        f"protected until spent\n~{decomposition['protected_btc']:.1f}M "
        f"({decomposition['protected_frac']*100:.0f}%)",
    ]
    values = [decomposition["irreducible_btc"], decomposition["migratable_btc"], decomposition["protected_btc"]]
    colors = ["#c0392b", "#e67e22", "#2980b9"]
    ax.pie(values, labels=labels, colors=colors, autopct="", startangle=90)
    ax.set_title("Bitcoin's supply split (cf. paper Figure 3)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure4_migration_race(
    scenario_results: pd.DataFrame, crqc_lines: Dict[str, float], save_path: Optional[str] = None
) -> plt.Figure:
    """Reproduce Figure 4: migration finish year vs CRQC arrival lines.

    Args:
        scenario_results: DataFrame from MigrationRaceModel.run_scenarios(),
            deduplicated to one row per scenario (finish_year).
        crqc_lines: dict of estimate_name -> crqc_arrival_year for the
            horizontal reference lines.
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scenarios = scenario_results.drop_duplicates(subset=["scenario"])
    scenario_names = scenarios["scenario"].tolist()
    finish_years = scenarios["finish_year"].tolist()

    colors = ["tab:blue", "tab:orange", "tab:red"]
    bars = ax.bar(scenario_names, finish_years, color=colors[: len(scenario_names)])
    for bar, year in zip(bars, finish_years):
        ax.text(bar.get_x() + bar.get_width() / 2, year + 0.3, f"{year:.0f}", ha="center")

    line_colors = {"aggressive": "tab:red", "survey_median": "tab:orange", "conservative": "tab:green"}
    for name, year in crqc_lines.items():
        ax.axhline(year, linestyle="--", color=line_colors.get(name, "gray"), alpha=0.7, label=f"{name} {year:.0f}")

    ax.set_ylabel("migration finish year")
    ax.set_title("Migration race vs CRQC arrival (cf. paper Figure 4)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure5_readiness(ratings_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Reproduce Figure 5: top-20 quantum-readiness ratings bar chart.

    Args:
        ratings_df: DataFrame from QuantumReadinessSurvey.load_ratings().
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    sorted_df = ratings_df.sort_values("rating", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))

    norm_ratings = (sorted_df["rating"] - 1) / 4.0
    colors = plt.cm.RdYlGn(norm_ratings)

    ax.barh(sorted_df["coin"], sorted_df["rating"], color=colors)
    for i, (coin, rating) in enumerate(zip(sorted_df["coin"], sorted_df["rating"])):
        ax.text(rating + 0.05, i, f"{rating}", va="center", fontsize=8)

    ax.set_xlabel("quantum-readiness rating (1 = no PQ effort ... 5 = PQ live by default)")
    ax.set_xlim(0, 5)
    ax.set_title("Top-20 quantum-readiness ratings (cf. paper Figure 5)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
