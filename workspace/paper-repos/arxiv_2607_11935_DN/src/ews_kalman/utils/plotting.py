"""
Plotting utilities reproducing Figure 1 (beta vs classical EWS overlay per
region), Figure 2 (lead-lag bar charts), and Figure 3 (simulation validation
grid) of arXiv:2607.11935.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def plot_figure1_overlay(region_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
    """Reproduce Figure 1: normalized beta, AR(1), MI, and permutation
    entropy overlaid per region.

    Args:
        region_results: dict keyed by region name, each value a dict with
            'beta', 'ar1_T', 'mi', 'perm_ent_T' arrays (already aligned/
            truncated to a common length) and 'dates' (array-like of
            datetimes, same length).
        save_path: optional path to save the figure (e.g. PNG).

    Returns:
        The matplotlib Figure.
    """
    regions = list(region_results.keys())
    fig, axes = plt.subplots(len(regions), 1, figsize=(10, 3.2 * len(regions)), sharex=False)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        r = region_results[region]
        dates = r["dates"]
        ax.plot(dates, _normalize(r["beta"]), label="beta (normalized)", linewidth=2)
        ax.plot(dates, _normalize(r["ar1_T"]), label="AR(1) T (normalized)", alpha=0.6)
        ax.plot(dates, _normalize(r["mi"]), label="MI T-q (normalized)", alpha=0.6)
        ax.plot(dates, _normalize(r["perm_ent_T"]), label="PermEnt T (normalized)", alpha=0.6)
        ax.set_title(region.capitalize())
        ax.set_ylabel("Signal (normalized)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Figure 1: beta (TVP-Kalman) vs classical EWS, by region", y=1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure2_leadlag_bars(table2_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
    """Reproduce Figure 2: cross-correlation optimal lag between beta
    (and derivatives) and each classical EWS signal, per region.

    Args:
        table2_results: dict keyed by region name, each value the nested
            dict returned by RegionSummaryComputer.compute_table2_row()
            for the 'beta' key specifically (signal_name -> {'lag', ...}).
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    regions = list(table2_results.keys())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5), sharex=True)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        signals = list(table2_results[region].keys())
        lags = [table2_results[region][s]["lag"] for s in signals]
        colors = ["tab:green" if lag > 0 else "tab:red" for lag in lags]
        ax.barh(signals, lags, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(region.capitalize())
        ax.set_xlabel("Lag (months) -> beta lead ->")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Figure 2: Lead-lag summary, beta vs all EWS", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_figure3_simulation_grid(
    simulation_raw: List[Dict], simulation_beta: List[np.ndarray], simulation_ar1: List[np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 3: 6x3 grid of (raw time series | beta | classical
    EWS) for the six simulated tipping-point systems.

    Args:
        simulation_raw: list of 6 dicts, each with 'x' (and optionally 'y')
            arrays and 'tipping_index'.
        simulation_beta: list of 6 beta(t) arrays (TVP-Kalman output).
        simulation_ar1: list of 6 AR(1) arrays (classical EWS output).
        save_path: optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    n = len(simulation_raw)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3 * n))

    for i, sim in enumerate(simulation_raw):
        tipping_idx = sim["tipping_index"]
        x = sim["x"]

        axes[i, 0].plot(x, label="x (driver/response)", color="tab:blue")
        if "y" in sim:
            axes[i, 1].axhline(0, color="gray", linewidth=0.5)
        axes[i, 0].axvline(tipping_idx, color="black", linestyle="--", label=f"Tipping t={tipping_idx}")
        axes[i, 0].set_title(sim.get("name", f"System {i+1}"))
        axes[i, 0].legend(fontsize=7)

        axes[i, 1].plot(simulation_beta[i], color="tab:green")
        axes[i, 1].axvline(tipping_idx, color="black", linestyle="--")
        axes[i, 1].set_title("beta (TVP-Kalman)")

        axes[i, 2].plot(simulation_ar1[i], color="tab:orange")
        axes[i, 2].axvline(tipping_idx, color="black", linestyle="--")
        axes[i, 2].set_title("Classical EWS (AR1)")

    fig.suptitle("Figure 3: Validation on six simulated tipping-point systems", y=1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
