"""
utils.py — Plotting helpers and utilities

Reproduces the paper's key figures:
  Fig 3a: Cost function convergence (R_emp vs SPSA iterations)
  Fig 3b: Decision boundary + support vectors + test point classification
  Fig 3c: Classification success vs circuit depth (QVC + QKE)
  Fig 4a: Kernel matrix heatmap (ideal vs estimated)
  Fig 4b: Row cut through kernel matrix

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python stdlib and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Fig 3a: Cost convergence
# ---------------------------------------------------------------------------

def plot_cost_convergence(
    cost_histories: Dict[int, List[List[float]]],
    output_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Reproduce Fig. 3a: R_emp(θ) convergence over SPSA iterations.

    Parameters
    ----------
    cost_histories : Dict[int, List[List[float]]]
        {depth: [[cost_iter_0, ..., cost_iter_249], ...]} — one list per dataset.
    output_path : str or Path, optional
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {0: "black", 1: "#2166ac", 2: "#4393c3", 3: "#92c5de", 4: "red"}
    labels = {0: "l=0", 4: "l=4"}

    for depth, histories in cost_histories.items():
        color = colors.get(depth, "grey")
        for hist in histories:
            iters = np.arange(len(hist))
            label = labels.get(depth, f"l={depth}")
            ax.plot(iters, hist, color=color, alpha=0.7, linewidth=1.2,
                    label=label if len(histories) == 1 else None)

    ax.set_xlabel("Trial step", fontsize=12)
    ax.set_ylabel(r"$R_\mathrm{emp}(\vec{\theta})$", fontsize=12)
    ax.set_title("Cost function convergence (reproducing Fig. 3a)", fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Fig 3b: Decision boundary
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    qke_model,            # QuantumKernelSVM (fitted)
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    resolution: int = 60,
    domain: Tuple[float, float] = (0.0, 6.2832),
    output_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Reproduce Fig. 3b: decision boundary, support vectors, test classification.

    Parameters
    ----------
    qke_model : QuantumKernelSVM
    X_train, y_train : training data
    X_test, y_test : test data
    y_pred : np.ndarray, optional — predicted labels for X_test
    resolution : int — grid resolution (60 for speed; paper shows high resolution)
    domain : (min, max) for both axes
    output_path : optional save path
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    lo, hi = domain
    xx, yy = np.meshgrid(
        np.linspace(lo, hi, resolution),
        np.linspace(lo, hi, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid (may take a few seconds)
    Z = qke_model.predict(grid_points, verbose=False).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background: decision regions (red=+1, blue=-1, white=boundary)
    ax.contourf(xx, yy, Z, levels=[-1.5, 0, 1.5],
                colors=["#b2d8e6", "#f4b8b8"], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], colors="white", linewidths=1.0)

    # Training data
    ax.scatter(X_train[y_train == +1, 0], X_train[y_train == +1, 1],
               c="white", edgecolors="black", s=60, zorder=3, label="Train +1")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
               c="black", edgecolors="black", s=60, zorder=3, label="Train -1")

    # Support vectors
    svs, _, sv_labels = qke_model.get_support_vectors()
    ax.scatter(svs[:, 0], svs[:, 1], c="lime", edgecolors="black",
               s=100, marker="o", zorder=4, label="Support vectors")

    # Test points (colour by true label, shape by correct/incorrect)
    if y_pred is None:
        y_pred = qke_model.predict(X_test)

    correct_mask = y_pred == y_test
    ax.scatter(X_test[correct_mask & (y_test == +1), 0],
               X_test[correct_mask & (y_test == +1), 1],
               c="white", edgecolors="black", marker="s", s=50, zorder=5)
    ax.scatter(X_test[correct_mask & (y_test == -1), 0],
               X_test[correct_mask & (y_test == -1), 1],
               c="black", edgecolors="black", marker="s", s=50, zorder=5)
    # Misclassified: red markers (A, B, C in paper)
    wrong = ~correct_mask
    if wrong.any():
        for xi, yi in zip(X_test[wrong], y_test[wrong]):
            ax.scatter(xi[0], xi[1], c="red", edgecolors="black",
                       marker="^", s=100, zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Decision boundary (reproducing Fig. 3b)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Fig 3c: Success vs depth
# ---------------------------------------------------------------------------

def plot_success_vs_depth(
    qvc_results: Dict[int, List[float]],
    qke_results: Dict[str, float],
    output_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Reproduce Fig. 3c: classification success rate vs circuit depth.

    Parameters
    ----------
    qvc_results : {depth: [rate_1, ..., rate_N]}
    qke_results : {"Set I": mean_rate, "Set II": ..., "Set III": ...}
    output_path : optional save path
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    depths = sorted(qvc_results.keys())
    means = [np.mean(qvc_results[d]) for d in depths]
    stderrs = [np.std(qvc_results[d]) / np.sqrt(len(qvc_results[d])) for d in depths]

    fig, ax = plt.subplots(figsize=(6, 5))

    # QVC scatter + error bars
    ax.errorbar(depths, means, yerr=stderrs, fmt="ko", markersize=5,
                capsize=4, label="QVC (mean ± SEM)", zorder=3)
    for d, rates in qvc_results.items():
        ax.scatter([d] * len(rates), rates, c="black", alpha=0.3, s=15, zorder=2)

    # QKE dashed lines (one per set)
    qke_colors = {"Set I": "blue", "Set II": "blue", "Set III": "blue"}
    for name, rate in qke_results.items():
        ax.axhline(rate, linestyle="--", color="blue", alpha=0.6,
                   linewidth=1.2, label=f"QKE {name}")

    ax.set_xlim(-0.5, max(depths) + 0.5)
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(depths)
    ax.set_xlabel("Depth (l)", fontsize=12)
    ax.set_ylabel("Classification success", fontsize=12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Success vs circuit depth (reproducing Fig. 3c)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Fig 4a/4b: Kernel matrix
# ---------------------------------------------------------------------------

def plot_kernel_matrix(
    K_estimated: np.ndarray,
    K_ideal: Optional[np.ndarray] = None,
    title: str = "Kernel matrix",
    cut_row: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Reproduce Fig. 4: kernel matrix heatmap and optional row cut.

    Parameters
    ----------
    K_estimated : np.ndarray, shape [N, N]
    K_ideal : np.ndarray, shape [N, N], optional
    title : str
    cut_row : int, optional — if given, plot the row cut as in Fig. 4b
    output_path : optional save path
    dpi : int

    Returns
    -------
    matplotlib Figure
    """
    has_ideal = K_ideal is not None
    has_cut = cut_row is not None

    n_cols = 2 if has_ideal else 1
    n_rows = 2 if has_cut else 1

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.35)

    # --- Heatmaps ---
    ax_est = fig.add_subplot(gs[0, 0])
    im = ax_est.imshow(K_estimated, aspect="auto", vmin=0, vmax=1,
                       cmap="gray_r", origin="upper")
    ax_est.set_title(r"Estimated $\hat{K}$", fontsize=11)
    ax_est.set_xlabel("Data index j")
    ax_est.set_ylabel("Data index i")
    plt.colorbar(im, ax=ax_est, fraction=0.046, pad=0.04)

    if has_ideal:
        ax_ideal = fig.add_subplot(gs[0, 1])
        im2 = ax_ideal.imshow(K_ideal, aspect="auto", vmin=0, vmax=1,
                               cmap="gray_r", origin="upper")
        ax_ideal.set_title(r"Ideal $K$", fontsize=11)
        ax_ideal.set_xlabel("Data index j")
        plt.colorbar(im2, ax=ax_ideal, fraction=0.046, pad=0.04)
        # Highlight cut row
        if has_cut:
            ax_est.axhline(cut_row, color="red", linewidth=1.5, linestyle="-")
            ax_ideal.axhline(cut_row, color="red", linewidth=1.5, linestyle="-")

    # --- Row cut (Fig. 4b style) ---
    if has_cut:
        ax_cut = fig.add_subplot(gs[1, :])
        indices = np.arange(K_estimated.shape[1])
        width = 0.35
        ax_cut.bar(indices - width/2, K_estimated[cut_row], width=width,
                   color="red", alpha=0.8, label=r"Estimated $\hat{K}$")
        if has_ideal:
            ax_cut.bar(indices + width/2, K_ideal[cut_row], width=width,
                       color="blue", alpha=0.8, label=r"Ideal $K$")
        ax_cut.axhline(0, color="black", linewidth=0.5)
        ax_cut.set_xlabel("Data index j")
        ax_cut.set_ylabel(f"$K_{{{cut_row},j}}$")
        ax_cut.set_title(f"Row {cut_row} cross-section (reproducing Fig. 4b)", fontsize=11)
        ax_cut.legend(fontsize=9)

    fig.suptitle(title, fontsize=12, y=1.01)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any], path: str | Path) -> None:
    """Save results dict as numpy .npz archive."""
    np.savez(path, **{k: np.array(v) for k, v in results.items()})


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
