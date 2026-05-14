"""
utils/plotting.py
─────────────────
Implied volatility surface visualisation and error table utilities.
Reproduces Figures 5.1, 5.2, 6.1 and Tables 5.1, 5.2, 6.1 from the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class IVSurfacePlotter:
    """
    3D implied volatility surface plots and error tables.

    Reproduces the visual style of Figures 5.1, 5.2, 6.1 from the paper.
    """

    def __init__(self, S0: float = 100.0):
        self.S0 = S0

    def __repr__(self) -> str:
        return f"IVSurfacePlotter(S0={self.S0})"

    def plot_surface_comparison(
        self,
        iv1: np.ndarray,          # [nT, nK]
        iv2: np.ndarray,          # [nT, nK]
        strikes: np.ndarray,      # [nK]
        maturities: np.ndarray,   # [nT]
        labels: List[str] = ("SIG IV", "Analytical IV"),
        title: str = "Comparison of Implied Volatility Surfaces",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot two IV surfaces on the same 3D axes for comparison.
        Reproduces the style of Figure 5.1 / 5.2 / 6.1 from the paper.

        Args:
            iv1:        [nT, nK] first IV surface (e.g. SIG).
            iv2:        [nT, nK] second IV surface (e.g. ASV or VIX).
            strikes:    [nK] strike grid.
            maturities: [nT] maturity grid.
            labels:     Legend labels for the two surfaces.
            title:      Plot title.
            save_path:  If given, save to this path.
            show:       If True, call plt.show().
        """
        if not _HAS_MPL:
            print("[IVSurfacePlotter] matplotlib not available — skipping plot.")
            return

        T_grid, K_grid = np.meshgrid(maturities, strikes, indexing="ij")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Surface 1 (scatter + line)
        ax.scatter(
            T_grid.flatten(), K_grid.flatten(), iv1.flatten(),
            c="steelblue", s=30, label=labels[0], zorder=5,
        )
        # Surface 2 (scatter + line, different marker)
        ax.scatter(
            T_grid.flatten(), K_grid.flatten(), iv2.flatten(),
            c="tomato", marker="^", s=30, label=labels[1], zorder=5,
        )
        # Connect points at same maturity
        for i in range(len(maturities)):
            ax.plot(
                [maturities[i]] * len(strikes), strikes, iv1[i, :],
                color="steelblue", linewidth=1.0, alpha=0.7,
            )
            ax.plot(
                [maturities[i]] * len(strikes), strikes, iv2[i, :],
                color="tomato", linewidth=1.0, alpha=0.7,
            )

        ax.set_xlabel("T (Maturity)")
        ax.set_ylabel("K (Strike)")
        ax.set_zlabel("IV (Implied Volatility)")
        ax.set_title(title)
        ax.legend(loc="upper left")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[IVSurfacePlotter] Saved to {save_path}")
        if show:
            plt.show()
        plt.close()

    def error_table(
        self,
        iv_model: np.ndarray,    # [nT, nK]  model IV surface
        iv_market: np.ndarray,   # [nT, nK]  market IV surface
        iv_analytical: np.ndarray,  # [nT, nK]  analytical (ASV/VIX) IV surface
        strikes: np.ndarray,
        maturities: np.ndarray,
        model_label: str = "SIG",
        analytical_label: str = "ASV",
        save_path: Optional[str] = None,
    ):
        """
        Print and optionally save the error table replicating Tables 5.1, 5.2, 6.1.

        Columns: T, K, e^{analytical}, e^{SIG}  (absolute IV errors)
        Entries marked (*) where e^{SIG} < e^{analytical}.

        Args:
            iv_model:      [nT, nK] signature model IV.
            iv_market:     [nT, nK] market (ground truth) IV.
            iv_analytical: [nT, nK] analytical approximation IV.
            strikes:       [nK] strike grid.
            maturities:    [nT] maturity grid.
            model_label:   Label for model column header.
            analytical_label: Label for analytical column header.
            save_path:     If given, save CSV to this path.

        Returns:
            DataFrame if pandas available, else None.
        """
        rows = []
        header = f"{'T':>6} {'K':>6}  {'e^'+analytical_label:>10}  {'e^'+model_label:>10}  Flag"
        print("\n" + "=" * len(header))
        print(f"  Calibration Error Table  ({analytical_label} vs {model_label})")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                e_analytical = abs(float(iv_analytical[i, j] - iv_market[i, j]))
                e_model = abs(float(iv_model[i, j] - iv_market[i, j]))
                flag = "(*)" if e_model < e_analytical else "   "
                print(f"{T:>6.2f} {K:>6.0f}  {e_analytical:>10.5f}  {e_model:>10.5f}  {flag}")
                rows.append({
                    "T": T, "K": K,
                    f"e_{analytical_label}": e_analytical,
                    f"e_{model_label}": e_model,
                    "sig_wins": e_model < e_analytical,
                })
        print("=" * len(header))

        sig_wins = sum(r["sig_wins"] for r in rows)
        print(f"  {model_label} outperforms {analytical_label} in {sig_wins}/{len(rows)} contracts.\n")

        if _HAS_PANDAS:
            df = _pd_import().DataFrame(rows)
            if save_path:
                df.to_csv(save_path, index=False)
                print(f"[IVSurfacePlotter] Error table saved to {save_path}")
            return df
        return None


def _pd_import():
    """Lazy pandas import."""
    import pandas as pd
    return pd


def print_calibration_summary(
    l_star: np.ndarray,
    loss_final: float,
    elapsed_seconds: float,
    experiment: str,
) -> None:
    """
    Print a calibration summary matching the paper's reported results.

    Args:
        l_star:          Calibrated ℓ* vector.
        loss_final:      Final loss L(ℓ*).
        elapsed_seconds: Wall-clock time.
        experiment:      Name of the experiment.
    """
    print("\n" + "=" * 60)
    print(f"  Calibration Summary — {experiment}")
    print("=" * 60)
    print(f"  Elapsed time:    {elapsed_seconds/60:.1f} min ({elapsed_seconds:.0f}s)")
    print(f"  Final loss L(ℓ*): {loss_final:.4e}")
    print(f"  ℓ* vector (n={len(l_star)}):")
    for i, v in enumerate(l_star):
        print(f"    ℓ[{i:2d}] = {v:+.8f}")
    print("=" * 60 + "\n")
