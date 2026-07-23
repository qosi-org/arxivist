"""
Formal end-to-end error-budget decomposition.

Implements Section 4.5 (Eq. 51-56) of arXiv:2607.12990: decomposes the total
relative CVA pipeline error into finite-grid/truncation, quantum-encoding
training, and amplitude-estimation contributions via a triangle-inequality
bound:

    eps_tot <= eps_grid + alpha_n * eps_enc + beta_Theta * eps_AE

SIR reference: mathematical_spec "End-to-end pipeline error decomposition".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ErrorBudgetResult:
    """Container for the full error-budget decomposition (Table 7)."""

    eps_grid: float
    eps_enc: float
    eps_ae: float
    alpha_n: float
    beta_theta: float
    structural_bound: float
    total_bound: float
    realised_statevector_discrepancy: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ErrorBudgetResult(eps_grid={self.eps_grid:.4%}, "
            f"eps_enc_scaled={self.alpha_n*self.eps_enc:.4%}, "
            f"eps_ae_scaled={self.beta_theta*self.eps_ae:.4%}, "
            f"total_bound={self.total_bound:.4%})"
        )


class ErrorBudget:
    """Computes the Section 4.5 error decomposition from the four pipeline
    CVA values: continuous MC benchmark, finite-grid tabulated benchmark,
    noiseless statevector-trained circuit value, and amplitude-estimation
    output value.
    """

    def __repr__(self) -> str:  # noqa: D105
        return "ErrorBudget()"

    def compute_budget(
        self,
        cva_cont_mc: float,
        cva_tab: float,
        cva_sv: float,
        cva_ae: float,
    ) -> ErrorBudgetResult:
        """Compute the full error-budget decomposition (Eq. 51-56).

        Args:
            cva_cont_mc: continuous-underlying Monte Carlo CVA benchmark
                (paper: 1.091).
            cva_tab: finite-grid tabulated CVA benchmark at the chosen
                discretisation (paper: 0.522 for n=4).
            cva_sv: noiseless statevector-evaluated trained-circuit CVA
                (paper: 0.670).
            cva_ae: amplitude-estimation-recovered CVA (from CABIQAE or
                another estimator, converted to monetary units).

        Returns:
            ErrorBudgetResult with all named quantities from Table 7.
        """
        eps_grid = abs(cva_tab - cva_cont_mc) / abs(cva_cont_mc)
        eps_enc = abs(cva_sv - cva_tab) / abs(cva_tab)
        eps_ae = abs(cva_ae - cva_sv) / abs(cva_sv) if cva_sv != 0 else 0.0

        alpha_n = cva_tab / cva_cont_mc
        beta_theta = cva_sv / cva_cont_mc

        structural_bound = eps_grid + alpha_n * eps_enc
        total_bound = structural_bound + beta_theta * eps_ae
        realised_discrepancy = abs(cva_sv - cva_cont_mc) / abs(cva_cont_mc)

        return ErrorBudgetResult(
            eps_grid=eps_grid,
            eps_enc=eps_enc,
            eps_ae=eps_ae,
            alpha_n=alpha_n,
            beta_theta=beta_theta,
            structural_bound=structural_bound,
            total_bound=total_bound,
            realised_statevector_discrepancy=realised_discrepancy,
        )
