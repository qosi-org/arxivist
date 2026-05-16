"""
src/geomherd/geometry/ricci_flow.py
Discrete Ricci flow and neckpinch singularity time (tau_sing).
Paper: arXiv:2605.11645, Section 2.4

WARNING: The per-step Ricci flow update rule is NOT explicitly stated in the paper.
Only the stopping criterion (first neckpinch) is specified. This implementation
uses a standard multiplicative update (Risk R1 from architecture plan):
    w_{s+1}(e) <- w_s(e) * (1 - step_size * kappa_OR^(s)(e))

The --flow_variant config flag allows switching to other update rules.
See Risk R1 in architecture_plan.json.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from geomherd.geometry.ricci_curvature import OllivierRicciComputer


class DiscreteRicciFlow:
    """
    Runs discrete Ricci flow from a graph snapshot and records the singularity time.

    Paper reference: Section 2.4
        tau_sing(t) = inf{s > 0 : exists e in E_t, kappa_OR^(s)(e) -> -inf}
        Used as a forward-looking proximity-to-collapse descriptor.

    # WARNING: low-confidence implementation (confidence 0.60)
    # TODO: verify the exact Ricci flow update rule from the paper or authors.
    # Current implementation: multiplicative update (standard graph Ricci flow)
    #   w_{s+1}(e) = w_s(e) * (1 - step_size * kappa_OR(e))
    # Alternative: additive: w_{s+1}(e) = w_s(e) - step_size * kappa_OR(e)
    # See: Lin, Lu, Yau (2011); Ni et al. (2019) for common discrete Ricci flow variants.

    Args:
        orc_computer: OllivierRicciComputer instance (reused for efficiency)
        step_size: Flow step size (ASSUMED: 0.01)
        max_iter: Maximum flow iterations (ASSUMED: 1000)
        neckpinch_threshold: kappa value below which neckpinch is declared (ASSUMED: -50.0)
        flow_variant: 'multiplicative' (default) or 'additive'
    """

    def __init__(
        self,
        orc_computer: Optional[OllivierRicciComputer] = None,
        step_size: float = 0.01,
        max_iter: int = 1000,
        neckpinch_threshold: float = -50.0,
        flow_variant: str = "multiplicative",
    ):
        self.orc = orc_computer or OllivierRicciComputer()
        self.step_size = step_size
        self.max_iter = max_iter
        self.neckpinch_threshold = neckpinch_threshold
        assert flow_variant in ("multiplicative", "additive"), \
            f"flow_variant must be 'multiplicative' or 'additive', got {flow_variant}"
        self.flow_variant = flow_variant

    def run(
        self,
        W_init: np.ndarray,
        edge_list: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Tuple[float, List[float]]:
        """
        Run discrete Ricci flow from the initial weight matrix W_init.

        Args:
            W_init: [N, N] initial weight matrix (current graph snapshot)
            edge_list: Optional pre-computed edge list for efficiency
        Returns:
            tau_sing: First neckpinch time (float; max_iter if no neckpinch detected)
            min_kappa_history: List of minimum kappa per flow step (for diagnostics)
        """
        N = W_init.shape[0]
        assert W_init.shape == (N, N), f"W_init must be [N,N], got {W_init.shape}"

        W = W_init.copy().astype(np.float64)
        min_kappa_history: List[float] = []

        for s in range(self.max_iter):
            if edge_list is None:
                rows, cols = np.where((W > 0) & (np.triu(np.ones((N, N)), k=1) > 0))
                current_edges = [(int(i), int(j), float(W[i, j]))
                                 for i, j in zip(rows, cols)]
            else:
                current_edges = [(i, j, float(W[i, j])) for (i, j, _) in edge_list
                                 if W[i, j] > 0]

            if not current_edges:
                # Graph dissolved — return current step as singularity
                return float(s), min_kappa_history

            kappa_dict = self.orc.compute(W, current_edges)

            if not kappa_dict:
                return float(s), min_kappa_history

            min_kappa = min(kappa_dict.values())
            min_kappa_history.append(min_kappa)

            # Neckpinch detection: kappa -> -inf proxy
            # Paper: tau_sing = inf{s>0: exists e, kappa^(s)(e) -> -inf}
            if min_kappa < self.neckpinch_threshold:
                return float(s), min_kappa_history

            # Flow update step
            # WARNING: low-confidence (Risk R1) — multiplicative update assumed
            for (i, j, _) in current_edges:
                kappa = kappa_dict.get((i, j), kappa_dict.get((j, i), 0.0))
                if self.flow_variant == "multiplicative":
                    # w <- w * (1 - step_size * kappa)
                    new_w = W[i, j] * (1.0 - self.step_size * kappa)
                else:  # additive
                    # w <- w - step_size * kappa
                    new_w = W[i, j] - self.step_size * kappa
                new_w = max(0.0, new_w)  # clamp to non-negative
                W[i, j] = new_w
                W[j, i] = new_w

        # No neckpinch detected within max_iter steps
        return float(self.max_iter), min_kappa_history

    def singularity_time(self, W_init: np.ndarray) -> float:
        """
        Convenience wrapper: run flow and return tau_sing.
        Paper: Section 2.4 — tau_sing(t) as forward-looking time-to-coordination.
        """
        tau_sing, _ = self.run(W_init)
        return tau_sing

    def __repr__(self) -> str:
        return (f"DiscreteRicciFlow(step_size={self.step_size}, "
                f"max_iter={self.max_iter}, variant={self.flow_variant})")
