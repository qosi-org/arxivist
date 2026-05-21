"""
src/geomherd/geometry/ricci_curvature.py
Ollivier-Ricci curvature computation via LP Wasserstein-1.
Paper: arXiv:2605.11645, Section 2.2 (Eqs. 2-3)

Implements:
  - Lazy-walk transition kernel (Eq. 2)
  - Ollivier-Ricci curvature via W1 solved by LP (Eq. 3)
  - Sign decomposition into E+/E- for herding/contagion detection
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import ot  # Python Optimal Transport (POT library)
except ImportError as e:
    raise ImportError(
        "POT (Python Optimal Transport) is required. Install with: pip install POT"
    ) from e


class OllivierRicciComputer:
    """
    Computes Ollivier-Ricci curvature on the agent interaction graph.

    Paper reference: Section 2.2, Eqs. 2-3
        Lazy-walk kernel (Eq. 2):
            mu_i(j) = alpha * delta_ij + (1 - alpha) * w_t(i,j) / sum_k(w_t(i,k))
        ORC (Eq. 3):
            kappa_OR(i,j;t) = 1 - W1(mu_i, mu_j) / d_t(i,j)
        where d_t(i,j) = w_t(i,j) (agreement weight as distance).

    Sign decomposition:
        E+ = {e: kappa_OR(e) > kappa_plus_thresh}  -> within-clique (herding)
        E- = {e: kappa_OR(e) < kappa_minus_thresh} -> bridges (contagion)

    Args:
        alpha: Lazy-walk laziness (default 0.5, matching Sandhu et al. 2016)
        kappa_plus_thresh: Threshold for E+ classification (default 0.1)
        kappa_minus_thresh: Threshold for E- classification (default -0.1)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        kappa_plus_thresh: float = 0.1,
        kappa_minus_thresh: float = -0.1,
    ):
        assert 0 < alpha < 1, f"alpha must be in (0,1), got {alpha}"
        assert kappa_plus_thresh > kappa_minus_thresh, \
            "kappa_plus_thresh must be > kappa_minus_thresh"
        self.alpha = alpha
        self.kappa_plus_thresh = kappa_plus_thresh
        self.kappa_minus_thresh = kappa_minus_thresh

    def _lazy_walk_kernel(
        self, W: np.ndarray, node: int
    ) -> np.ndarray:
        """
        Compute the lazy-walk probability measure for a single node.

        Eq. 2: mu_i(j) = alpha * delta_ij + (1 - alpha) * w(i,j) / sum_k w(i,k)

        Args:
            W: [N, N] weight matrix (dense)
            node: index i
        Returns:
            mu: [N] probability vector summing to 1
        """
        N = W.shape[0]
        row = W[node, :].copy()
        row_sum = row.sum()
        mu = np.zeros(N, dtype=np.float64)
        if row_sum > 0:
            mu = (1.0 - self.alpha) * row / row_sum
        mu[node] += self.alpha  # lazy self-loop
        return mu

    def _wasserstein1_lp(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> float:
        """
        Compute W1(mu, nu) via linear programming using POT's exact EMD solver.

        Paper: Section 2.2, Appendix C — 'POT for exact W1 via linear programming'

        Args:
            mu: [N] source probability measure
            nu: [N] target probability measure
            cost_matrix: [N, N] ground metric (pairwise distances)
        Returns:
            W1 distance (float)
        """
        # POT ot.emd2 solves the exact Earth Mover's Distance via LP
        w1 = ot.emd2(mu, nu, cost_matrix)
        return float(w1)

    def compute(
        self,
        W: np.ndarray,
        edge_list: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute Ollivier-Ricci curvature for all edges.

        Args:
            W: [N, N] dense weight matrix (after sparsification)
            edge_list: Optional list of (i, j, w) edges; if None, inferred from W > 0
        Returns:
            kappa_dict: {(i,j): kappa_OR(i,j)} for each retained edge
        """
        N = W.shape[0]
        assert W.shape == (N, N), f"W must be square [N,N], got {W.shape}"

        if edge_list is None:
            rows, cols = np.where((W > 0) & (np.triu(np.ones((N, N)), k=1) > 0))
            edge_list = [(int(i), int(j), float(W[i, j])) for i, j in zip(rows, cols)]

        # Cost matrix: use shortest-path distances on the weighted graph
        # Paper: d_t(i,j) = w_t(i,j) (similarity-as-distance)
        # For the W1 ground metric, we use the full pairwise w matrix as cost
        # NOTE: d(i,j)=w(i,j) means high agreement => short distance (correct sign)
        cost_matrix = W.astype(np.float64)
        # For nodes with no direct edge, use w=0 (maximum distance in this metric)
        # We set 0-weight pairs to a large cost to avoid trivial transport
        cost_matrix_for_transport = np.where(
            cost_matrix > 0, cost_matrix, np.zeros_like(cost_matrix)
        )
        # Use 1 - w as cost (higher agreement = lower cost = shorter distance)
        # This preserves: d(i,j) = w(i,j) but in transport cost space
        # Following paper: d_t(i,j) = w_t(i,j) directly as edge length
        ground_metric = cost_matrix_for_transport.copy()

        kappa_dict: Dict[Tuple[int, int], float] = {}
        for (i, j, w_ij) in edge_list:
            if w_ij <= 0:
                continue
            mu_i = self._lazy_walk_kernel(W, i)
            mu_j = self._lazy_walk_kernel(W, j)
            w1 = self._wasserstein1_lp(mu_i, mu_j, ground_metric)
            # Eq. 3: kappa_OR(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
            # d(i,j) = w_ij (agreement weight as distance)
            kappa = 1.0 - (w1 / w_ij) if w_ij > 0 else 0.0
            kappa_dict[(i, j)] = kappa

        return kappa_dict

    def mean_curvature_plus(self, kappa_dict: Dict[Tuple[int, int], float]) -> float:
        """
        Mean Ollivier-Ricci curvature over E+ (positive/herding edges).
        Paper: Section 2.2 — kappa_bar_plus_OR(t) = mean over E+_t
        """
        vals = [v for v in kappa_dict.values() if v > self.kappa_plus_thresh]
        return float(np.mean(vals)) if vals else 0.0

    def mean_curvature_all(self, kappa_dict: Dict[Tuple[int, int], float]) -> float:
        """Mean ORC over all edges."""
        vals = list(kappa_dict.values())
        return float(np.mean(vals)) if vals else 0.0

    def beta_minus(self, kappa_dict: Dict[Tuple[int, int], float]) -> float:
        """
        Fraction of edges in E- (bridge/contagion edges).
        Paper: Section 2.2 — beta_minus(t) = |E-_t| / |E_t|
        """
        if not kappa_dict:
            return 0.0
        n_minus = sum(1 for v in kappa_dict.values() if v < self.kappa_minus_thresh)
        return float(n_minus) / len(kappa_dict)

    def sign_decompose(
        self, kappa_dict: Dict[Tuple[int, int], float]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Decompose edges into E+, E0, E-.
        Returns (E_plus, E_zero, E_minus) dicts.
        """
        E_plus = {k: v for k, v in kappa_dict.items() if v > self.kappa_plus_thresh}
        E_minus = {k: v for k, v in kappa_dict.items() if v < self.kappa_minus_thresh}
        E_zero = {k: v for k, v in kappa_dict.items()
                  if self.kappa_minus_thresh <= v <= self.kappa_plus_thresh}
        return E_plus, E_zero, E_minus

    def __repr__(self) -> str:
        return (f"OllivierRicciComputer(alpha={self.alpha}, "
                f"kappa+={self.kappa_plus_thresh}, kappa-={self.kappa_minus_thresh})")
