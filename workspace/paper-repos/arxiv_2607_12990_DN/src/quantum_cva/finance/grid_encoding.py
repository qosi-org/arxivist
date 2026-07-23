"""
Finite time-market grid construction.

Implements Section 2.2.1 (domain truncation, Eq. and surrounding text) and
Section 3.2.1 ("Classical benchmark") of arXiv:2607.12990: truncating each
underlying's price domain to D_k = [mu-3*sigma, mu+3*sigma] (clipped at 0),
binning into N_k = 2^{n_k} bins, and building the joint probability tensor
P_{i,j} and positive-exposure tensor V+_{i,j} that both the classical
finite-grid benchmark and the quantum circuit consume.

SIR reference: architecture.modules "joint probability tensor P_{i,j}",
"positive exposure tensor V+_{i,j}"; tensor_semantics entries for both.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from quantum_cva.finance.classical_cva import BlackScholesPricer, Instrument


class FiniteGridBuilder:
    """Builds the finite time-market grid objects consumed by both the
    classical finite-grid CVA benchmark and the quantum encoding circuit.

    Args:
        pricer: a BlackScholesPricer instance for exposure-tensor construction.
    """

    def __init__(self, pricer: BlackScholesPricer) -> None:
        self.pricer = pricer

    def __repr__(self) -> str:  # noqa: D105
        return "FiniteGridBuilder()"

    def truncate_domain(self, mu: float, sigma: float, n_std: float = 3.0) -> Tuple[float, float]:
        """Truncated price domain D_k = [max(mu - n_std*sigma, 0), mu + n_std*sigma].

        Args:
            mu: risk-neutral mean price at the CVA horizon, mu_S,k(0,T) (Eq. 7).
            sigma: risk-neutral std of price at the CVA horizon, eta_S,k(0,T) (Eq. 8).
            n_std: truncation width in standard deviations (paper uses 3).

        Returns:
            (lower, upper) domain bounds.
        """
        return max(mu - n_std * sigma, 0.0), mu + n_std * sigma

    def bin_edges(self, lower: float, upper: float, n_bins: int) -> np.ndarray:
        """Uniform bin edges over the truncated domain (N_k = 2**n_qubits bins).

        Returns:
            Array of length n_bins+1.
        """
        return np.linspace(lower, upper, n_bins + 1)

    def build_probability_tensor(
        self, paths: np.ndarray, bin_edges_list: List[np.ndarray], monitoring_dates: np.ndarray
    ) -> np.ndarray:
        """Empirical joint probability tensor P_{i,j} (Eq. 32), built by
        counting simulated states in each joint price bin per monitoring date,
        after truncation and renormalisation.

        Args:
            paths: simulated paths, shape [n_paths, M, d].
            bin_edges_list: list of length d, each an array of bin edges for
                that underlying.
            monitoring_dates: t_1, ..., t_M (only used for M = len(...)).

        Returns:
            Tensor of shape [M, N_1, ..., N_d], summing to 1 over the whole
            tensor (uniform time marginal pi_i = 1/M is folded in, per Eq. 32).
        """
        M = len(monitoring_dates)
        d = len(bin_edges_list)
        n_bins = [len(edges) - 1 for edges in bin_edges_list]
        tensor = np.zeros([M] + n_bins)

        for i in range(M):
            bin_indices = []
            valid_mask = np.ones(paths.shape[0], dtype=bool)
            for k in range(d):
                edges = bin_edges_list[k]
                idx = np.digitize(paths[:, i, k], edges) - 1
                in_domain = (idx >= 0) & (idx < n_bins[k])
                valid_mask &= in_domain
                idx_clipped = np.clip(idx, 0, n_bins[k] - 1)
                bin_indices.append(idx_clipped)

            # Renormalise by the truncated mass Z_i (only paths within domain)
            valid_indices = tuple(bi[valid_mask] for bi in bin_indices)
            counts = np.zeros(n_bins)
            np.add.at(counts, valid_indices, 1)
            if counts.sum() > 0:
                counts = counts / counts.sum()  # conditional bin probability Q_D(S_ti in B_j)
            tensor[i] = counts / M  # uniform time marginal pi_i = 1/M folded in

        return tensor

    def build_exposure_tensor(
        self,
        instruments: List[Instrument],
        bin_edges_list: List[np.ndarray],
        underlyings: List[str],
        monitoring_dates: np.ndarray,
        r: float,
        dividend_yields: Dict[str, float],
        volatilities: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Positive-exposure tensor V+_{i,j} evaluated at each bin's left
        endpoint (per-bin convention stated in Section 2.2.1, footnote 4).

        Args:
            instruments: netting-set instruments (Table 3).
            bin_edges_list: per-underlying bin edges.
            underlyings: ordered underlying names matching bin_edges_list.
            monitoring_dates: t_1, ..., t_M.
            r: flat risk-free rate.
            dividend_yields: per-underlying q_k.
            volatilities: per-underlying sigma arrays aligned to monitoring_dates.

        Returns:
            Tensor of shape [M, N_1, ..., N_d] with V+_{i,j} = max(sum
            trades, 0).
        """
        M = len(monitoring_dates)
        n_bins = [len(edges) - 1 for edges in bin_edges_list]
        tensor = np.zeros([M] + n_bins)

        # left-endpoint bin representative prices, per underlying
        bin_left_prices = [edges[:-1] for edges in bin_edges_list]

        for i, t in enumerate(monitoring_dates):
            for j_idx in np.ndindex(*n_bins):
                total = 0.0
                for inst in instruments:
                    k = underlyings.index(inst.underlying)
                    S = bin_left_prices[k][j_idx[k]]
                    T_remaining = max(inst.maturity_years - t, 0.0)
                    if T_remaining <= 0:
                        continue
                    q = dividend_yields[inst.underlying]
                    sigma = volatilities[inst.underlying][min(i, len(volatilities[inst.underlying]) - 1)]
                    if inst.option_type == "call":
                        price = self.pricer.call_price(S, inst.strike, T_remaining, r, q, sigma)
                    elif inst.option_type == "put":
                        price = self.pricer.put_price(S, inst.strike, T_remaining, r, q, sigma)
                    else:
                        price = self.pricer.forward_price(S, inst.strike, T_remaining, r, q)
                    total += inst.sign * inst.multiplier * price
                tensor[(i,) + j_idx] = max(total, 0.0)

        return tensor

    def rescale_constants(
        self,
        exposure_tensor: np.ndarray,
        discount_vec: np.ndarray,
        default_incr_vec: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Rescaling constants C_v, C_p, C_q so that v~, p~, q~ all lie in
        [0, 1] (Eq. 34): pick each constant as (a small safety margin above)
        the maximum of the corresponding raw quantity.

        Returns:
            (C_v, C_p, C_q).
        """
        c_v = float(np.max(exposure_tensor)) * 1.0001 if np.max(exposure_tensor) > 0 else 1.0
        c_p = float(np.max(discount_vec)) * 1.0001 if np.max(discount_vec) > 0 else 1.0
        c_q = float(np.max(default_incr_vec)) * 1.0001 if np.max(default_incr_vec) > 0 else 1.0
        return c_v, c_p, c_q
