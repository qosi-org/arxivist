"""
signatures/compute.py
─────────────────────
Truncated signature computation, time augmentation, shuffle product table,
and Q-matrix assembly for the signature-based volatility model.

Paper sections:
  - Section 3:     Rough path theory and signatures
  - Section 3.4:   Time-augmented rough paths
  - Section 4.1:   Signature approximation to volatility
  - Proposition 4.2 / Eq. (4.6):  Q-matrix via shuffle products
  - Remark 4.3:    Q(T) depends on signature up to level 2N+1
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Multi-index utilities
# ─────────────────────────────────────────────────────────────────────────────

def sig_dimension(d: int, N: int) -> int:
    """
    Total number of signature coordinates for a d-dim path truncated at level N.
    = Σ_{k=0}^{N} d^k = (d^{N+1} - 1) / (d - 1)  for d > 1, else N+1.
    """
    if d == 1:
        return N + 1
    return (d ** (N + 1) - 1) // (d - 1)


def all_multi_indices(d: int, N: int) -> List[Tuple[int, ...]]:
    """
    Return all multi-indices of length 0 to N over alphabet {0,...,d-1},
    in lexicographic order by length then value.
    """
    indices: List[Tuple[int, ...]] = [()]  # empty index = level 0
    for level in range(1, N + 1):
        for idx in itertools.product(range(d), repeat=level):
            indices.append(idx)
    return indices


def build_label_maps(d: int, N: int) -> Tuple[Dict, Dict]:
    """
    Build bijection between multi-indices and integer labels (0-indexed).
    Returns:
        idx_to_label: {multi_index: int}
        label_to_idx: {int: multi_index}
    """
    indices = all_multi_indices(d, N)
    idx_to_label = {idx: i for i, idx in enumerate(indices)}
    label_to_idx = {i: idx for i, idx in enumerate(indices)}
    return idx_to_label, label_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Shuffle product
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_product(I: Tuple[int, ...], J: Tuple[int, ...]) -> Dict[Tuple[int, ...], int]:
    """
    Compute the shuffle product e_I ⊔ e_J.  (Definition 3.3)

    The shuffle product of two multi-indices I=(i₁,...,iₙ) and J=(j₁,...,jₘ)
    returns a dictionary {multi_index: coefficient} representing the linear
    combination Σ c_K e_K.

    The result lives in level |I|+|J|.

    Example (from paper, Section 3.1):
        shuffle((1,2), (3,)) = {(1,3,2): 1, (3,1,2): 1, (1,2,3): 1}

    Args:
        I, J: Multi-indices as tuples of integers.

    Returns:
        dict mapping result multi-index → integer coefficient.
    """
    if len(I) == 0:
        return {J: 1}
    if len(J) == 0:
        return {I: 1}

    # Recursive definition: eI ⊔ eJ = (eI' ⊔ eJ) ⊗ e_{iₙ} + (eI ⊔ eJ') ⊗ e_{jₘ}
    I_prime = I[:-1]
    J_prime = J[:-1]
    result: Dict[Tuple[int, ...], int] = defaultdict(int)

    # Left term: (I' ⊔ J) ⊗ iₙ
    for term, coeff in shuffle_product(I_prime, J).items():
        result[term + (I[-1],)] += coeff

    # Right term: (I ⊔ J') ⊗ jₘ
    for term, coeff in shuffle_product(I, J_prime).items():
        result[term + (J[-1],)] += coeff

    return dict(result)


def build_shuffle_table(d: int, N: int) -> Dict[Tuple, Dict[Tuple, int]]:
    """
    Precompute shuffle products for all pairs (I, J) with |I|+|J| ≤ 2N+1.
    Required for Q-matrix assembly (Eq. 4.6, Remark 4.3).

    Also appends the time-dimension index 0, i.e. computes (e_I ⊔ e_J) ⊗ e_0.

    Returns:
        table: {(I, J): {K: coeff}} where K is a multi-index of length |I|+|J|+1
               (the +1 accounts for ⊗ e_0 in the Q definition).
    """
    # All multi-indices up to level N (for the 15 signature coordinates at N=3)
    level_N_indices = all_multi_indices(d, N)
    table = {}
    for I in level_N_indices:
        for J in level_N_indices:
            shuffled = shuffle_product(I, J)
            # Append time index 0: (eI ⊔ eJ) ⊗ e_0  →  each term K becomes K+(0,)
            appended = {K + (0,): coeff for K, coeff in shuffled.items()}
            table[(I, J)] = appended
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Signature computer
# ─────────────────────────────────────────────────────────────────────────────

class SignatureComputer:
    """
    Computes truncated signatures of time-augmented paths via iterated Riemann sums.

    For a d-dimensional path X, the time-augmented path X̂ = (t, X) lives in R^{d+1}.
    With d=1 (scalar primary process), the augmented path is 2-dimensional.

    The truncated signature at level N of a 2D path has:
        Σ_{k=0}^{N} 2^k  coordinates

    For N=3:  1+2+4+8 = 15 coordinates  (Section 4.1, paper)
    For N=7:  Σ_{k=0}^{7} 2^k = 255 coordinates  (for Q-matrix assembly, Remark 4.3)

    Implementation note: Uses vectorised numpy Riemann sums over the discretised path.
    The paper states linear interpolation is used (Section 4.3), which is the default
    for piecewise-constant Euler paths.
    """

    def __init__(self, d: int, N: int):
        """
        Args:
            d: Path dimension (1 for scalar primary; 2 after time-augmentation).
            N: Truncation level.
        """
        self.d = d
        self.N = N
        self.n_coords = sig_dimension(d, N)
        self.idx_to_label, self.label_to_idx = build_label_maps(d, N)

    def __repr__(self) -> str:
        return f"SignatureComputer(d={self.d}, N={self.N}, n_coords={self.n_coords})"

    def time_augment(
        self,
        paths: np.ndarray,  # [nMC, T_steps+1]
        dt: float,
    ) -> np.ndarray:
        """
        Augment scalar paths with a time channel.  (Section 3.4)

        X̂_t = (t, X_t)  ∈  R ⊕ V

        Args:
            paths: [nMC, T_steps+1] primary process paths.
            dt:    Time step size.

        Returns:
            aug_paths: [nMC, T_steps+1, 2]  (channel 0 = time, channel 1 = X)
        """
        nMC, T_steps_plus1 = paths.shape
        time_grid = np.arange(T_steps_plus1, dtype=np.float64) * dt  # [T_steps+1]
        time_channel = np.broadcast_to(time_grid, (nMC, T_steps_plus1))  # [nMC, T+1]
        aug = np.stack([time_channel, paths], axis=-1)  # [nMC, T+1, 2]
        return aug.copy()

    def compute_signature_paths(
        self,
        aug_paths: np.ndarray,  # [nMC, T_steps+1, 2]
    ) -> np.ndarray:
        """
        Compute the truncated signature of each path at every time step.
        Returns S(X)^{≤N}_t for all t, giving a path of signature vectors.

        This is needed for the stochastic integral ∫vec(S^{≤N}_s)dZ_s.

        Uses iterated Riemann sums on the linearly-interpolated path.
        Implements Definition 3.7 (signature via iterated integrals).

        Args:
            aug_paths: [nMC, T_steps+1, 2] time-augmented paths.

        Returns:
            sig_paths: [nMC, T_steps+1, n_coords] signature paths.
                       sig_paths[:, t, :] = S(X)^{≤N}_{0,t}.
        """
        assert aug_paths.ndim == 3 and aug_paths.shape[2] == self.d, (
            f"Expected [nMC, T+1, {self.d}], got {aug_paths.shape}"
        )
        nMC, T_plus1, d = aug_paths.shape
        T = T_plus1 - 1
        sig = np.zeros((nMC, T_plus1, self.n_coords), dtype=np.float64)
        # Level 0: always 1
        sig[:, :, 0] = 1.0

        # Increments: dX_t = X_{t+1} - X_t, shape [nMC, T, d]
        dX = np.diff(aug_paths, axis=1)  # [nMC, T, d]

        # Build signature level by level using Chen's identity (recursive Riemann sums)
        # We track the "running signature up to level k" for each time step.
        # sig[:, t, L(I)] = ∫_{0<t1<...<tk<t} dX^{i1}_{t1}...dX^{ik}_{tk}

        # Level 1: S^i_{0,t} = X^i_t - X^i_0
        for i in range(d):
            lbl = self.idx_to_label[(i,)]
            sig[:, 1:, lbl] = np.cumsum(dX[:, :, i], axis=1)

        # Levels 2..N: S^{I+(i)}_{0,t} = ∫_0^t S^I_{0,s} dX^i_s
        #              ≈ Σ_{j=0}^{t-1} S^I_{0,t_j} · ΔX^i_j  (left-point Riemann sum)
        for level in range(2, self.N + 1):
            prev_level_indices = [
                idx for idx in self.idx_to_label if len(idx) == level - 1
            ]
            for I in prev_level_indices:
                lbl_I = self.idx_to_label[I]
                for i in range(d):
                    I_new = I + (i,)
                    lbl_new = self.idx_to_label[I_new]
                    # Riemann sum: cumulative sum of S^I_{0,t_j} * dX^i_j
                    # sig[:, j+1, lbl_new] += sig[:, j, lbl_I] * dX[:, j, i]
                    increments = sig[:, :-1, lbl_I] * dX[:, :, i]  # [nMC, T]
                    sig[:, 1:, lbl_new] = np.cumsum(increments, axis=1)

        return sig

    def compute_terminal_signature(
        self,
        aug_paths: np.ndarray,  # [nMC, T_steps+1, 2]
    ) -> np.ndarray:
        """
        Compute the truncated signature at the terminal time T only.
        More memory-efficient than compute_signature_paths when only S^{≤N}_T is needed.

        Returns:
            sig_T: [nMC, n_coords]
        """
        sig_paths = self.compute_signature_paths(aug_paths)
        return sig_paths[:, -1, :]  # [nMC, n_coords]


# ─────────────────────────────────────────────────────────────────────────────
# Q-matrix assembler
# ─────────────────────────────────────────────────────────────────────────────

class QMatrixAssembler:
    """
    Assembles the 15×15 matrix Q(T) from extended signatures via shuffle products.

    From Proposition 4.2 and Eq. (4.6):
        Q(T)_{L(I),L(J)} = -½ <(e_I ⊔ e_J) ⊗ e_0,  S(X)^{≤2N+1}_T>

    where N=3 (signature truncation level) and the extended signature S^{≤7}
    lives in a 255-dimensional space.  (Remark 4.3)

    The shuffle table is precomputed once at construction (Risk R1 mitigation).
    """

    def __init__(self, N: int = 3, d: int = 2):
        """
        Args:
            N: Signature truncation level (paper: N=3).
            d: Path dimension after time augmentation (always 2 for this paper).
        """
        self.N = N
        self.d = d
        self.n_coords = sig_dimension(d, N)         # 15 for d=2, N=3
        self.ext_N = 2 * N + 1                      # 7
        self.n_ext_coords = sig_dimension(d, self.ext_N)  # 255 for d=2, N_ext=7

        # Label maps for both levels
        self.idx_to_lbl_N, self.lbl_to_idx_N = build_label_maps(d, N)
        self.idx_to_lbl_ext, _ = build_label_maps(d, self.ext_N)

        # Precompute shuffle table: {(I, J): {K: coeff}} where K has length |I|+|J|+1
        print(f"[QMatrixAssembler] Precomputing shuffle table for N={N}, d={d}...")
        self._shuffle_table = build_shuffle_table(d, N)
        print(f"[QMatrixAssembler] Shuffle table ready ({len(self._shuffle_table)} pairs).")

    def __repr__(self) -> str:
        return (f"QMatrixAssembler(N={self.N}, d={self.d}, "
                f"Q_shape=[{self.n_coords},{self.n_coords}], "
                f"ext_sig_dim={self.n_ext_coords})")

    def assemble(
        self,
        ext_sig: np.ndarray,  # [nMC, 255]
    ) -> np.ndarray:
        """
        Assemble Q(T) from extended signatures.  (Eq. 4.6)

        Q(T)_{L(I),L(J)} = -½ <(e_I ⊔ e_J) ⊗ e_0,  S(X)^{≤2N+1}_T>
                         = -½ Σ_K c^{IJ}_K · S^K_T

        where the sum is over all K appearing in the shuffle expansion,
        and c^{IJ}_K are the shuffle coefficients.

        Args:
            ext_sig: [nMC, n_ext_coords] extended terminal signatures S^{≤2N+1}_T.

        Returns:
            Q: [nMC, n_coords, n_coords] symmetric matrix (negative semi-definite).
        """
        assert ext_sig.shape[1] == self.n_ext_coords, (
            f"Extended signature dimension mismatch: "
            f"expected {self.n_ext_coords}, got {ext_sig.shape[1]}"
        )
        nMC = ext_sig.shape[0]
        Q = np.zeros((nMC, self.n_coords, self.n_coords), dtype=np.float64)

        all_N_idx = [idx for idx in self.idx_to_lbl_N]

        for I in all_N_idx:
            lI = self.idx_to_lbl_N[I]
            for J in all_N_idx:
                lJ = self.idx_to_lbl_N[J]
                shuffle_expansion = self._shuffle_table[(I, J)]
                val = np.zeros(nMC, dtype=np.float64)
                for K, coeff in shuffle_expansion.items():
                    if K in self.idx_to_lbl_ext:
                        lK = self.idx_to_lbl_ext[K]
                        val += coeff * ext_sig[:, lK]
                    # If K exceeds ext level, contribution is zero (truncation)
                Q[:, lI, lJ] = -0.5 * val

        return Q  # [nMC, 15, 15]

    def cholesky(
        self,
        Q: np.ndarray,  # [nMC, 15, 15]
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute Cholesky factorisation of -Q(T) per path.  (Section 4.1)

        Since Q is negative semi-definite, -Q is positive semi-definite.
        A small diagonal regularisation eps·I is added for numerical stability.  (Risk R7)

        Returns:
            U: [nMC, 15, 15] upper-triangular Cholesky factors such that
               -Q[j] ≈ U[j]ᵀ U[j]
        """
        neg_Q = -Q  # positive semi-definite
        n = neg_Q.shape[1]
        neg_Q += eps * np.eye(n)[None, :, :]  # [nMC, 15, 15]

        U = np.zeros_like(neg_Q)
        fail_count = 0
        for j in range(neg_Q.shape[0]):
            try:
                U[j] = np.linalg.cholesky(neg_Q[j]).T  # upper triangular
            except np.linalg.LinAlgError:
                fail_count += 1
                # Fallback: increase regularisation
                U[j] = np.linalg.cholesky(neg_Q[j] + 1e-6 * np.eye(n)).T
        if fail_count > 0:
            print(f"[QMatrixAssembler.cholesky] WARNING: {fail_count}/{neg_Q.shape[0]} "
                  "paths needed extra regularisation.")
        return U
