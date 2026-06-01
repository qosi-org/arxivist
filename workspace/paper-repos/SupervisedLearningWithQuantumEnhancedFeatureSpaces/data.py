"""
data.py — Synthetic Quantum Dataset

Generates the artificial classification dataset used in Havlicek et al. (2018),
Section "The data" and Supplementary "Classification problems".

Label generation rule  [EQ14]:
    m(x) = +1  if  ⟨Φ(x)| V† (Z_1 Z_2) V |Φ(x)⟩ ≥ Δ
    m(x) = −1  if  ⟨Φ(x)| V† (Z_1 Z_2) V |Φ(x)⟩ ≤ −Δ
    (reject)   if  |value| < Δ   (white regions in Fig. 3b)

where:
  - V ∈ SU(4) is a fixed random unitary (seed-controlled)  [ASSUMED conf=0.80]
  - Z_1 Z_2 is the parity operator (diagonal: +1 for even parity, -1 for odd)
  - Δ = 0.3 is the separation gap  [paper]
  - x ∈ (0, 2π]²  [paper domain]

NOTE: The specific V used in the paper is not published. Any seeded V produces
a valid, 100%-separable dataset. The visual decision boundary will differ from
Fig. 3b but the algorithmic performance will be identical.

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import unitary_group

from qsvm.feature_map import FeatureMap


class SyntheticQuantumDataset:
    """
    Synthetic classification dataset based on quantum feature map geometry.

    The dataset is perfectly separable by the quantum feature map (by construction),
    so a correctly implemented QVC/QKE should achieve 100% success.

    Parameters
    ----------
    n_per_label : int
        Number of accepted data points per label class. Paper uses 20.
    gap : float
        Separation gap Δ. Paper uses 0.3.
    seed : int
        Master random seed. Controls both V ∈ SU(4) and data point sampling.
    n_qubits : int
        Must match feature_map.n_qubits. Paper uses 2.
    domain_min : float
        Lower bound of data domain (exclusive). Paper: (0, 2π].
    domain_max : float
        Upper bound of data domain (inclusive). Paper: (0, 2π] = (0, 6.2832].
    """

    def __init__(
        self,
        n_per_label: int = 20,
        gap: float = 0.3,
        seed: int = 42,
        n_qubits: int = 2,
        domain_min: float = 0.0001,
        domain_max: float = 6.2832,
    ) -> None:
        if gap <= 0 or gap > 1:
            raise ValueError(f"gap must be in (0,1], got {gap}")
        if n_per_label < 1:
            raise ValueError(f"n_per_label must be >= 1, got {n_per_label}")

        self.n_per_label = n_per_label
        self.gap = gap
        self.seed = seed
        self.n_qubits = n_qubits
        self.domain_min = domain_min
        self.domain_max = domain_max

        # Instantiate feature map for statevector computation
        self._feature_map = FeatureMap(n_qubits=n_qubits, reps=2)

        # Fixed V ∈ SU(4)  [ASSUMED: seeded for reproducibility — conf=0.80]
        self._V = self._sample_unitary(seed)

        # Pre-build Z1Z2 parity operator matrix (diagonal)
        self._Z1Z2 = self._build_parity_operator()

        # Transformed operator V† Z1Z2 V
        self._VdagZ1Z2V = self._V.conj().T @ self._Z1Z2 @ self._V

        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the full dataset via rejection sampling.

        Samples x ∈ (domain_min, domain_max]² uniformly; accepts if
        |⟨Φ(x)| V† Z₁Z₂ V |Φ(x)⟩| ≥ gap  [EQ14].

        Returns
        -------
        X : np.ndarray, shape [2*n_per_label, n_qubits]
            Data points in (0, 2π]².
        y : np.ndarray, shape [2*n_per_label], dtype int
            Labels in {+1, -1}.
        """
        rng = np.random.default_rng(self.seed + 1)   # separate from V seed

        pos_samples: list = []
        neg_samples: list = []

        max_attempts = self.n_per_label * 1000
        attempts = 0

        while (
            len(pos_samples) < self.n_per_label or len(neg_samples) < self.n_per_label
        ) and attempts < max_attempts:
            x = rng.uniform(self.domain_min, self.domain_max, size=self.n_qubits)
            label = self.label_point(x)
            if label == +1 and len(pos_samples) < self.n_per_label:
                pos_samples.append(x)
            elif label == -1 and len(neg_samples) < self.n_per_label:
                neg_samples.append(x)
            attempts += 1

        if len(pos_samples) < self.n_per_label or len(neg_samples) < self.n_per_label:
            raise RuntimeError(
                f"Could not generate enough samples after {max_attempts} attempts. "
                f"Try reducing gap (currently {self.gap}) or increasing domain."
            )

        X_pos = np.array(pos_samples)
        X_neg = np.array(neg_samples)
        X = np.vstack([X_pos, X_neg])
        y = np.array([+1] * self.n_per_label + [-1] * self.n_per_label)

        self._X = X
        self._y = y
        return X, y

    def label_point(self, x: np.ndarray) -> Optional[int]:
        """
        Compute label for a single data point using  [EQ14].

        Expectation value:  ⟨Φ(x)| V† (Z₁Z₂) V |Φ(x)⟩

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]

        Returns
        -------
        int : +1, -1, or None (rejected — white region in Fig. 3b)
        """
        assert x.shape == (self.n_qubits,), (
            f"Expected x shape [{self.n_qubits}], got {x.shape}"
        )

        # Statevector |Φ(x)⟩  — shape [2^n]
        sv = self._feature_map.get_statevector(x)

        # Expectation value ⟨Φ(x)| V† Z₁Z₂ V |Φ(x)⟩  [EQ14]
        # = sv† @ VdagZ1Z2V @ sv   (matrix element)
        val = float(np.real(sv.conj() @ self._VdagZ1Z2V @ sv))

        if val >= self.gap:
            return +1
        elif val <= -self.gap:
            return -1
        else:
            return None   # rejected (white region in Fig. 3b)

    def split(
        self,
        test_n_per_label: int = 20,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate train and test splits as disjoint sets.

        The training set is self.generate() (n_per_label each).
        The test set is freshly drawn with a different seed.

        Parameters
        ----------
        test_n_per_label : int
            Test set size per label. Paper uses 20.
        seed : int, optional
            Seed for test set. If None, uses self.seed + 999.

        Returns
        -------
        X_train, y_train, X_test, y_test
        """
        if self._X is None:
            self.generate()

        test_seed = seed if seed is not None else self.seed + 999
        test_ds = SyntheticQuantumDataset(
            n_per_label=test_n_per_label,
            gap=self.gap,
            seed=test_seed,
            n_qubits=self.n_qubits,
            domain_min=self.domain_min,
            domain_max=self.domain_max,
        )
        # Use same V (same V_seed = self.seed)
        test_ds._V = self._V
        test_ds._VdagZ1Z2V = self._VdagZ1Z2V
        X_test, y_test = test_ds.generate()

        return self._X, self._y, X_test, y_test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_unitary(self, seed: int) -> np.ndarray:
        """
        Sample V ∈ SU(4) using scipy Haar-random unitary group.

        ASSUMED: fixed seed — conf=0.80. Paper's specific V is not published.
        The decision boundary shape will differ from Fig. 3b but the dataset
        remains 100%-separable by the feature map.
        """
        dim = 2 ** self.n_qubits   # 4 for n_qubits=2
        # scipy.stats.unitary_group draws from U(dim); normalise to SU(dim)
        U = unitary_group.rvs(dim, random_state=seed)
        # Normalise determinant to 1 (SU vs U)
        det = np.linalg.det(U)
        V = U / (det ** (1.0 / dim))
        return V.astype(complex)

    def _build_parity_operator(self) -> np.ndarray:
        """
        Build Z_1 ⊗ Z_2 diagonal matrix in the computational basis.

        Z_1 Z_2 eigenvalue for basis state |z_1 z_2⟩ is (-1)^(z_1 + z_2).
        For n_qubits=2, dim=4: diag = [+1, -1, -1, +1]
           |00⟩ → +1,  |01⟩ → -1,  |10⟩ → -1,  |11⟩ → +1
        """
        dim = 2 ** self.n_qubits
        diag = np.zeros(dim)
        for i in range(dim):
            bitstring = format(i, f"0{self.n_qubits}b")
            parity = sum(int(b) for b in bitstring) % 2
            diag[i] = 1 - 2 * parity   # (-1)^parity
        return np.diag(diag).astype(complex)

    def __repr__(self) -> str:
        return (
            f"SyntheticQuantumDataset("
            f"n_per_label={self.n_per_label}, "
            f"gap={self.gap}, "
            f"seed={self.seed}, "
            f"n_qubits={self.n_qubits})"
        )
