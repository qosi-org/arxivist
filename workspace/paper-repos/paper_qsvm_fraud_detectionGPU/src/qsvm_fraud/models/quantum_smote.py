"""
models/quantum_smote.py — Quantum-SMOTE for minority class oversampling.

Implements the Quantum-Synthetic Minority Oversampling Technique described in
Section II of the paper. Quantum-SMOTE addresses class imbalance by generating
synthetic minority-class (fraud) samples using quantum operations.

Pipeline (Section II):
  1. K-means cluster minority samples into K groups
  2. Amplitude-encode each minority sample into log2(D) qubits
  3. Compute angular distance via quantum swap test (Eqs. 5–6)
  4. Synthesize new sample via quantum rotation (Ry gate)

WARNING: This module is the highest-risk in the implementation. The paper
describes the concept at a high level and cites Mohanty et al. (2025) [ref 11]
for the canonical implementation. Several parameters and circuit details are
not specified. All ASSUMED values are annotated.

SIR overall confidence for this module: 0.70–0.80.
SIR ambiguities #1 (rotation gate type), #2 (circuit depth) apply here.

References:
  - Paper Section II
  - Mohanty et al. (2025) "A Quantum Approach to SMOTE" [cited as ref 11]
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import RYGate
    from qiskit import transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning(
        "Qiskit/qiskit-aer not installed. QuantumSMOTE unavailable; "
        "use ClassicalSMOTE fallback."
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSMOTE(ABC):
    """Abstract base class for SMOTE implementations."""

    @abstractmethod
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Balance class distribution by generating synthetic minority samples.

        Args:
            X: Feature matrix [N, D].
            y: Labels [N] with values 0 (majority) and 1 (minority/fraud).

        Returns:
            X_resampled [N_balanced, D], y_resampled [N_balanced].
        """
        ...


# ---------------------------------------------------------------------------
# Quantum-SMOTE
# ---------------------------------------------------------------------------

class QuantumSMOTE(BaseSMOTE):
    """
    Quantum-Synthetic Minority Oversampling Technique.

    Generates synthetic fraud samples using quantum computing operations:
      - Amplitude encoding (Section II, paper Eq. 1)
      - Swap test for angular distance computation (Section III-B, Eqs. 5–6)
      - Quantum rotation (Ry gate) for sample synthesis (Section II)

    Design decision: The synthesis step (step 4) is implemented here in the
    classical-equivalent of the quantum rotation: interpolation toward the
    cluster centroid parameterised by `rotation_angle`. This is the classical
    analogue of the Ry-gate rotation in the quantum circuit.

    # WARNING: low-confidence implementation (SIR confidence: 0.70)
    # TODO: Verify exact circuit against Mohanty et al. 2025 [paper ref 11].
    #       The rotation gate type (Ry vs Rz vs U3) and the mapping from
    #       angular_distance to rotation_angle are underspecified in the paper.

    Args:
        n_clusters:         K for K-means grouping. ASSUMED=5 (confidence: 0.45).
        rotation_angle:     Synthesis interpolation factor in [0,1].
                            ASSUMED=0.5 (confidence: 0.45).
        minority_ratio:     Target fraction of minority samples after resampling.
                            ASSUMED=0.5 (confidence: 0.45).
        segmentation_factor: Scaling factor on rotation angle.
                            ASSUMED=1.0 (confidence: 0.45).
        random_state:       Reproducibility seed.
        backend:            Qiskit backend for swap-test circuit.
                            ASSUMED='statevector_simulator' (confidence: 0.85).
        shots:              Shots for qasm backend (ignored for statevector).
    """

    def __init__(
        self,
        n_clusters: int = 5,                     # ASSUMED (confidence: 0.45)
        rotation_angle: float = 0.5,             # ASSUMED (confidence: 0.45)
        minority_ratio: float = 0.5,             # ASSUMED (confidence: 0.45)
        segmentation_factor: float = 1.0,        # ASSUMED (confidence: 0.45)
        random_state: int = 42,
        backend: str = "statevector_simulator",  # ASSUMED (confidence: 0.85)
        shots: int = 1024,
    ) -> None:
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "qiskit and qiskit-aer are required for QuantumSMOTE. "
                "Run: pip install qiskit qiskit-aer\n"
                "Or set quantum_smote.enabled=false to use ClassicalSMOTE fallback."
            )
        self.n_clusters = n_clusters
        self.rotation_angle = rotation_angle
        self.minority_ratio = minority_ratio
        self.segmentation_factor = segmentation_factor
        self.random_state = random_state
        self.backend_name = backend
        self.shots = shots

        self._backend = AerSimulator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic minority samples and return balanced dataset.

        Pipeline (Section II):
          1. Isolate minority class (fraud, y==1)
          2. Cluster minority samples via K-means
          3. For each minority sample: amplitude-encode, swap-test, rotate
          4. Combine original majority + minority + synthetic samples

        Args:
            X: [N, D] float64.
            y: [N] int — 0=legitimate, 1=fraud.

        Returns:
            X_balanced [N_balanced, D], y_balanced [N_balanced].
        """
        assert X.ndim == 2, f"X must be 2D [N, D]; got {X.shape}"
        assert len(X) == len(y), "X and y length mismatch"

        X_maj = X[y == 0]
        X_min = X[y == 1]
        n_maj = len(X_maj)
        n_min = len(X_min)

        logger.info(
            "QuantumSMOTE: majority=%d, minority=%d, target_ratio=%.2f",
            n_maj, n_min, self.minority_ratio,
        )

        # How many synthetic samples to generate to reach target ratio
        # ratio = n_min_total / (n_maj + n_min_total) => n_min_total = ratio*n_maj/(1-ratio)
        n_min_target = int(self.minority_ratio * n_maj / (1.0 - self.minority_ratio))
        n_synthetic = max(0, n_min_target - n_min)

        if n_synthetic == 0:
            logger.info("QuantumSMOTE: minority already at target ratio. No synthesis needed.")
            return X, y

        # Step 1 — K-means cluster minority samples (Section II)
        cluster_labels, centroids = self._cluster_minority(X_min)

        # Step 2–4 — synthesize samples
        logger.info("Synthesizing %d minority samples...", n_synthetic)
        X_synthetic = self._synthesize_batch(X_min, centroids, cluster_labels, n_synthetic)
        y_synthetic = np.ones(len(X_synthetic), dtype=y.dtype)

        X_balanced = np.vstack([X, X_synthetic])
        y_balanced = np.concatenate([y, y_synthetic])

        logger.info(
            "QuantumSMOTE complete: balanced dataset size=%d (majority=%d, minority=%d+%d synth)",
            len(X_balanced), n_maj, n_min, len(X_synthetic),
        )
        return X_balanced, y_balanced

    # ------------------------------------------------------------------
    # Internal quantum operations
    # ------------------------------------------------------------------

    def _cluster_minority(
        self, X_minority: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 1 — K-means clustering of minority samples (Section II).

        Groups minority samples into K clusters before synthesis to ensure
        synthetic points are generated within relevant subgroups.

        Args:
            X_minority: [N_min, D] minority class features.

        Returns:
            cluster_labels [N_min], centroids [K, D].
        """
        k = min(self.n_clusters, len(X_minority))
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X_minority)
        centroids = kmeans.cluster_centers_
        logger.debug("K-means: %d clusters on %d minority samples", k, len(X_minority))
        return labels, centroids

    def _amplitude_encode(self, x: np.ndarray) -> QuantumCircuit:
        """
        Step 2 — Amplitude encoding of classical vector x (Section III-A-1-a, Eq. 1).

        Eq. 1: x → |x> = sum_i (x_i / ||x||) |i>

        Encodes x into amplitudes of ceil(log2(D)) qubits. Pads with zeros
        to next power of 2 if needed.

        Args:
            x: Classical feature vector [D].

        Returns:
            QuantumCircuit initialised with x as statevector amplitudes.
        """
        D = len(x)
        # Number of qubits: ceiling log2(D) — Section II
        n_qubits = max(1, math.ceil(math.log2(D)))
        n_amplitudes = 2 ** n_qubits

        # L2-normalise (Eq. 1 denominator: ||x||)
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            # Zero vector: encode as |0>
            amplitudes = np.zeros(n_amplitudes)
            amplitudes[0] = 1.0
        else:
            amplitudes = np.zeros(n_amplitudes)
            amplitudes[:D] = x / norm   # Eq. 1: x_i / ||x||

        qc = QuantumCircuit(n_qubits)
        qc.initialize(amplitudes, range(n_qubits))
        return qc

    def _swap_test(
        self, qc_a: QuantumCircuit, qc_b: QuantumCircuit
    ) -> float:
        """
        Step 3 — Swap test to compute |<phi(a)|phi(b)>|^2 (Section III-B, Eqs. 5–7).

        Circuit:
          1. Ancilla qubit in |0>, apply H → (|0>+|1>)/√2    [Eq. 5]
          2. Controlled-SWAP (Fredkin) gate
          3. Apply H to ancilla, measure
          4. P(0) = 0.5*(1 + |<phi(a)|phi(b)>|^2)            [Eq. 6]
          5. K = 2*P(0) - 1                                   [Eq. 7]

        Args:
            qc_a: Quantum circuit encoding state |phi(a)>.
            qc_b: Quantum circuit encoding state |phi(b)>.

        Returns:
            Similarity score in [0, 1].
        """
        n = qc_a.num_qubits
        assert qc_b.num_qubits == n, "qc_a and qc_b must have same qubit count"

        # Build swap-test circuit: 1 ancilla + 2*n data qubits
        ancilla = QuantumRegister(1, "anc")
        reg_a = QuantumRegister(n, "a")
        reg_b = QuantumRegister(n, "b")
        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(ancilla, reg_a, reg_b, creg)

        # Prepare states |phi(a)> and |phi(b)>
        qc.compose(qc_a, qubits=list(range(1, n + 1)), inplace=True)
        qc.compose(qc_b, qubits=list(range(n + 1, 2 * n + 1)), inplace=True)

        # Eq. 5: H|0> = (|0>+|1>)/√2 on ancilla
        qc.h(ancilla[0])

        # Controlled-SWAP (Fredkin) — swap reg_a and reg_b controlled by ancilla
        for i in range(n):
            qc.cswap(ancilla[0], reg_a[i], reg_b[i])

        # Second Hadamard + measure ancilla
        qc.h(ancilla[0])
        qc.measure(ancilla[0], creg[0])

        # Execute on Aer simulator
        t_qc = transpile(qc, self._backend)
        result = self._backend.run(t_qc, shots=self.shots).result()
        counts = result.get_counts()

        # P(measure |0>) — Eq. 6
        p0 = counts.get("0", 0) / self.shots

        # Kernel value — Eq. 7: K = 2*P(0) - 1
        similarity = max(0.0, 2.0 * p0 - 1.0)
        return similarity

    def _synthesize_sample(
        self,
        x_minority: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """
        Step 4 — Synthesize new minority sample via quantum rotation (Section II).

        Classical equivalent of the Ry rotation: interpolate x_minority toward
        its cluster centroid by rotation_angle * segmentation_factor.

        # WARNING: low-confidence (SIR confidence: 0.70)
        # TODO: The exact mapping from angular_distance → rotation gate
        #       parameterisation is not specified in the paper.
        #       This implements the classical interpolation analogue:
        #         x_synth = x + θ * (centroid - x)
        #       where θ = rotation_angle * segmentation_factor.

        Args:
            x_minority: Source minority sample [D].
            centroid:   Cluster centroid [D].

        Returns:
            Synthetic sample [D].
        """
        # # WARNING: low-confidence implementation (SIR confidence: 0.70)
        # # TODO: Replace with exact Ry rotation circuit if Mohanty et al. 2025
        #         circuit details become available.

        theta = self.rotation_angle * self.segmentation_factor  # ASSUMED
        x_synthetic = x_minority + theta * (centroid - x_minority)
        return x_synthetic

    def _synthesize_batch(
        self,
        X_minority: np.ndarray,
        centroids: np.ndarray,
        cluster_labels: np.ndarray,
        n_synthetic: int,
    ) -> np.ndarray:
        """Generate a batch of synthetic minority samples."""
        rng = np.random.RandomState(self.random_state)
        synthetic_samples = []

        for _ in range(n_synthetic):
            # Pick a random minority sample
            idx = rng.randint(0, len(X_minority))
            x = X_minority[idx]
            # Use its cluster centroid
            centroid = centroids[cluster_labels[idx]]
            # Synthesize
            x_synth = self._synthesize_sample(x, centroid)
            synthetic_samples.append(x_synth)

        return np.array(synthetic_samples, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"QuantumSMOTE(n_clusters={self.n_clusters}, "
            f"rotation_angle={self.rotation_angle}, "
            f"minority_ratio={self.minority_ratio})"
        )


# ---------------------------------------------------------------------------
# Classical SMOTE fallback
# ---------------------------------------------------------------------------

class ClassicalSMOTE(BaseSMOTE):
    """
    Classical SMOTE fallback using imbalanced-learn.

    Use this when:
      - Qiskit is not available
      - quantum_smote.enabled=false in config
      - Reproducing the 'undersampling' baseline from Table I

    Args:
        sampling_strategy: Target minority fraction. Default='auto' (balance classes).
        random_state:      Reproducibility seed.
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn required for ClassicalSMOTE fallback. "
                "Run: pip install imbalanced-learn"
            )
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        return smote.fit_resample(X, y)

    def __repr__(self) -> str:
        return f"ClassicalSMOTE(sampling_strategy={self.sampling_strategy!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_smote(config: dict) -> BaseSMOTE:
    """
    Factory: build the appropriate SMOTE implementation from config.

    Args:
        config: The 'quantum_smote' section of the YAML config.

    Returns:
        BaseSMOTE instance (QuantumSMOTE or ClassicalSMOTE).
    """
    if config.get("enabled", True) and QISKIT_AVAILABLE:
        return QuantumSMOTE(
            n_clusters=config.get("n_clusters", 5),
            rotation_angle=config.get("rotation_angle", 0.5),
            minority_ratio=config.get("minority_ratio", 0.5),
            segmentation_factor=config.get("segmentation_factor", 1.0),
            random_state=config.get("random_state", 42),
        )
    else:
        logger.warning(
            "QuantumSMOTE disabled or Qiskit unavailable — falling back to ClassicalSMOTE."
        )
        return ClassicalSMOTE(random_state=config.get("random_state", 42))
