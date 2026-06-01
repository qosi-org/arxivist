"""
variational_classifier.py — Quantum Variational Classifier (QVC)

Implements Protocol 1 from Havlicek et al. (2018), Section "Quantum variational
classification". The classifier consists of:

  1. Feature map 𝒰_Φ(x) encoding data into quantum state [EQ2]
  2. Variational circuit W(θ) as trainable separating hyperplane [EQ6]
  3. Parity measurement f = Z_1 Z_2 → labels {+1, −1} [EQ6]
  4. SPSA optimisation of empirical risk R_emp(θ) [EQ7, EQ8]

Circuit structure for W(θ)  [Supp. EQ31-33]:
    W(θ) = U^(l)_loc(θ_l) U_ent ... U^(2)_loc(θ_2) U_ent U^(1)_loc(θ_1)
    U^(t)_loc(θ_t) = ⊗_m exp(i/2 θ^z_{m,t} Z_m) exp(i/2 θ^y_{m,t} Y_m)
    U_ent = Π_{(i,j)∈E} CZ(i,j)   [linear chain for n=2: CZ(0,1)]

Misclassification cost uses sigmoid approximation to binomial CDF [EQ8]:
    Pr(ỹ ≠ y) ≈ sigmoid(√R · ((1+yb)/2 - p̂_y) / √(2(1-p̂_y)p̂_y))

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from tqdm import tqdm

from qsvm.feature_map import FeatureMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _parity(bitstring: str) -> int:
    """
    Parity function f(z) = (-1)^(Σ z_i) ∈ {+1, -1}.

    Implements the boolean function f = Z_1 Z_2 used as measurement [EQ6].
    Qiskit returns bitstrings in little-endian order (rightmost = qubit 0).
    """
    return 1 - 2 * (bitstring.count("1") % 2)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumVariationalClassifier:
    """
    Quantum Variational Classifier (QVC) — Protocol 1 of Havlicek et al. (2018).

    Parameters
    ----------
    feature_map : FeatureMap
        Feature map instance 𝒰_Φ(x).
    depth : int
        Number of W(θ) entangling layers l ∈ {0,1,2,3,4}.
        depth=0 means one U_loc layer only (no CZ gates). Paper tests all 5.
    backend : AerSimulator
        Qiskit Aer backend for circuit execution.
    shots : int
        Default shots for inference. Training uses spsa.shots_cost/shots_eval.
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        depth: int,
        backend: Optional[AerSimulator] = None,
        shots: int = 1024,
    ) -> None:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        self.feature_map = feature_map
        self.depth = depth
        self.shots = shots

        if backend is None:
            self.backend = AerSimulator(method="automatic")
        else:
            self.backend = backend

        # Parameter dimensionality: 2 angles (θ^y, θ^z) per qubit, per layer
        # W has (depth+1) local layers → 2 * n_qubits * (depth+1) angles
        n = self.feature_map.n_qubits
        self.n_theta = 2 * n * (depth + 1)   # [Supp. EQ32]

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_variational_circuit(self, theta: np.ndarray) -> QuantumCircuit:
        """
        Build W(θ) circuit (no feature map, no measurement).

        W(θ) = U^(l)_loc(θ_l) U_ent ... U^(1)_loc(θ_1)   [Supp. EQ31]

        Parameters
        ----------
        theta : np.ndarray, shape [n_theta]

        Returns
        -------
        QuantumCircuit (n_qubits wide, no classical registers)
        """
        assert theta.shape == (self.n_theta,), (
            f"Expected theta shape [{self.n_theta}], got {theta.shape}"
        )
        n = self.feature_map.n_qubits
        qc = QuantumCircuit(n, name=f"W(d={self.depth})")

        # theta is laid out as [θ^y_0,t=1, θ^z_0,t=1, θ^y_1,t=1, θ^z_1,t=1, ..., layer2, ...]
        idx = 0
        for layer in range(self.depth + 1):
            # U^(t)_loc: per-qubit SU(2) rotation  [Supp. EQ32]
            for q in range(n):
                theta_y = theta[idx];     idx += 1
                theta_z = theta[idx];     idx += 1
                # U(θ_m,t) = exp(i/2 θ^z Z) exp(i/2 θ^y Y)
                qc.ry(theta_y, q)   # exp(i/2 θ^y Y)  [Supp. EQ32]
                qc.rz(theta_z, q)   # exp(i/2 θ^z Z)  [Supp. EQ32]

            # U_ent = CZ along linear chain (not applied after final U_loc) [Supp. EQ33]
            if layer < self.depth:
                for q in range(n - 1):
                    qc.cz(q, q + 1)   # CZ(q, q+1)  [Supp. EQ33]

        return qc

    def build_full_circuit(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        measure: bool = True,
    ) -> QuantumCircuit:
        """
        Full classification circuit: 𝒰_Φ(x) followed by W(θ).

        Optionally appends Z-basis measurements on all qubits.

        Parameters
        ----------
        x : np.ndarray, shape [n_qubits]
        theta : np.ndarray, shape [n_theta]
        measure : bool

        Returns
        -------
        QuantumCircuit
        """
        n = self.feature_map.n_qubits
        phi_circ = self.feature_map.get_circuit(x)   # 𝒰_Phi(x)
        w_circ = self.build_variational_circuit(theta)  # W(θ)

        if measure:
            qc = QuantumCircuit(n, n, name="QVC")
        else:
            qc = QuantumCircuit(n, name="QVC_nomeas")

        qc.compose(phi_circ, inplace=True)
        qc.compose(w_circ, inplace=True)

        if measure:
            qc.measure(range(n), range(n))

        return qc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_probs(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        b: float = 0.0,
        shots: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Estimate p̂_y(x) = Pr(measure label y | x, θ)  [EQ6].

        Uses parity measurement f(z) = Z_1 Z_2 to map bitstrings → labels.

        Parameters
        ----------
        x : np.ndarray
        theta : np.ndarray
        b : float
            Bias (not used in probability estimation, kept for API clarity).
        shots : int, optional
            Override self.shots.

        Returns
        -------
        Dict {+1: p̂_+1, -1: p̂_-1}
        """
        n_shots = shots if shots is not None else self.shots
        qc = self.build_full_circuit(x, theta, measure=True)

        job = self.backend.run(qc, shots=n_shots)
        counts = job.result().get_counts(qc)

        # Accumulate counts by parity label  [EQ6, "parity function f = Z1Z2"]
        label_counts: Dict[int, int] = {+1: 0, -1: 0}
        for bitstring, count in counts.items():
            label = _parity(bitstring)
            label_counts[label] += count

        # Empirical distribution p̂_y(x) = r_y / R  [Alg. 1, step 14]
        p_plus  = label_counts[+1] / n_shots
        p_minus = label_counts[-1] / n_shots
        return {+1: p_plus, -1: p_minus}

    def predict(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        b: float = 0.0,
        shots: Optional[int] = None,
    ) -> int:
        """
        Assign label via decision rule  [EQ9]:
            ỹ = y  if  p̂_y > p̂_{-y} − y·b

        Parameters
        ----------
        x, theta, b : see get_probs
        shots : int, optional

        Returns
        -------
        int : +1 or -1
        """
        probs = self.get_probs(x, theta, b, shots)
        # Decision rule [EQ9]: ỹ = +1 if p̂_+1 > p̂_-1 - b
        # Equivalently: ỹ = +1 if p̂_+1 + b > p̂_-1 (rearranged for y=+1)
        # b*=0 at inference per paper ("b*=0 used in final classification")
        if probs[+1] + b > probs[-1]:
            return +1
        return -1

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        theta: np.ndarray,
        b: float = 0.0,
        shots: Optional[int] = None,
    ) -> float:
        """
        Classification success rate on a test set.

        Parameters
        ----------
        X_test : np.ndarray, shape [N, n_qubits]
        y_test : np.ndarray, shape [N], values in {+1, -1}
        theta, b : trained parameters
        shots : int, optional

        Returns
        -------
        float in [0, 1]
        """
        correct = sum(
            1 for x, y in zip(X_test, y_test)
            if self.predict(x, theta, b, shots) == y
        )
        return correct / len(y_test)

    # ------------------------------------------------------------------
    # Cost function  [EQ7, EQ8]
    # ------------------------------------------------------------------

    def cost_function(
        self,
        theta: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        b: float,
        shots: int,
    ) -> float:
        """
        Empirical risk R_emp(θ) [EQ7] using sigmoid approximation to binomial
        CDF for the misclassification probability [EQ8].

        R_emp(θ) = (1/|T|) Σ_{x∈T} Pr(ỹ(x) ≠ m(x))   [EQ7]

        Pr(ỹ ≠ y) ≈ sigmoid(√R · ((1+y·b)/2 − p̂_y) / √(2·(1−p̂_y)·p̂_y))  [EQ8]

        Parameters
        ----------
        theta : np.ndarray, shape [n_theta]
        X_train : np.ndarray, shape [N, n_qubits]
        y_train : np.ndarray, shape [N]
        b : float  (bias parameter)
        shots : int  (R in EQ8; paper uses R=200 for cost, 2000 for actual)

        Returns
        -------
        float in [0, 1]
        """
        total_cost = 0.0
        R = shots

        for x, y in zip(X_train, y_train):
            probs = self.get_probs(x, theta, b=b, shots=R)
            p_y = probs[int(y)]

            # Numerically safe sigmoid argument  [EQ8]
            denom = math.sqrt(max(2.0 * (1.0 - p_y) * p_y, 1e-12))
            numerator = (1.0 + y * b) / 2.0 - p_y
            arg = math.sqrt(R) * numerator / denom
            cost_i = _sigmoid(arg)
            total_cost += cost_i

        return total_cost / len(y_train)

    # ------------------------------------------------------------------
    # SPSA optimiser  [Paper: "Spall's SPSA stochastic gradient descent"]
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 250,
        shots_cost: int = 200,
        # ASSUMED: Spall canonical gain-sequence defaults — conf=0.55
        # WARNING: Tune these if cost does not decrease. See config.yaml.
        a: float = 0.628,
        c: float = 0.1,
        A: float = 100.0,
        alpha_spsa: float = 0.602,
        gamma_spsa: float = 0.101,
        bias_range: Tuple[float, float] = (-1.0, 1.0),
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Train QVC using SPSA (Spall 1997/2000) to minimise R_emp(θ, b)  [EQ7].

        SPSA gain sequences  [Spall 1997, ASSUMED defaults — conf=0.55]:
            a_k = a / (k + 1 + A)^alpha
            c_k = c / (k + 1)^gamma

        Update rule:
            gradient estimate g_k = (cost(θ+c_k·δ) − cost(θ−c_k·δ)) / (2·c_k·δ)
            θ_{k+1} = θ_k − a_k · g_k

        Bias b is appended as an extra parameter to the SPSA vector.

        Parameters
        ----------
        X_train : np.ndarray, shape [N, n_qubits]
        y_train : np.ndarray, shape [N]
        n_iter : int  (paper uses 250 iterations)
        shots_cost : int  (paper uses R=200 for cost to smooth gradient)
        a, c, A, alpha_spsa, gamma_spsa : SPSA gain-sequence parameters
        bias_range : Tuple[float, float]  (clip b to this range after each update)
        verbose : bool

        Returns
        -------
        theta_star : np.ndarray, shape [n_theta]
        b_star : float
        cost_history : List[float]  (R_emp at each iteration)
        """
        # Initialise parameters (theta + bias as one vector)
        rng = np.random.default_rng(42)
        params = np.append(
            rng.uniform(-np.pi, np.pi, self.n_theta),
            0.0,   # bias b initialised to 0
        )
        n_params = len(params)
        cost_history: List[float] = []

        pbar = tqdm(range(n_iter), desc=f"SPSA (depth={self.depth})", disable=not verbose)
        for k in pbar:
            # Gain sequences  [ASSUMED: Spall canonical — conf=0.55]
            a_k = a / (k + 1 + A) ** alpha_spsa
            c_k = c / (k + 1) ** gamma_spsa

            # Rademacher perturbation vector δ ∈ {+1,−1}^n_params
            delta = rng.choice([-1.0, 1.0], size=n_params)

            params_plus  = params + c_k * delta
            params_minus = params - c_k * delta

            # Clip bias to valid range
            params_plus[-1]  = np.clip(params_plus[-1],  *bias_range)
            params_minus[-1] = np.clip(params_minus[-1], *bias_range)

            theta_plus,  b_plus  = params_plus[:-1],  float(params_plus[-1])
            theta_minus, b_minus = params_minus[:-1], float(params_minus[-1])

            cost_plus  = self.cost_function(theta_plus,  X_train, y_train, b_plus,  shots_cost)
            cost_minus = self.cost_function(theta_minus, X_train, y_train, b_minus, shots_cost)

            # Gradient estimate
            grad = (cost_plus - cost_minus) / (2.0 * c_k * delta)

            # Parameter update
            params = params - a_k * grad
            params[-1] = np.clip(params[-1], *bias_range)   # keep b in range

            # Record cost at current params for logging
            current_cost = self.cost_function(
                params[:-1], X_train, y_train, float(params[-1]), shots_cost
            )
            cost_history.append(current_cost)

            if verbose:
                pbar.set_postfix({"R_emp": f"{current_cost:.4f}", "b": f"{params[-1]:.3f}"})

        theta_star = params[:-1]
        b_star = float(params[-1])
        return theta_star, b_star, cost_history

    def __repr__(self) -> str:
        return (
            f"QuantumVariationalClassifier("
            f"feature_map={self.feature_map!r}, "
            f"depth={self.depth}, "
            f"n_theta={self.n_theta})"
        )
