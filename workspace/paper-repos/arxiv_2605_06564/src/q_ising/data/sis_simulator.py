"""
Susceptible-Infected-Susceptible (SIS) simulator for Q-Ising experiments.
Implements the three-step dynamics: churn → seed → spread.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Appendix E.1.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from q_ising.data.network import NetworkData
from q_ising.data.panel import ObservationalPanel


class SISSimulator:
    """Heterogeneous SIS dynamics on a fixed network.

    At each period, three ordered sub-steps execute (Appendix E.1):
      1. Churn: each adopted node i reverts to susceptible with prob delta_i.
      2. Seed: one node from the selected bin is forced to adopt (perfect treatment).
      3. Spread: each adopted node i infects each susceptible neighbor j with prob beta_i.

    Infection probability for node j: p_adopt_j = 1 - prod_{i adopted neighbors}(1 - beta_i).
    (Appendix E.1, Eq. sis_transmission)

    Args:
        network: NetworkData with adjacency and bin assignments.
        spread_rates: Node-specific spread probability beta_i, shape [N] or [K] (bin-level).
        churn_rates: Node-specific churn probability delta_i, shape [N] or [K] (bin-level).
        spread_by_bin: If True, spread_rates and churn_rates are indexed by bin (K-dim).
    """

    def __init__(
        self,
        network: NetworkData,
        spread_rates: np.ndarray,
        churn_rates: np.ndarray,
        spread_by_bin: bool = True,
    ) -> None:
        self.network = network
        N = network.N

        if spread_by_bin:
            # Expand bin-level rates to node-level
            assert network.bin_labels is not None, "Bin labels required for bin-level spread rates"
            K_given = len(spread_rates)
            self._spread = np.array([spread_rates[network.bin_labels[i]] for i in range(N)], dtype=float)
            self._churn = np.array([churn_rates[network.bin_labels[i]] for i in range(N)], dtype=float)
        else:
            self._spread = np.array(spread_rates, dtype=float)
            self._churn = np.array(churn_rates, dtype=float)

        assert self._spread.shape == (N,), f"spread_rates must be length N={N}"
        assert self._churn.shape == (N,), f"churn_rates must be length N={N}"

    def step(self, y: np.ndarray, action: int, rng: np.random.Generator) -> np.ndarray:
        """Advance one period of SIS dynamics.

        Args:
            y: Current binary adoption state [N].
            action: Node index to treat (forced adoption).
            rng: NumPy random generator (for reproducibility).

        Returns:
            y_next: New adoption state [N].
        """
        N = self.network.N
        y = y.copy().astype(np.int32)

        # Step 1: Churn — adopted nodes revert with prob delta_i (Appendix E.1)
        churn_mask = (y == 1) & (rng.random(N) < self._churn)
        y[churn_mask] = 0

        # Step 2: Seed — force-adopt the selected node (Appendix E.1, "perfect treatment")
        y[action] = 1

        # Step 3: Spread — each adopted node tries to infect susceptible neighbors
        # p_adopt_j = 1 - prod_{i in adopted neighbors of j}(1 - beta_i)
        # (Appendix E.1, Eq. sis_transmission)
        y_next = y.copy()
        adopted = np.where(y == 1)[0]
        for j in range(N):
            if y[j] == 0:  # susceptible
                neighbors_adopted = [i for i in self.network.get_neighbors(j) if y[i] == 1]
                if neighbors_adopted:
                    # Complement product formula (Appendix E.1)
                    p_adopt = 1.0 - np.prod([1.0 - self._spread[i] for i in neighbors_adopted])
                    if rng.random() < p_adopt:
                        y_next[j] = 1

        return y_next

    def generate_panel(
        self,
        T: int,
        policy: Callable[[np.ndarray, int], int],
        y0: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> ObservationalPanel:
        """Generate an observational panel under a given policy.

        Args:
            T: Number of periods.
            policy: Callable mapping (y_prev, t) -> node_action.
            y0: Initial adoption state [N]. Defaults to all-zeros.
            seed: Random seed.

        Returns:
            ObservationalPanel with T periods of data.
        """
        N = self.network.N
        rng = np.random.default_rng(seed)

        if y0 is None:
            y0 = np.zeros(N, dtype=np.int32)

        actions = np.zeros(T, dtype=np.int32)
        outcomes = np.zeros((T, N), dtype=np.int32)

        y = y0.copy()
        for t in range(1, T + 1):
            a_t = policy(y, t)
            y_next = self.step(y, a_t, rng)
            actions[t - 1] = a_t
            outcomes[t - 1] = y_next
            y = y_next

        return ObservationalPanel(y0=y0, actions=actions, outcomes=outcomes)

    def run_test(
        self,
        policy: Callable[[np.ndarray], int],
        H: int,
        n_runs: int,
        seed: int = 0,
    ) -> np.ndarray:
        """Evaluate a policy over multiple independent test episodes.

        Each episode starts from zero adoption (Section 5 evaluation protocol).

        Args:
            policy: Callable mapping state s [2K] -> bin action, OR node action.
            H: Test horizon.
            n_runs: Number of independent episodes.
            seed: Base seed; each run uses seed + run_idx.

        Returns:
            rewards: Array of shape [n_runs, H] with per-period adoption rates.
        """
        N = self.network.N
        rewards = np.zeros((n_runs, H))

        for run in range(n_runs):
            rng = np.random.default_rng(seed + run)
            y = np.zeros(N, dtype=np.int32)  # Start from no adoption (Section 5)
            for h in range(H):
                a_h = policy(y, h)
                y = self.step(y, a_h, rng)
                rewards[run, h] = y.mean()   # Eq. reward

        return rewards

    def __repr__(self) -> str:
        return f"SISSimulator(N={self.network.N}, K={self.network.K})"
