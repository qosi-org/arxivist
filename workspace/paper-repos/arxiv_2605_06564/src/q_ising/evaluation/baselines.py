"""
Baseline policies for Q-Ising evaluation.
Implements the 5 reference policies described in Section 5 of arXiv:2605.06564.

Baselines:
  1. RandomPolicy         — uniform random bin selection
  2. DegreePolicy         — target highest-degree node (degree centrality)
  3. LIRPolicy            — local influence ranking (Liu et al. 2017)
  4. DegreeBinPolicy      — highest-degree untreated node per bin (topology heuristic)
  5. PlainDQNPolicy       — offline RL without Ising state augmentation

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 5.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from q_ising.data.network import NetworkData


class RandomPolicy:
    """Uniform random bin selection policy (Section 5).

    Also used as the historical data-generating policy.

    Args:
        network: NetworkData with bin assignments.
    """

    def __init__(self, network: NetworkData) -> None:
        self.network = network

    def __call__(self, y: np.ndarray, t: int = 0) -> int:
        """Select a random node uniformly from a randomly chosen bin."""
        k = np.random.randint(0, self.network.K)
        members = self.network.get_bin_members(k)
        return int(np.random.choice(members))

    def __repr__(self) -> str:
        return f"RandomPolicy(K={self.network.K})"


class DegreePolicy:
    """Degree centrality policy: always treat the highest-degree node (Section 5).

    Greedy topology heuristic that ignores susceptibility and adoption state.

    Args:
        network: NetworkData.
    """

    def __init__(self, network: NetworkData) -> None:
        self.network = network
        degrees = np.array([len(network.get_neighbors(i)) for i in range(network.N)])
        self._order = np.argsort(degrees)[::-1]  # descending degree order

    def __call__(self, y: np.ndarray, t: int = 0) -> int:
        """Return the highest-degree node (ignores current adoption state)."""
        return int(self._order[0])

    def __repr__(self) -> str:
        return "DegreePolicy()"


class LIRPolicy:
    """Local Influence Ranking policy (Liu et al. 2017) (Section 5).

    Identifies local degree leaders to avoid the rich-club effect.
    Selects node i whose neighbors have low degree (maximum local influence).

    ASSUMED: LIR score = degree(i) / mean(degree(neighbors(i))) as a proxy
    for the Liu et al. 2017 approach. Exact formula requires the original paper.
    # TODO: Verify LIR score formula against Liu et al. 2017.

    Args:
        network: NetworkData.
    """

    def __init__(self, network: NetworkData) -> None:
        self.network = network
        degrees = np.array([len(network.get_neighbors(i)) for i in range(network.N)])

        lir_scores = np.zeros(network.N)
        for i in range(network.N):
            neighbors = network.get_neighbors(i)
            if not neighbors:
                lir_scores[i] = 0.0
            else:
                neighbor_deg = np.mean([degrees[j] for j in neighbors])
                # ASSUMED: LIR score = own degree / mean neighbor degree
                lir_scores[i] = degrees[i] / max(neighbor_deg, 1.0)

        self._order = np.argsort(lir_scores)[::-1]

    def __call__(self, y: np.ndarray, t: int = 0) -> int:
        """Return the highest-LIR-score node."""
        return int(self._order[0])

    def __repr__(self) -> str:
        return "LIRPolicy()"


class DegreeBinPolicy:
    """Degree-bin policy: iterate over bins and select highest-degree untreated node (Section 5).

    At each period, selects the bin with the highest-degree node not yet adopted,
    and treats that node.

    Args:
        network: NetworkData with bin assignments.
    """

    def __init__(self, network: NetworkData) -> None:
        self.network = network
        self._degrees = np.array([len(network.get_neighbors(i)) for i in range(network.N)])

    def __call__(self, y: np.ndarray, t: int = 0) -> int:
        """Select the highest-degree susceptible (y_i=0) node, stratified by bin."""
        # Find best unadopted node per bin, then select global best
        best_node = -1
        best_deg = -1
        for k in range(self.network.K):
            for node in self.network.get_bin_members(k):
                if y[node] == 0 and self._degrees[node] > best_deg:
                    best_deg = self._degrees[node]
                    best_node = node

        if best_node == -1:
            # All nodes adopted — fall back to random
            return int(np.random.randint(0, self.network.N))
        return best_node

    def __repr__(self) -> str:
        return f"DegreeBinPolicy(K={self.network.K})"


class PlainDQNPolicy:
    """Plain DQN policy: offline RL without Ising state augmentation (Section 5).

    Uses raw observed bin-level adoption rates as state, without counterfactual
    Ising probabilities. This is the ablation baseline demonstrating the value
    of the Ising augmentation.

    Args:
        network: NetworkData.
        trained_policy: Pre-trained policy callable (from CQLTrainer, no Ising states).
    """

    def __init__(
        self,
        network: NetworkData,
        trained_policy: Optional[Callable] = None,
    ) -> None:
        self.network = network
        self._policy = trained_policy

    def set_policy(self, policy: Callable) -> None:
        """Set the underlying trained policy."""
        self._policy = policy

    def get_raw_state(self, y: np.ndarray) -> np.ndarray:
        """Build state using only observed bin adoption rates (no Ising augmentation).

        This is the ablation: state = y_bar_{t-1} only (K-dim, not 2K-dim).

        Args:
            y: Current adoption state [N].

        Returns:
            Raw state [K].
        """
        y_bar = np.zeros(self.network.K)
        for k in range(self.network.K):
            members = self.network.get_bin_members(k)
            y_bar[k] = y[members].mean() if members else 0.0
        return y_bar.astype(np.float32)

    def __call__(self, y: np.ndarray, t: int = 0) -> int:
        """Return action from trained plain DQN policy."""
        if self._policy is None:
            raise RuntimeError("PlainDQNPolicy requires a trained policy. Call set_policy() first.")
        state = self.get_raw_state(y)
        # Map bin action to a node
        bin_action = self._policy(state)
        members = self.network.get_bin_members(int(bin_action))
        return int(np.random.choice(members))

    def __repr__(self) -> str:
        return f"PlainDQNPolicy(K={self.network.K}, trained={self._policy is not None})"
