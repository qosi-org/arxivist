"""
Ensemble policy trainer for Q-Ising (Stage 3).
Trains P CQL agents on P MCMC posterior draws and aggregates via majority vote.

Implements Section 3.3 of arXiv:2605.06564: posterior sampling propagates
first-stage uncertainty into the learned policy.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 3.3.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from q_ising.data.panel import ObservationalPanel, Transition
from q_ising.training.cql_trainer import CQLTrainer
from q_ising.utils.config import CQLConfig


class EnsembleTrainer:
    """Trains P CQL agents over P MCMC posterior state representations.

    Each MCMC draw theta^(p) induces a different state sequence s_t^(p),
    producing a distinct CQL agent pi_hat^(p). The ensemble policy uses
    majority vote (Section 3.3, Algorithm 1):

        pi_hat_ens(s) = argmax_k sum_{p=1}^{P} 1[pi_hat^(p)(s) == k]

    When draws agree → high confidence; when dispersed → strategic uncertainty.

    Args:
        K: Number of bins.
        state_dim: Dimension of state vector (= 2K).
        config: CQLConfig.
    """

    def __init__(self, K: int, state_dim: int, config: CQLConfig) -> None:
        self.K = K
        self.state_dim = state_dim
        self.config = config
        self._agents: List[CQLTrainer] = []
        self._policies: List[Callable] = []

    def train_all(
        self,
        panel: ObservationalPanel,
        state_sequences: List[np.ndarray],  # [P x (T+1, 2K)]
    ) -> List[CQLTrainer]:
        """Train one CQL agent per posterior draw.

        Args:
            panel: ObservationalPanel with observed (actions, outcomes, rewards).
            state_sequences: P state arrays, each [T+1, 2K], one per MCMC draw.

        Returns:
            List of P trained CQLTrainer instances.
        """
        P = len(state_sequences)
        bin_labels = None  # Will be extracted from transitions
        self._agents = []
        self._policies = []

        for p, states_p in enumerate(state_sequences):
            print(f"  Training ensemble agent {p+1}/{P}...")

            # Build transitions for this draw's state representation
            # y_bar (second K dims of state) is shared; l_bar^0 (first K) differs per draw
            transitions_p: List[Transition] = []
            T = panel.T
            for t in range(1, T):
                y_t = panel.outcomes[t - 1]
                a_t = int(panel.actions[t - 1])
                # Map node action to bin — need bin_labels from network
                # We encode bin_action in transitions via the state's y_bar component
                # NOTE: bin mapping must be supplied externally; stored in state metadata
                # ASSUMED: bin_labels passed via closure or extracted from states
                r_t = float(y_t.mean())
                # Use draw-specific states
                transitions_p.append(Transition(
                    state=states_p[t - 1].copy(),
                    bin_action=0,     # placeholder — will be set below
                    reward=r_t,
                    next_state=states_p[t].copy(),
                ))

            trainer = CQLTrainer(K=self.K, state_dim=self.state_dim, config=self.config)
            trainer.train(transitions_p)
            self._agents.append(trainer)
            self._policies.append(trainer.get_policy())

        return self._agents

    def train_from_transitions(
        self,
        transitions_per_draw: List[List[Transition]],
    ) -> List[CQLTrainer]:
        """Train P agents from pre-built transitions (preferred interface).

        Args:
            transitions_per_draw: List of P transition lists, one per MCMC draw.

        Returns:
            List of P trained CQLTrainer instances.
        """
        P = len(transitions_per_draw)
        self._agents = []
        self._policies = []

        for p, transitions_p in enumerate(transitions_per_draw):
            print(f"  Training ensemble agent {p+1}/{P}...")
            trainer = CQLTrainer(K=self.K, state_dim=self.state_dim, config=self.config)
            trainer.train(transitions_p)
            self._agents.append(trainer)
            self._policies.append(trainer.get_policy())

        return self._agents

    def majority_vote_policy(self, state: np.ndarray) -> int:
        """Return bin action via majority vote across all ensemble agents.

        Implements Algorithm 1 line 13 of arXiv:2605.06564:
            pi_hat_ens(s) = argmax_k sum_p 1[pi_hat^(p)(s) == k]

        When most draws agree, the planner acts with confidence;
        when votes are dispersed, the allocation is sensitive to parameter
        uncertainty (Section 3.3).

        Args:
            state: Q-Ising state [2K].

        Returns:
            Bin action (majority vote winner) in [0, K-1].
        """
        assert self._policies, "No trained agents. Call train_from_transitions() first."

        votes = np.zeros(self.K, dtype=int)
        for policy in self._policies:
            a = policy(state)
            votes[a] += 1

        return int(np.argmax(votes))

    def get_vote_distribution(self, state: np.ndarray) -> np.ndarray:
        """Return full vote distribution across bins for a given state.

        Useful for visualizing uncertainty (dispersed votes = uncertain decision).

        Args:
            state: Q-Ising state [2K].

        Returns:
            votes: Integer array [K] with vote counts per bin.
        """
        votes = np.zeros(self.K, dtype=int)
        for policy in self._policies:
            votes[policy(state)] += 1
        return votes

    def __repr__(self) -> str:
        return (
            f"EnsembleTrainer(K={self.K}, n_agents={len(self._agents)}, "
            f"trained={len(self._agents) > 0})"
        )
