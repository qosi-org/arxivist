"""
State constructor for Q-Ising (Stage 1, post-inference).
Converts per-node Ising estimates into bin-level Q-Ising states.

Implements Equations eq_latent_state, eq_bin_state, and eq_qising_state
from Section 3.1 of arXiv:2605.06564.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 3.1.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from q_ising.data.network import NetworkData
from q_ising.data.panel import ObservationalPanel
from q_ising.models.ising import DynamicIsingModel


class StateConstructor:
    """Builds bin-level Q-Ising state vectors from a fitted Ising model.

    The Q-Ising state is defined as (Section 3.1, Eq. eq_qising_state):
        s_t = (l_bar^0_t, y_bar_{t-1}) in [0,1]^K x [0,1]^K

    where:
      l_bar^0_{t,k} = (1/|B_k|) * sum_{i in B_k} sigma(eta_{i,t}(empty; theta_hat))
                      (forward-looking, counterfactual model-based component)
      y_bar_{t-1,k} = (1/|B_k|) * sum_{i in B_k} y_{i,t-1}
                      (backward-looking, observed adoption)

    Args:
        ising_model: Fitted DynamicIsingModel.
        network: NetworkData with bin assignments (must match ising_model).
    """

    def __init__(self, ising_model: DynamicIsingModel, network: NetworkData) -> None:
        assert network.bin_labels is not None
        self.ising = ising_model
        self.network = network
        self.K = network.K
        self.N = network.N

    def build_state(
        self,
        y_prev: np.ndarray,
        t: int,
        param_dict: Optional[Dict] = None,
    ) -> np.ndarray:
        """Build a single Q-Ising state vector for period t.

        Args:
            y_prev: Adoption state at t-1, shape [N].
            t: Current period index (used for context; not directly used in state).
            param_dict: Optional per-node parameter dict override (for ensemble).

        Returns:
            s_t: Q-Ising state [2K] = concat(l_bar^0_t, y_bar_{t-1}).
        """
        # --- l_bar^0_t: counterfactual bin-level adoption probability ---
        # Eq. eq_latent_state: l_hat^0_{i,t} = sigma(eta_{i,t}(empty; theta_hat))
        l_hat_0 = np.zeros(self.N)
        for i in range(self.N):
            l_hat_0[i] = self.ising.counterfactual_prob(y_prev, node=i, param_dict=param_dict)

        # Eq. eq_bin_state: l_bar^0_{t,k} = mean over bin B_k
        l_bar_0 = np.zeros(self.K)
        for k in range(self.K):
            members = self.network.get_bin_members(k)
            l_bar_0[k] = l_hat_0[members].mean() if members else 0.0

        # --- y_bar_{t-1}: observed bin-level adoption ---
        # Eq. eq_bin_state: y_bar_{t-1,k} = mean(y_{i,t-1} for i in B_k)
        y_bar = np.zeros(self.K)
        for k in range(self.K):
            members = self.network.get_bin_members(k)
            y_bar[k] = y_prev[members].mean() if members else 0.0

        # Eq. eq_qising_state: s_t = (l_bar^0_t, y_bar_{t-1}) concatenated
        return np.concatenate([l_bar_0, y_bar]).astype(np.float32)

    def build_all_states(self, panel: ObservationalPanel) -> np.ndarray:
        """Build Q-Ising states for all T+1 time steps in the panel.

        Returns states s_1, s_2, ..., s_{T+1} where s_t conditions on y_{t-1}.
        State s_1 conditions on y_0 (initial state); s_{T+1} conditions on y_T.

        Args:
            panel: ObservationalPanel.

        Returns:
            states: Array of shape [T+1, 2K].
        """
        T = panel.T
        states = np.zeros((T + 1, 2 * self.K), dtype=np.float32)

        # s_1 conditions on y_0
        states[0] = self.build_state(panel.y0, t=1)

        for t in range(1, T + 1):
            y_prev = panel.outcomes[t - 1]  # y_t (previous for next state s_{t+1})
            states[t] = self.build_state(y_prev, t=t + 1)

        return states

    def build_states_ensemble(
        self,
        panel: ObservationalPanel,
        theta_draws: List[Dict],
    ) -> List[np.ndarray]:
        """Build state sequences for each MCMC posterior draw.

        Used in Stage 3 ensemble policy (Section 3.3):
        Each draw theta^(p) induces a separate state sequence s_t^(p).
        Note: y_bar_{t-1} is shared across all draws (observed data).

        Args:
            panel: ObservationalPanel.
            theta_draws: List of P parameter dicts [{node -> vec}, ...].

        Returns:
            List of P state arrays, each [T+1, 2K].
        """
        state_sequences = []
        for draw in theta_draws:
            T = panel.T
            states = np.zeros((T + 1, 2 * self.K), dtype=np.float32)
            states[0] = self.build_state(panel.y0, t=1, param_dict=draw)
            for t in range(1, T + 1):
                y_prev = panel.outcomes[t - 1]
                states[t] = self.build_state(y_prev, t=t + 1, param_dict=draw)
            state_sequences.append(states)
        return state_sequences

    def __repr__(self) -> str:
        return f"StateConstructor(N={self.N}, K={self.K}, state_dim={2*self.K})"
