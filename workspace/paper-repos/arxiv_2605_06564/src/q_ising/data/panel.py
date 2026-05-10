"""
Observational panel data container for Q-Ising.
Stores the single offline trajectory D = {y_0, (a_t, y_t)_{t=1}^{T_train}}.

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    """A single RL transition for offline Q-learning (Stage 2).

    Fields correspond to the (s_t, b_t, r_t, s_{t+1}) tuples in Section 3.2.
    """
    state: np.ndarray         # s_t  — Q-Ising state [2K]
    bin_action: int           # b_t  — bin-level action in [0, K-1]
    reward: float             # r_t  — network-wide adoption rate (Eq. reward)
    next_state: np.ndarray    # s_{t+1}


class ObservationalPanel:
    """Single observational panel collected under a historical policy.

    Stores D = {y_0, (a_t, y_t)_{t=1}^{T_train}} as described in Section 2.

    Args:
        y0: Initial adoption state [N], binary.
        actions: Node-level actions a_t for t=1..T, shape [T].
        outcomes: Node adoption outcomes y_t for t=1..T, shape [T, N].
    """

    def __init__(
        self,
        y0: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
    ) -> None:
        assert y0.ndim == 1, f"y0 must be 1D, got shape {y0.shape}"
        assert actions.ndim == 1, f"actions must be 1D, got shape {actions.shape}"
        assert outcomes.ndim == 2, f"outcomes must be [T, N], got shape {outcomes.shape}"
        assert outcomes.shape[0] == actions.shape[0], \
            "actions and outcomes must have the same length T"
        assert outcomes.shape[1] == y0.shape[0], \
            "outcomes shape[1] must equal N (number of nodes)"

        self.y0 = y0.astype(np.int32)
        self.actions = actions.astype(np.int32)
        self.outcomes = outcomes.astype(np.int32)
        self.T = actions.shape[0]
        self.N = y0.shape[0]

    def get_period(self, t: int) -> Tuple[np.ndarray, int, np.ndarray]:
        """Return (y_{t-1}, a_t, y_t) for period t (1-indexed).

        Args:
            t: Period index, 1 <= t <= T.

        Returns:
            Tuple of (prev_outcome [N], action scalar, outcome [N]).
        """
        assert 1 <= t <= self.T, f"t={t} out of range [1, {self.T}]"
        y_prev = self.y0 if t == 1 else self.outcomes[t - 2]
        return y_prev, int(self.actions[t - 1]), self.outcomes[t - 1]

    def to_rl_transitions(
        self,
        states: np.ndarray,
        bin_labels: np.ndarray,
    ) -> List[Transition]:
        """Convert panel to list of RL transitions (s_t, b_t, r_t, s_{t+1}).

        Maps node-level actions a_t to bin-level actions b_t.
        Reward r_t is mean adoption rate (Eq. reward in SIR).

        Args:
            states: Q-Ising state sequence, shape [T+1, 2K].
                    states[0] = s_1, states[T] = s_{T+1}.
            bin_labels: Node-to-bin assignment array [N].

        Returns:
            List of Transition objects of length T-1.
        """
        assert states.shape[0] == self.T + 1, \
            f"states must have T+1={self.T+1} rows, got {states.shape[0]}"

        transitions = []
        for t in range(1, self.T):  # t = 1..T-1 (need s_{t+1})
            y_t = self.outcomes[t - 1]            # y_t
            a_t = int(self.actions[t - 1])        # node-level action
            b_t = int(bin_labels[a_t])            # map to bin (Section 3.2)
            r_t = float(y_t.mean())               # Eq. reward: (1/N) sum y_{i,t}
            transitions.append(Transition(
                state=states[t - 1].copy(),
                bin_action=b_t,
                reward=r_t,
                next_state=states[t].copy(),
            ))
        return transitions

    def __repr__(self) -> str:
        return f"ObservationalPanel(N={self.N}, T={self.T})"
