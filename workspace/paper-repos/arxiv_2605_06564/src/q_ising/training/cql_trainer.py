"""
Conservative Q-Learning (CQL) trainer for Q-Ising (Stage 2).
Wraps d3rlpy's CQL implementation with Q-Ising-specific dataset construction.

Implements the offline Q-learning step from Section 3.2 of arXiv:2605.06564,
including the CQL objective (Eq. eq_cql_loss).

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 3.2, Appendix E.2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from q_ising.data.panel import Transition
from q_ising.utils.config import CQLConfig


class CQLTrainer:
    """Trains a CQL Q-function over the Q-Ising state-action space.

    The Q-function Q: S x [K] -> R is trained with the CQL objective
    (Eq. eq_cql_loss, Section 3.2):

        L_CQL(Q) = Bellman_error + alpha * conservative_penalty

    where the conservative penalty penalizes Q-values on all actions while
    pushing up values on observed actions, addressing distributional shift.

    Args:
        K: Number of bins (action space size).
        state_dim: Dimension of Q-Ising state (= 2K).
        config: CQLConfig with all hyperparameters.
    """

    def __init__(self, K: int, state_dim: int, config: CQLConfig) -> None:
        self.K = K
        self.state_dim = state_dim
        self.config = config
        self._model = None
        self._trained = False

    def build_dataset(self, transitions: List[Transition]):
        """Convert Q-Ising transitions to d3rlpy MDPDataset.

        Args:
            transitions: List of (s_t, b_t, r_t, s_{t+1}) from panel.

        Returns:
            d3rlpy MDPDataset object.
        """
        try:
            import d3rlpy
        except ImportError:
            raise ImportError("d3rlpy is required. pip install d3rlpy")

        observations = np.stack([t.state for t in transitions]).astype(np.float32)
        actions = np.array([t.bin_action for t in transitions], dtype=np.int32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_observations = np.stack([t.next_state for t in transitions]).astype(np.float32)
        terminals = np.zeros(len(transitions), dtype=np.float32)
        terminals[-1] = 1.0   # Mark episode end

        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        return dataset

    def train(self, transitions: List[Transition], save_dir: Optional[str] = None):
        """Train CQL on offline transitions.

        Uses hyperparameters from Appendix E.2:
          - hidden_layers: [256, 256]
          - learning_rate: 3e-4
          - batch_size: 64
          - max_steps: 30,000
          - alpha: 0.1 (conservative penalty)
          - discount: 0.8

        Args:
            transitions: Offline RL transitions.
            save_dir: Optional directory to save model checkpoints.

        Returns:
            Trained d3rlpy CQL model.
        """
        try:
            import d3rlpy
            from d3rlpy.algos import DiscreteCQL
        except ImportError:
            raise ImportError("d3rlpy is required. pip install d3rlpy")

        dataset = self.build_dataset(transitions)

        cfg = self.config
        encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=cfg.hidden_layers,
            activation=cfg.activation,
            use_batch_norm=cfg.batch_norm,
            dropout_rate=cfg.dropout_rate,
        )

        # DiscreteCQL (Appendix E.2, Section 3.2, Eq. eq_cql_loss)
        cql = DiscreteCQL(
            learning_rate=cfg.learning_rate,
            alpha=cfg.alpha,                    # conservative penalty weight (Appendix E.2)
            gamma=cfg.discount,                 # discount psi (Appendix E.2)
            batch_size=cfg.batch_size,
            encoder_factory=encoder_factory,
        )

        cql.fit(
            dataset,
            n_steps=cfg.max_steps,
            n_steps_per_epoch=cfg.steps_per_epoch,
            evaluators={},                       # No online evaluation in offline RL
            save_interval=cfg.max_steps,         # Only save at end
            experiment_name=save_dir or "q_ising_cql",
            with_timestamp=False,
            logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
        )

        self._model = cql
        self._trained = True
        return cql

    def get_policy(self) -> Callable[[np.ndarray], int]:
        """Return a greedy policy function: s -> argmax_k Q(s, k).

        Returns:
            Callable mapping state array [2K] to bin action in [0, K-1].
        """
        assert self._trained, "Trainer must be fitted before calling get_policy()"
        model = self._model

        def policy(state: np.ndarray) -> int:
            s = state.reshape(1, -1).astype(np.float32)
            action = model.predict(s)[0]
            return int(action)

        return policy

    def save(self, path: str) -> None:
        """Save trained CQL model."""
        assert self._trained
        self._model.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved CQL model."""
        try:
            from d3rlpy.algos import DiscreteCQL
        except ImportError:
            raise ImportError("d3rlpy required")
        self._model = DiscreteCQL.from_json(path)
        self._trained = True

    def __repr__(self) -> str:
        return (
            f"CQLTrainer(K={self.K}, state_dim={self.state_dim}, "
            f"trained={self._trained})"
        )
