"""
training/replay_buffer.py — Uniform Experience Replay Buffer.

Implements a circular replay buffer with uniform sampling.
Paper explicitly disables prioritised replay (Section 3.2, footnote 1):
  "Prioritised sampling caused more frequent resampling of highly noisy
   instances where learning was particularly difficult, hence degrading
   performance."

Buffer size: 2e6 transitions (Table 2).

Paper: arXiv:2301.08688 — Section 3.2 and Table 2.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Transition(NamedTuple):
    """Single SARSD transition for experience replay."""
    state: NDArray
    action: int
    reward: float
    next_state: NDArray
    done: bool


class ReplayBuffer:
    """Circular uniform experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store (default 2_000_000).
        obs_dim: Dimensionality of the flattened observation vector.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int = 2_000_000,
        obs_dim: int = 1000,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        # Pre-allocate arrays for efficiency
        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)

        self._ptr: int = 0
        self._size: int = 0

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Args:
            transition: A (state, action, reward, next_state, done) tuple.
        """
        idx = self._ptr
        self._states[idx] = transition.state
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._next_states[idx] = transition.next_state
        self._dones[idx] = transition.done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Sample a random batch of transitions (uniform sampling, per paper).

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) arrays.
        """
        assert self._size >= batch_size, (
            f"Buffer has {self._size} transitions, requested batch_size={batch_size}."
        )
        indices = self.rng.integers(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity:,}, "
            f"size={self._size:,}, obs_dim={self.obs_dim})"
        )
