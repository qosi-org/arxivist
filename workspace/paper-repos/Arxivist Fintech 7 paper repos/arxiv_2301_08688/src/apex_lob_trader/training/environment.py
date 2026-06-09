"""
training/environment.py — OpenAI Gym LOB Trading Environment.

Implements the RL environment described in Section 4.2:
  - State space: history of LOB snapshots + signal + portfolio state
  - Action space: 7 discrete actions (buy/sell @ bid/mid/ask + skip)
  - Reward: w_dir * R_dir + (1 - w_dir) * R_pnl  (Eq. 3, Section 4.2)
  - Inventory constraints: [pos_min, pos_max] = [-10, 10]
  - Market order triggered if passive fill would breach inventory bounds

Paper: arXiv:2301.08688 — Section 4.2.
Based on ABIDES simulator concept [7]; here implemented as a lightweight
gym wrapper over the LOBDataset for reproducibility without ABIDES dependency.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from numpy.typing import NDArray

from apex_lob_trader.data.signal_generator import SignalGenerator
from apex_lob_trader.data.lob_dataset import LOBDataset


# Action encoding: index → (direction, price_level)
# direction: +1=buy, -1=sell | price_level: -1=bid, 0=mid, +1=ask
ACTION_MAP: dict[int, tuple[int, int]] = {
    0: (-1, -1),  # sell @ bid  (passive sell — resting at best bid)
    1: (-1,  0),  # sell @ mid
    2: (-1, +1),  # sell @ ask  (aggressive sell — crosses spread)
    3: (+1, -1),  # buy  @ bid  (aggressive buy — crosses spread)
    4: (+1,  0),  # buy  @ mid
    5: (+1, +1),  # buy  @ ask  (passive buy — resting at best ask)
    6: None,      # skip
}
NUM_ACTIONS = 7


class LOBTradingEnv(gym.Env):
    """Limit Order Book trading environment (Section 4.2).

    Observation:
        Flat vector of the last `history_len` LOB snapshots, each containing:
        [time_remaining, cash, inventory, d_down, d_neutral, d_up,
         ask_price, ask_vol, agent_ask_vol, bid_price, bid_vol, agent_bid_vol]
        Shape: [history_len * state_dim_per_step]

    Actions:
        0: sell@bid  1: sell@mid  2: sell@ask
        3: buy@bid   4: buy@mid   5: buy@ask
        6: skip

    Reward (Eq. 3):
        r_t = w_dir * R_dir + (1 - w_dir) * R_pnl
        R_pnl = log(M_t) - log(M_{t-1})
        R_dir = kappa * [-1,0,1] · d_t * X_t
        w_dir decays by psi each step.

    Args:
        dataset: LOBDataset instance with episodes loaded.
        cfg: Full config dictionary.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset: LOBDataset, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg

        env_cfg = cfg["env"]
        sig_cfg = cfg["signal"]
        train_cfg = cfg["training"]
        reward_cfg = train_cfg["reward"]

        self.history_len: int = env_cfg["history_len"]
        self.state_dim: int = env_cfg["state_dim_per_step"]
        self.pos_min: int = env_cfg["inventory"]["pos_min"]
        self.pos_max: int = env_cfg["inventory"]["pos_max"]
        self.action_freq: float = 0.1  # seconds between steps

        # Reward parameters (Section 4.2, Eq. 3)
        self.kappa: float = reward_cfg["kappa"]
        self.w_dir: float = reward_cfg["w_dir_initial"]
        self.psi: float = reward_cfg["psi_decay"]

        self.signal_gen = SignalGenerator(
            a_H=sig_cfg["noise_level_a"],
            a_L=sig_cfg["a_L"],
            phi=sig_cfg["persistence_phi"],
            horizon_h=sig_cfg["horizon_h"],
            threshold_k=sig_cfg["return_threshold_k"],
        )

        obs_size = self.history_len * self.state_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # Episode state
        self._episode_data: Optional[NDArray] = None
        self._t: int = 0
        self._cash: float = 0.0
        self._inventory: int = 0
        self._history: list[NDArray] = []
        self._episode_idx: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray, dict]:
        """Reset environment to start of a new episode."""
        super().reset(seed=seed)

        # Pick episode (cycle through dataset)
        self._episode_idx = (self._episode_idx + 1) % len(self.dataset)
        self._episode_data = self.dataset[self._episode_idx]
        self._T = len(self._episode_data)

        self._t = self.history_len  # start after enough history
        self._cash = 0.0
        self._inventory = 0
        self._agent_bid_vol = 0
        self._agent_ask_vol = 0
        self.signal_gen.reset()

        # Pre-fill history buffer
        self._history = []
        mid_prices = self.dataset.get_mid_prices(self._episode_idx)
        for i in range(self.history_len):
            self.signal_gen.step(mid_prices, i)
            self._history.append(self._make_snapshot(i))

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: Integer action index (0–6).

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self._episode_data is not None, "Call reset() before step()."
        assert 0 <= action < NUM_ACTIONS

        row = self._episode_data[self._t]
        bid_price = float(row[1])
        ask_price = float(row[3])
        mid_price = float(row[5])

        # Compute mark-to-market BEFORE action (Eq. 3)
        M_prev = self._cash + self._inventory * mid_price

        # ── Execute action (Section 4.2) ────────────────────────────────────
        if action < 6:  # not skip
            direction, price_level = ACTION_MAP[action]
            new_inv = self._inventory + direction

            # Check inventory constraint (Section 4.2)
            if self.pos_min <= new_inv <= self.pos_max:
                exec_price = self._get_exec_price(direction, price_level, bid_price, ask_price, mid_price)
                self._inventory = new_inv
                self._cash -= direction * exec_price  # buy costs cash, sell gains cash

        # Market order to enforce constraints after passive fills (Section 4.2)
        if self._inventory > self.pos_max:
            self._cash += mid_price * (self._inventory - self.pos_max)
            self._inventory = self.pos_max
        elif self._inventory < self.pos_min:
            self._cash -= mid_price * (self.pos_min - self._inventory)
            self._inventory = self.pos_min

        # ── Advance time and update signal ─────────────────────────────────
        self._t += 1
        mid_prices = self.dataset.get_mid_prices(self._episode_idx)
        signal = self.signal_gen.step(mid_prices, self._t)

        # Update history
        self._history.append(self._make_snapshot(self._t))
        if len(self._history) > self.history_len:
            self._history.pop(0)

        # ── Compute reward (Eq. 3, Section 4.2) ────────────────────────────
        row_new = self._episode_data[self._t]
        mid_price_new = float(row_new[5])
        M_new = self._cash + self._inventory * mid_price_new

        # R_pnl = log(M_t) - log(M_{t-1})  (Eq. 3)
        if M_prev > 0 and M_new > 0:
            R_pnl = float(np.log(M_new + 1e-10) - np.log(M_prev + 1e-10))
        else:
            R_pnl = 0.0

        # R_dir = kappa * [-1,0,1] · d_t * X_t  (Eq. 3)
        dir_vector = np.array([-1.0, 0.0, 1.0])
        R_dir = float(self.kappa * np.dot(dir_vector, signal) * self._inventory)

        # Total reward with decaying w_dir curriculum (Section 4.2)
        reward = float(self.w_dir * R_dir + (1.0 - self.w_dir) * R_pnl)
        self.w_dir *= self.psi  # decay directional weight

        terminated = self._t >= self._T - 1
        obs = self._get_obs()
        info = {
            "mid_price": mid_price_new,
            "inventory": self._inventory,
            "cash": self._cash,
            "M": M_new,
            "R_pnl": R_pnl,
            "R_dir": R_dir,
            "w_dir": self.w_dir,
            "signal": signal.tolist(),
        }
        return obs, reward, terminated, False, info

    def _get_exec_price(
        self,
        direction: int,
        price_level: int,
        bid: float,
        ask: float,
        mid: float,
    ) -> float:
        """Map (direction, price_level) → execution price.

        Section 4.2: price_level ∈ {-1=bid, 0=mid, +1=ask}
        Buy aggressive = at ask price. Sell aggressive = at bid price.
        """
        if price_level == -1:
            return bid
        elif price_level == 0:
            return mid
        else:
            return ask

    def _make_snapshot(self, t: int) -> NDArray:
        """Build state vector for time step t (Section 4.2)."""
        row = self._episode_data[t]
        T_remaining = (self._T - t) / self._T  # normalised time remaining
        signal = self.signal_gen.current_signal  # [d_down, d_neutral, d_up]

        snapshot = np.array([
            T_remaining,
            self._cash / 1e4,       # normalise cash
            float(self._inventory) / self.pos_max,
            signal[0],
            signal[1],
            signal[2],
            float(row[3]),          # ask_price
            float(row[4]) / 1e3,    # ask_vol (normalised)
            float(self._agent_ask_vol),
            float(row[1]),          # bid_price
            float(row[2]) / 1e3,    # bid_vol (normalised)
            float(self._agent_bid_vol),
        ], dtype=np.float32)

        # Truncate or pad to state_dim
        return snapshot[: self.state_dim]

    def _get_obs(self) -> NDArray:
        """Flatten history buffer into observation vector."""
        history_arr = np.stack(self._history[-self.history_len:], axis=0)  # [history_len, state_dim]
        return history_arr.flatten().astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"LOBTradingEnv(history_len={self.history_len}, "
            f"state_dim={self.state_dim}, "
            f"pos_bounds=[{self.pos_min},{self.pos_max}])"
        )
