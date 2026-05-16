"""
src/geomherd/simulation/cws_substrate.py
Cividino-Sornette continuous-spin (CWS) agent-based model substrate.
Paper: arXiv:2605.11645, Section 3.2
Reference substrate: Cividino, Westphal, Sornette (2023), Phys Rev Research 5(1):013009

N=66 agents, na=4 assets, coupling kappa controls herding.
Each agent's action driven by private signal + neighbor average + idiosyncratic noise.
Linear price impact rule (Assumption A4 in paper).

ASSUMED: Full CWS mechanics inferred from paper description and Cividino 2023.
See Risk R5 in architecture_plan.json.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class CWSSubstrate:
    """
    Cividino-Sornette continuous-spin agent-based model.

    Paper reference: Section 3.2
        N=66 agents, na=4 assets, coupling kappa in {0.5, 0.8, 1.2, 1.8, 2.5}.
        kappa < 1: subcritical (independent decisions)
        kappa > 1: supercritical (herding / cascade)

    Action update (ASSUMED, from paper description + Cividino 2023):
        Each agent i computes a 'spin' s_i as a noisy function of:
          - Private fundamental signal f_i ~ N(0, sigma_f)
          - Peer pressure: kappa * mean(s_j for j in neighbors(i))
          - Idiosyncratic noise: eta_i ~ N(0, sigma_eta)
        Action is discretized: buy if s_i > threshold, sell if s_i < -threshold, else hold.

    Linear price impact (Assumption A4):
        r_{asset,t} = beta * M_t + xi_t
        where M_t = mean order flow across agents.

    Args:
        N: Number of agents (default 66)
        na: Number of assets (default 4)
        kappa: Coupling strength (herding parameter)
        seed: Random seed
        sbase: CWS sbase parameter (Appendix C)
        spost: CWS spost parameter (Appendix C)
    """

    ACTION_BUY = 0
    ACTION_HOLD = 1
    ACTION_SELL = 2
    ACTION_NAMES = {0: "buy", 1: "hold", 2: "sell"}

    def __init__(
        self,
        N: int = 66,
        na: int = 4,
        kappa: float = 1.2,
        seed: int = 42,
        sbase: float = 0.6,
        spost: float = 1.6,
        sigma_f: float = 0.5,    # ASSUMED: private signal noise
        sigma_eta: float = 0.3,  # ASSUMED: idiosyncratic noise
        action_threshold: float = 0.3,  # ASSUMED: discretization threshold
        beta_impact: float = 0.1,       # ASSUMED: price impact coefficient
        sigma_xi: float = 0.02,         # ASSUMED: return noise
    ):
        self.N = N
        self.na = na
        self.kappa = kappa
        self.sbase = sbase
        self.spost = spost
        self.sigma_f = sigma_f
        self.sigma_eta = sigma_eta
        self.action_threshold = action_threshold
        self.beta_impact = beta_impact
        self.sigma_xi = sigma_xi
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        # State
        self._spins: np.ndarray = np.zeros((N, na), dtype=np.float32)
        self._actions: np.ndarray = np.ones(N, dtype=np.int32)  # start: all hold
        self._prices: np.ndarray = np.ones(na, dtype=np.float32) * 100.0
        self._t: int = 0
        # Heterogeneous coupling matrix (ASSUMED: random sparse neighbor graph)
        self._adj = self._init_adjacency()

    def _init_adjacency(self) -> np.ndarray:
        """
        Initialize neighbor adjacency matrix for peer pressure computation.
        ASSUMED: Erdos-Renyi random graph with p=0.3 for N=66.
        Paper mentions 'neighbors weighted by coupling kappa' without full specification.
        """
        N = self.N
        adj = self._rng.random((N, N)) < 0.3
        np.fill_diagonal(adj, False)
        # Make symmetric
        adj = adj | adj.T
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return adj.astype(np.float32) / row_sums

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to initial state. Returns initial actions [N]."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
            self._adj = self._init_adjacency()
        self._spins = self._rng.standard_normal((self.N, self.na)).astype(np.float32) * 0.1
        self._actions = np.ones(self.N, dtype=np.int32)  # hold
        self._prices = np.ones(self.na, dtype=np.float32) * 100.0
        self._t = 0
        return self._actions.copy()

    def _spins_to_actions(self, spins: np.ndarray) -> np.ndarray:
        """Discretize per-agent aggregate spin to {buy, hold, sell}."""
        # Aggregate over assets with equal weight
        agg = spins.mean(axis=1)  # [N]
        actions = np.ones(self.N, dtype=np.int32)  # hold
        actions[agg > self.action_threshold] = self.ACTION_BUY
        actions[agg < -self.action_threshold] = self.ACTION_SELL
        return actions

    def step(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Advance one simulator step.

        Returns:
            actions: [N] int array of agent actions {0=buy, 1=hold, 2=sell}
            prices: [na] asset prices
            info: dict with order_parameter, returns, t
        """
        # Private fundamental signal
        f = self._rng.standard_normal((self.N, self.na)) * self.sigma_f

        # Peer pressure: kappa * weighted-mean spin of neighbors
        peer = self.kappa * (self._adj @ self._spins)  # [N, na]

        # Idiosyncratic noise
        eta = self._rng.standard_normal((self.N, self.na)) * self.sigma_eta

        # Update spins (continuous action / "spin" in the Ising/CWS sense)
        self._spins = np.tanh(f + peer + eta).astype(np.float32)

        # Discretize to actions
        self._actions = self._spins_to_actions(self._spins)

        # Linear price impact (Assumption A4 in paper)
        # M_t = mean order flow (buy=+1, hold=0, sell=-1 for each agent)
        flow = np.where(self._actions == self.ACTION_BUY, 1.0,
                        np.where(self._actions == self.ACTION_SELL, -1.0, 0.0))
        M_t = flow.mean()
        xi = self._rng.standard_normal(self.na) * self.sigma_xi
        returns = self.beta_impact * M_t + xi
        self._prices = (self._prices * (1 + returns)).astype(np.float32)

        # Order parameter V_a(t): mean fraction of agents on the majority side
        # Paper uses V_a(t) = mean(|s_i|) or similar; here use action agreement
        order_parameter = float(np.abs(flow.mean()))

        self._t += 1
        info = {
            "order_parameter": order_parameter,
            "returns": returns,
            "prices": self._prices.copy(),
            "M_t": M_t,
            "t": self._t,
        }
        return self._actions.copy(), self._prices.copy(), info

    def get_order_parameter(self) -> float:
        """V_a(t): mean signed action (buy/sell flow), used as herding order parameter."""
        flow = np.where(self._actions == self.ACTION_BUY, 1.0,
                        np.where(self._actions == self.ACTION_SELL, -1.0, 0.0))
        return float(np.abs(flow.mean()))

    def __repr__(self) -> str:
        return (f"CWSSubstrate(N={self.N}, na={self.na}, kappa={self.kappa}, "
                f"t={self._t})")
