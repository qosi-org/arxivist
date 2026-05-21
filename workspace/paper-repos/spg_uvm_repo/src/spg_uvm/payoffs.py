"""
Option payoff functions for SPG-UVM experiments.

Implements all options from Section 4.2 of arXiv:2605.06670:
  - GeoOutperformer        (Section 4.2.1)
  - OutperformerSpread     (Section 4.2.1)
  - BestOfButterfly        (Section 4.2.2)
  - GeoCallSpread          (Section 4.2.3)
  - CallSharpe             (Section 4.2.4, path-dependent)

All payoffs take asset prices at maturity X_T of shape [B, d] and return [B].
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GeoOutperformer(nn.Module):
    """
    Geometric outperformer payoff (Section 4.2.1):

        g(x) = max(0, (prod_{i=2}^d x^i)^{1/(d-1)} - x^1)

    Asset 1 is the benchmark; assets 2..d form the geometric mean basket.

    Args:
        d: Number of assets (d >= 2).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        assert d >= 2, "GeoOutperformer requires d >= 2"
        self.d = d

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Asset prices at maturity, shape [B, d].

        Returns:
            Payoff, shape [B].
        """
        assert x.shape[1] == self.d, f"Expected [B,{self.d}], got {x.shape}"
        benchmark = x[:, 0]                              # X^1_T [B]
        basket = x[:, 1:]                                # X^{2..d}_T [B, d-1]
        geo_mean = torch.exp(torch.log(basket).mean(dim=1))  # [B] geometric mean
        return torch.relu(geo_mean - benchmark)

    def __repr__(self) -> str:
        return f"GeoOutperformer(d={self.d})"


class OutperformerSpread(nn.Module):
    """
    Outperformer spread payoff (Section 4.2.1, d=2):

        g(x) = max(0, x^2 - 0.9 * x^1) - max(0, x^2 - 1.1 * x^1)

    A spread on the outperformance of asset 2 over asset 1.
    (Equivalent to a bull spread on x^2/x^1 - 1.)

    Args:
        d: Number of assets (must be 2).
        k_low:  Lower multiplier (default 0.9).
        k_high: Upper multiplier (default 1.1).
    """

    def __init__(self, d: int = 2, k_low: float = 0.9, k_high: float = 1.1) -> None:
        super().__init__()
        assert d == 2, "OutperformerSpread is defined for d=2"
        self.d = d
        self.k_low = k_low
        self.k_high = k_high

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.d
        x1, x2 = x[:, 0], x[:, 1]
        long_call = torch.relu(x2 - self.k_low * x1)
        short_call = torch.relu(x2 - self.k_high * x1)
        return long_call - short_call

    def __repr__(self) -> str:
        return f"OutperformerSpread(k_low={self.k_low}, k_high={self.k_high})"


class BestOfButterfly(nn.Module):
    """
    Best-of butterfly payoff (Section 4.2.2):

        g(x) = max(0, X_max - K1) - 2*max(0, X_max - K_mid) + max(0, X_max - K2)

    where X_max = max(x^1, ..., x^d) and K_mid = (K1 + K2) / 2.

    Parameters from paper: K1=85, K2=115, sigma in [0.3,0.5], rho in [0.3,0.5],
    T=0.25, r=0.05, d=2. (Section 4.2.2)

    Args:
        d:  Number of assets.
        K1: Lower strike (default 85).
        K2: Upper strike (default 115).
    """

    def __init__(self, d: int, K1: float = 85.0, K2: float = 115.0) -> None:
        super().__init__()
        self.d = d
        self.K1 = K1
        self.K2 = K2
        self.K_mid = (K1 + K2) / 2.0

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.d
        x_max = x.max(dim=1).values  # [B]
        g = (torch.relu(x_max - self.K1)
             - 2.0 * torch.relu(x_max - self.K_mid)
             + torch.relu(x_max - self.K2))
        return g

    def __repr__(self) -> str:
        return f"BestOfButterfly(d={self.d}, K1={self.K1}, K2={self.K2})"


class GeoCallSpread(nn.Module):
    """
    Geometric call spread payoff (Section 4.2.3):

        g(x) = max(0, geo_mean(x) - K1) - max(0, geo_mean(x) - K2)

    where geo_mean(x) = (prod_{i=1}^d x^i)^{1/d}.

    Used for the high-dimensional experiments (d = 2, 5, 10, 20, 40, 80)
    with fixed correlation rho=0. (Section 4.2.3)

    Args:
        d:  Number of assets.
        K1: Lower strike (default 90).
        K2: Upper strike (default 110).
    """

    def __init__(self, d: int, K1: float = 90.0, K2: float = 110.0) -> None:
        super().__init__()
        self.d = d
        self.K1 = K1
        self.K2 = K2

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.d
        geo_mean = torch.exp(torch.log(x.clamp(min=1e-8)).mean(dim=1))  # [B]
        long_call = torch.relu(geo_mean - self.K1)
        short_call = torch.relu(geo_mean - self.K2)
        return long_call - short_call

    def __repr__(self) -> str:
        return f"GeoCallSpread(d={self.d}, K1={self.K1}, K2={self.K2})"


class CallSharpe(nn.Module):
    """
    Call Sharpe payoff (Section 4.2.4, d=1, path-dependent):

        g(X_T, V_T) = max(0, X_T - K) / sqrt(V_T)

    where V_T is the annualized realized monthly volatility (RV):

        V_T = (1/m) * sum_{k=1}^m [log(X_{t_k} / X_{t_{k-1}})]^2 * (12 / (T/m))

    This is a path-dependent payoff requiring the full path X_{t_0}, ..., X_{t_m}.
    The augmented state at time n is (X_n, A1_n, A2_n) where:
        A1_n = sum_{k past monitoring dates} [log ratio]^2
        A2_n = last observed monitoring price (for the next log ratio)

    Section 4.2.4 of arXiv:2605.06670.

    Args:
        d:           Number of assets (must be 1 for Call Sharpe).
        K:           Strike price (default 100).
        m:           Number of monthly monitoring dates (default 12 for T=1).
        T:           Time to maturity.
        annualization_factor: Annualizes realized vol (default 12 for monthly).
    """

    def __init__(
        self,
        d: int = 1,
        K: float = 100.0,
        m: int = 12,
        T: float = 1.0,
        annualization_factor: float = 12.0,
    ) -> None:
        super().__init__()
        assert d == 1, "CallSharpe as defined is for d=1"
        self.d = d
        self.K = K
        self.m = m
        self.T = T
        self.annualization_factor = annualization_factor

    def forward(self, x_T: Tensor, realized_vol: Tensor) -> Tensor:
        """
        Compute Call Sharpe payoff.

        Args:
            x_T:          Terminal asset price [B] or [B,1].
            realized_vol: Realized volatility V_T [B] or [B,1]. Must be > 0.

        Returns:
            Payoff [B].
        """
        if x_T.dim() == 2:
            x_T = x_T[:, 0]
        if realized_vol.dim() == 2:
            realized_vol = realized_vol[:, 0]
        payoff = torch.relu(x_T - self.K) / torch.sqrt(realized_vol.clamp(min=1e-8))
        return payoff

    def __repr__(self) -> str:
        return f"CallSharpe(K={self.K}, m={self.m}, T={self.T})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PAYOFF_REGISTRY = {
    "geo_outperformer": GeoOutperformer,
    "outperformer_spread": OutperformerSpread,
    "best_of_butterfly": BestOfButterfly,
    "geo_call_spread": GeoCallSpread,
    "call_sharpe": CallSharpe,
}


def build_payoff(name: str, d: int, K1: float = 90.0, K2: float = 110.0, **kwargs) -> nn.Module:
    """
    Instantiate a payoff function by name.

    Args:
        name: One of the PAYOFF_REGISTRY keys.
        d:    Number of assets.
        K1:   Lower strike.
        K2:   Upper strike.

    Returns:
        Payoff nn.Module.
    """
    if name not in PAYOFF_REGISTRY:
        raise ValueError(
            f"Unknown payoff '{name}'. Available: {list(PAYOFF_REGISTRY.keys())}"
        )
    cls = PAYOFF_REGISTRY[name]
    if name in ("geo_outperformer", "outperformer_spread"):
        return cls(d=d)
    elif name == "call_sharpe":
        return cls(d=d, K=K1, **kwargs)
    else:
        return cls(d=d, K1=K1, K2=K2)
