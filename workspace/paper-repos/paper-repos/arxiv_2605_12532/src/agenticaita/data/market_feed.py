"""
data/market_feed.py — Abstract market data feed interface.

Paper: AGENTICAITA (arxiv:2605.12532), Section 3
Market data flows through the public HTTPS channel (direct, unauthenticated).
Authenticated orders route through the private Tor+VPN channel (execution/routing.py).

SIR Ambiguity: The DEX exchange is not identified in the paper.
This module defines the abstract interface; a concrete implementation must be
provided for the target DEX. MockMarketFeed (data/mock_feed.py) is available
for offline testing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MarketFeed(ABC):
    """
    Abstract base class for market data feeds.

    STUB: Subclass this and implement all abstract methods for your target DEX.
    See data/mock_feed.py for a fully working synthetic implementation.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to market data source."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    async def get_price(self, asset: str) -> float:
        """Fetch current last-traded price for asset."""
        ...

    @abstractmethod
    async def get_ohlcv(self, asset: str, n_bars: int = 20) -> List[List[float]]:
        """
        Fetch n_bars of 1-minute OHLCV candles.

        Returns list of [open, high, low, close, volume].
        """
        ...

    @abstractmethod
    async def get_l2(self, asset: str) -> dict:
        """
        Fetch live L2 orderbook snapshot.

        Returns {"bids": [[price, size], ...], "asks": [[price, size], ...]}.
        """
        ...

    @abstractmethod
    async def get_funding_rate(self, asset: str) -> Optional[float]:
        """Fetch current perpetual funding rate."""
        ...

    @abstractmethod
    async def list_assets(self) -> List[str]:
        """Return list of all available asset symbols."""
        ...
