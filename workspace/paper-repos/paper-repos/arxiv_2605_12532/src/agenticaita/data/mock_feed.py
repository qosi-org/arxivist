"""
data/mock_feed.py — Deterministic mock market data feed.

Used for offline testing and CI without a live DEX connection.
Generates synthetic OHLCV + L2 data with configurable volatility regimes.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


class MockMarketFeed:
    """
    Deterministic synthetic market feed for dry-run testing.

    Simulates price walks with configurable volatility and occasional
    Z-score spikes to trigger the AZTE pipeline.

    Args:
        assets: List of asset symbols to simulate.
        seed: Random seed for reproducibility.
        base_prices: Optional dict of {symbol: initial_price}.
    """

    def __init__(
        self,
        assets: List[str],
        seed: int = 42,
        base_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        self.assets = assets
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._prices: Dict[str, float] = {}
        self._price_history: Dict[str, List[float]] = {a: [] for a in assets}
        self._ohlcv_history: Dict[str, List[List[float]]] = {a: [] for a in assets}
        self._tick = 0

        default_prices = {
            "BTC": 84000.0, "ETH": 1600.0, "SOL": 130.0, "FARTCOIN": 0.85,
            "XPL": 0.012, "CC": 0.055, "HEMI": 3.20, "AVAX": 18.0,
        }
        if base_prices:
            default_prices.update(base_prices)

        for asset in assets:
            self._prices[asset] = default_prices.get(asset, self.rng.uniform(0.01, 100.0))
            self._price_history[asset] = [self._prices[asset]]

    def tick(self, spike_asset: Optional[str] = None) -> Dict[str, float]:
        """
        Advance all asset prices by one 60-second tick.

        Args:
            spike_asset: If specified, inject a large return to this asset
                         to trigger AZTE (useful for testing the pipeline).

        Returns:
            Dict of {asset: new_price}.
        """
        self._tick += 1
        new_prices = {}

        for asset in self.assets:
            p = self._prices[asset]
            # Correlated assets (high BTC correlation)
            btc_corr = 0.8 if asset in ("ETH", "AVAX", "SOL") else 0.1

            # Base return: small random walk
            base_return = self.np_rng.normal(0, 0.001)

            if asset == "BTC":
                btc_factor = base_return
            else:
                btc_factor = btc_corr * self.np_rng.normal(0, 0.001)
                idio_factor = (1.0 - btc_corr) * self.np_rng.normal(0, 0.0015)
                base_return = btc_factor + idio_factor

            # Spike injection for testing
            if spike_asset == asset:
                base_return += self.rng.choice([-1, 1]) * self.rng.uniform(0.005, 0.02)

            new_p = max(p * (1.0 + base_return), p * 1e-6)
            self._prices[asset] = new_p
            self._price_history[asset].append(new_p)
            new_prices[asset] = new_p

            # Build OHLCV bar
            open_p = p
            close_p = new_p
            high_p = max(open_p, close_p) * (1 + abs(self.np_rng.normal(0, 0.0005)))
            low_p = min(open_p, close_p) * (1 - abs(self.np_rng.normal(0, 0.0005)))
            volume = abs(self.np_rng.normal(100000, 30000))
            bar = [open_p, high_p, low_p, close_p, volume]
            self._ohlcv_history[asset].append(bar)

        return new_prices

    def get_ohlcv(self, asset: str, n: int = 20) -> List[List[float]]:
        """Return last n OHLCV bars for an asset."""
        return self._ohlcv_history[asset][-n:]

    def get_l2(self, asset: str) -> dict:
        """Generate a synthetic L2 orderbook snapshot."""
        mid = self._prices[asset]
        spread_pct = 0.0005
        bids = [[mid * (1 - spread_pct * (i + 1)), self.rng.uniform(100, 5000)] for i in range(5)]
        asks = [[mid * (1 + spread_pct * (i + 1)), self.rng.uniform(100, 5000)] for i in range(5)]
        return {"bids": bids, "asks": asks, "mid": mid}

    def get_price(self, asset: str) -> float:
        """Return current price for an asset."""
        return self._prices[asset]

    def get_funding_rate(self, asset: str) -> float:
        """Synthetic funding rate (contango regime ~ small positive)."""
        return self.np_rng.normal(0.0001, 0.00005)

    def get_price_history(self, asset: str, n: int = 30) -> List[float]:
        """Return last n prices for CBD correlation computation."""
        return self._price_history[asset][-n:]
