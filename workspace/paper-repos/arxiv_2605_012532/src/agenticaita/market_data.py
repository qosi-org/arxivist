"""
market_data.py — Async Market Data Feed (Section 4.1, 4.2)
Fetches OHLCV, L2 orderbook, and funding rates via ccxt async.

Public channel: direct HTTPS (Section 4.7 — market data not privacy-gated).
Exchange is configurable; defaults to the ccxt exchange specified in config.

STUB: The specific DEX used in the paper was not named.
      Configure exchange_id in config.yaml for your target DEX.
      ccxt supports Hyperliquid, dYdX, and others.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

from .schemas import OHLCVBar, L2Orderbook, L2Level, MarketContext

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """
    Async market data fetcher.
    Wraps ccxt for OHLCV + orderbook + funding rate retrieval.
    Section 4.1, 4.2 of arxiv:2605.12532.
    """

    def __init__(self, exchange_id: str = "stub", ohlcv_limit: int = 20, l2_depth: int = 20) -> None:
        self.exchange_id = exchange_id
        self.ohlcv_limit = ohlcv_limit
        self.l2_depth = l2_depth
        self._exchange = None

    async def _get_exchange(self):
        """Lazy-initialize ccxt async exchange."""
        if self._exchange is None:
            try:
                import ccxt.async_support as ccxt
                if self.exchange_id == "stub":
                    logger.warning("MarketDataFeed: using stub — no real data will be fetched")
                    return None
                cls = getattr(ccxt, self.exchange_id)
                self._exchange = cls({"enableRateLimit": True})
            except (ImportError, AttributeError) as e:
                logger.error(f"ccxt exchange init failed: {e}")
                return None
        return self._exchange

    async def get_price(self, asset: str) -> float:
        """Fetch current mid price for asset."""
        exchange = await self._get_exchange()
        if exchange is None:
            return 0.0
        try:
            ticker = await exchange.fetch_ticker(asset)
            return float(ticker.get("last") or ticker.get("close") or 0.0)
        except Exception as e:
            logger.error(f"get_price {asset}: {e}")
            return 0.0

    async def get_ohlcv(self, asset: str) -> list[OHLCVBar]:
        """
        Fetch last N 1-minute OHLCV bars.
        Section 4.2: 20-bar 1-minute context window.
        """
        exchange = await self._get_exchange()
        if exchange is None:
            return []
        try:
            raw = await exchange.fetch_ohlcv(asset, timeframe="1m", limit=self.ohlcv_limit)
            return [
                OHLCVBar(
                    timestamp=datetime.utcfromtimestamp(row[0] / 1000),
                    open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5],
                )
                for row in raw
            ]
        except Exception as e:
            logger.error(f"get_ohlcv {asset}: {e}")
            return []

    async def get_l2_orderbook(self, asset: str) -> L2Orderbook:
        """Fetch live L2 orderbook snapshot."""
        exchange = await self._get_exchange()
        if exchange is None:
            return L2Orderbook(asset=asset, timestamp=datetime.utcnow(), bids=[], asks=[])
        try:
            raw = await exchange.fetch_order_book(asset, limit=self.l2_depth)
            bids = [L2Level(price=b[0], size=b[1]) for b in raw.get("bids", [])]
            asks = [L2Level(price=a[0], size=a[1]) for a in raw.get("asks", [])]
            return L2Orderbook(asset=asset, timestamp=datetime.utcnow(), bids=bids, asks=asks)
        except Exception as e:
            logger.error(f"get_l2_orderbook {asset}: {e}")
            return L2Orderbook(asset=asset, timestamp=datetime.utcnow(), bids=[], asks=[])

    async def get_funding_rate(self, asset: str) -> float:
        """Fetch current perpetual funding rate."""
        exchange = await self._get_exchange()
        if exchange is None:
            return 0.0
        try:
            info = await exchange.fetch_funding_rate(asset)
            return float(info.get("fundingRate") or 0.0)
        except Exception as e:
            logger.debug(f"get_funding_rate {asset}: {e}")
            return 0.0

    async def build_market_context(
        self,
        asset: str,
        z_score: float,
        r_t: float,
        omega: float,
        memory_briefing: str,
    ) -> MarketContext:
        """Assemble full MarketContext for Analyst agent."""
        price = await self.get_price(asset)
        ohlcv = await self.get_ohlcv(asset)
        l2 = await self.get_l2_orderbook(asset)
        funding = await self.get_funding_rate(asset)
        return MarketContext(
            asset=asset,
            timestamp=datetime.utcnow(),
            current_price=price,
            ohlcv=ohlcv,
            l2_orderbook=l2,
            funding_rate=funding,
            z_score=z_score,
            return_magnitude=r_t,
            omega=omega,
            memory_briefing=memory_briefing,
        )

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()

    def __repr__(self) -> str:
        return f"MarketDataFeed(exchange={self.exchange_id}, ohlcv_limit={self.ohlcv_limit})"
