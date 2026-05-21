"""
exchange.py — Exchange Adapter (Section 4.7)
Abstract base class + STUB implementation for DEX perpetuals order routing.

STUB: The paper does not name the exchange. Replace StubExchangeAdapter
with a concrete implementation (e.g., HyperliquidAdapter, DydxAdapter).

Privacy routing: Section 4.7 specifies dual-channel architecture:
  - Public channel: direct HTTPS for market data
  - Private channel: SOCKS5h → Tor → VPN for live order routing
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderSpec:
    asset: str
    side: Literal["long", "short"]
    size_usd: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_price: Optional[float] = None
    filled_size_usd: Optional[float] = None


class ExchangeAdapter(ABC):
    """Abstract base for DEX exchange adapters. Exchange not named in paper — implement subclass."""

    @abstractmethod
    async def place_order(self, order: "OrderSpec") -> Optional["OrderResult"]: ...

    @abstractmethod
    async def is_tor_active(self) -> bool: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    async def safety_gate_check(self) -> bool:
        """Section 4.7: require Tor AND exchange reachable before LIVE order."""
        tor_ok = await self.is_tor_active()
        exchange_ok = await self.health_check()
        if not tor_ok:
            logger.error("Safety gate FAILED: Tor not active")
        if not exchange_ok:
            logger.error("Safety gate FAILED: exchange not reachable")
        return tor_ok and exchange_ok


class StubExchangeAdapter(ExchangeAdapter):
    """
    STUB: Replace with a real DEX implementation.
    Paper (Section 4.7) does not name the exchange.
    """

    async def place_order(self, order: OrderSpec) -> Optional[OrderResult]:
        raise NotImplementedError(
            "STUB: Implement a real ExchangeAdapter subclass. See exchange.py."
        )

    async def is_tor_active(self) -> bool:
        import asyncio
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 9050), timeout=3.0)
            w.close(); await w.wait_closed()
            return True
        except Exception:
            return False

    async def health_check(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "StubExchangeAdapter(UNIMPLEMENTED)"


def get_exchange_adapter(name: str) -> ExchangeAdapter:
    adapters = {
        "stub": StubExchangeAdapter,
        # "hyperliquid": HyperliquidAdapter,
        # "dydx": DydxAdapter,
    }
    cls = adapters.get(name)
    if cls is None:
        raise ValueError(f"Unknown adapter '{name}'. Available: {list(adapters)}")
    return cls()
