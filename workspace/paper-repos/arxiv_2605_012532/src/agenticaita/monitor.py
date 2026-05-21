"""
monitor.py — 60-Second Monitoring Poller (Section 4.1)
Polls all monitored assets on a 60-second interval, feeds prices into
the AZTE, and dispatches trigger events to the SDP pipeline.

Architecture note: each asset is polled independently via asyncio.gather.
The IGP (inside the SDP) handles concurrency — the monitor fires triggers
freely; the pipeline serializes execution.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .azte import AdaptiveZScoreTriggerEngine
from .cbd import CorrelationBreakDiversification
from .pipeline import SequentialDeliberativePipeline
from .memory import EpisodicMemory
from .market_data import MarketDataFeed
from .config import AgenticAITAConfig

logger = logging.getLogger(__name__)


class MonitoringPoller:
    """
    60-second asset monitoring loop.
    Section 4.1 of arxiv:2605.12532.

    Polls each asset independently; feeds prices into AZTE (Eq. 1–3).
    Trigger events are forwarded to the SDP for agent deliberation.
    BTC price is tracked separately for CBD decorrelation computation (Eq. 9).
    """

    BTC_SYMBOL = "BTC/USDT:USDT"  # Adjust symbol format to your DEX convention

    def __init__(
        self,
        cfg: AgenticAITAConfig,
        azte: AdaptiveZScoreTriggerEngine,
        cbd: CorrelationBreakDiversification,
        pipeline: SequentialDeliberativePipeline,
        memory: EpisodicMemory,
        market_data: MarketDataFeed,
    ) -> None:
        self.cfg = cfg
        self.azte = azte
        self.cbd = cbd
        self.pipeline = pipeline
        self.memory = memory
        self.market_data = market_data
        self._assets: list[str] = []
        self._running = False

    def load_assets(self, path: Optional[str] = None) -> None:
        """Load asset symbols from file (one symbol per line)."""
        asset_path = Path(path or self.cfg.assets.monitor_list_path)
        if not asset_path.exists():
            logger.warning(f"Asset list not found: {asset_path} — using empty list")
            return
        symbols = [
            line.strip() for line in asset_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self._assets = symbols
        logger.info(f"Loaded {len(symbols)} assets from {asset_path}")

    async def poll_asset(self, asset: str) -> None:
        """Poll one asset: update AZTE, CBD, dispatch trigger if fired."""
        timestamp = datetime.utcnow()

        price = await self.market_data.get_price(asset)
        if price == 0.0:
            logger.debug(f"Skipping {asset}: price fetch returned 0")
            return

        # Update CBD price buffer for decorrelation
        self.cbd.update_price(asset, price, self.cfg.azte.rolling_window)

        # AZTE: compute Z-score and check trigger (Eq. 1–3)
        trigger = self.azte.update(
            asset=asset,
            price=price,
            timestamp=timestamp,
            per_asset_cooldown_s=self.cfg.igp.per_asset_cooldown_s,
        )

        # Persist vol sample for hot-restart
        if self.azte._baselines.get(asset):
            baseline = self.azte._baselines[asset]
            z = self.azte.compute_z_score(
                self.azte.compute_return_magnitude(price, list(baseline.prices)[-2])
                if len(baseline.prices) >= 2 else 0.0,
                baseline.returns,
            )
            await self.memory.store_vol_sample(asset, price, z, 0.0)

        if trigger:
            logger.info(f"Dispatching trigger: {asset} → SDP")
            # Fire and forget — IGP handles serialization internally
            asyncio.create_task(self.pipeline.handle_trigger(trigger))

    async def poll_loop(self) -> None:
        """Main 60-second polling loop across all assets."""
        self._running = True
        logger.info(f"MonitoringPoller: starting loop for {len(self._assets)} assets")

        # Hot-restart: pre-load vol history from DB
        for asset in self._assets:
            await self.azte.load_history(asset, self.memory)

        while self._running:
            tick_start = asyncio.get_event_loop().time()

            # Fetch BTC price for CBD decorrelation baseline
            btc_price = await self.market_data.get_price(self.BTC_SYMBOL)
            if btc_price > 0:
                self.cbd.update_btc_price(btc_price, self.cfg.azte.rolling_window)

            # Poll all assets concurrently (I/O-bound; IGP handles downstream serialization)
            tasks = [self.poll_asset(asset) for asset in self._assets]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Log pipeline friction every 10 ticks
            if self.pipeline.stats["total_invocations"] % 10 == 0 and \
               self.pipeline.stats["total_invocations"] > 0:
                logger.info(f"Friction rate: {self.pipeline.friction_rate():.1%} | Stats: {self.pipeline.stats}")

            # Sleep for the remainder of the polling interval
            elapsed = asyncio.get_event_loop().time() - tick_start
            sleep_time = max(0, self.cfg.azte.polling_interval_s - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
        logger.info("MonitoringPoller: stop requested")

    def __repr__(self) -> str:
        return f"MonitoringPoller(assets={len(self._assets)}, interval={self.cfg.azte.polling_interval_s}s)"
