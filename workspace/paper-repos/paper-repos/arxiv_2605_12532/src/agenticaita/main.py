"""
main.py — AGENTICAITA system orchestrator and polling loop.

Paper: AGENTICAITA (arxiv:2605.12532)
Wires together all components: AZTE → IGP → SDP with Analyst/RM/Executor.
Runs the 60-second polling loop across all monitored assets.

Usage:
    python scripts/run_dryrun.py --config configs/default.yaml
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from agenticaita.agents.analyst import AnalystAgent
from agenticaita.agents.executor import ExecutorAgent
from agenticaita.agents.risk_manager import RiskManagerAgent
from agenticaita.data.mock_feed import MockMarketFeed
from agenticaita.memory.episodic import EpisodicMemory
from agenticaita.pipeline.igp import IGP
from agenticaita.pipeline.sdp import SDP
from agenticaita.scoring.cbd import CBD
from agenticaita.trigger.azte import AZTE
from agenticaita.utils.config import Config, load_config, set_seed

logger = logging.getLogger(__name__)


class AGENTICAITA:
    """
    Top-level system orchestrator.

    Manages the full polling loop:
      for each tick:
        for each asset:
          azte.update(asset, price) → trigger?
          igp.try_acquire() → admitted?
          sdp.run(trigger, ohlcv, l2, funding) → outcome
          igp.release()

    Args:
        config: Loaded Config instance.
        assets: List of symbols to monitor.
    """

    def __init__(self, config: Config, assets: List[str]) -> None:
        self.config = config
        self.assets = assets
        self._running = False

        # --- Core Components ---
        self.memory = EpisodicMemory(config.database.path)
        self.azte = AZTE(config.trigger)
        self.igp = IGP(config.igp)
        self.cbd = CBD(config.cbd)

        # --- LLM Agents ---
        self.analyst = AnalystAgent(config.llm)
        self.risk_manager = RiskManagerAgent(config.llm, config.risk_manager)
        self.executor = ExecutorAgent(
            mode=config.execution.mode,
            tor_socks_host=config.execution.tor_socks_host,
            tor_socks_port=config.execution.tor_socks_port,
        )

        # --- Pipeline ---
        self.sdp = SDP(
            analyst=self.analyst,
            risk_manager=self.risk_manager,
            executor=self.executor,
            memory=self.memory,
            cbd=self.cbd,
        )

        # --- Mock feed (replace with live MarketFeed for DEX integration) ---
        self.feed = MockMarketFeed(assets=assets, seed=config.seed)

    async def startup(self) -> None:
        """Initialize database and hot-restart AZTE rolling buffers."""
        await self.memory.connect()
        logger.info(f"[System] EpisodicMemory connected: {self.config.database.path}")

        # Hot-restart: restore AZTE buffers from persisted vol_history
        for asset in self.assets:
            returns = await self.memory.get_returns(asset, self.config.trigger.window_bars)
            prices = await self.memory.get_prices(asset, self.config.trigger.window_bars)
            if returns:
                self.azte.hot_restart(asset, returns, prices)

        logger.info(f"[System] AGENTICAITA ready — {len(self.assets)} assets, mode={self.config.execution.mode}")

    async def shutdown(self) -> None:
        """Graceful shutdown — flush pending writes."""
        self._running = False
        stats = self.sdp.session_stats()
        logger.info(f"[System] Session stats: {stats}")
        await self.memory.close()
        logger.info("[System] shutdown complete")

    async def run(self, max_ticks: Optional[int] = None) -> None:
        """
        Main 60-second polling loop.

        Paper: Section 4.1 — 'A 60-second polling loop computes the
        instantaneous return magnitude for each monitored asset.'

        Args:
            max_ticks: Stop after this many ticks (None = run indefinitely).
        """
        await self.startup()
        self._running = True
        tick = 0

        try:
            while self._running:
                if max_ticks is not None and tick >= max_ticks:
                    break

                # Advance mock feed one tick (replace with live price fetch)
                new_prices = self.feed.tick()

                # Persist volatility samples
                for asset, price in new_prices.items():
                    prev_prices = self.feed.get_price_history(asset, 2)
                    if len(prev_prices) >= 2:
                        r_t = abs(prev_prices[-1] - prev_prices[-2]) / prev_prices[-2]
                        await self.memory.append_vol(asset, r_t, price)

                # AZTE: check each asset for trigger
                for asset in self.assets:
                    price = self.feed.get_price(asset)
                    trigger = await self.azte.update(asset, price)

                    if trigger is None:
                        continue

                    # IGP: try to acquire pipeline lock
                    admitted = await self.igp.try_acquire(asset)
                    if not admitted:
                        await self.memory.log_event("pipeline_busy", asset, {
                            "z_score": trigger.z_score
                        })
                        continue

                    # SDP: run deliberative pipeline
                    try:
                        ohlcv = self.feed.get_ohlcv(asset, n=20)
                        l2 = self.feed.get_l2(asset)
                        funding = self.feed.get_funding_rate(asset)

                        outcome = await self.sdp.run(trigger, ohlcv, l2, funding)
                        logger.info(f"[Loop] {asset}: {outcome.outcome}")
                    finally:
                        self.igp.release(asset)

                tick += 1
                if max_ticks is None:
                    await asyncio.sleep(self.config.polling.interval_s)
                # (In test mode with max_ticks, no sleep for speed)

        except asyncio.CancelledError:
            logger.info("[System] polling loop cancelled")
        finally:
            await self.shutdown()


async def run_system(config_path: str, assets: List[str], max_ticks: Optional[int] = None) -> None:
    """Entry-point coroutine — called from scripts/run_dryrun.py."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    system = AGENTICAITA(cfg, assets)

    # Handle SIGINT/SIGTERM gracefully
    loop = asyncio.get_running_loop()

    def _stop():
        system._running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass  # Windows

    await system.run(max_ticks=max_ticks)
