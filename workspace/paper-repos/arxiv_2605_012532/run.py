"""
run.py — AGENTICAITA Main Trading Loop Entrypoint
arxiv:2605.12532 (Letteri 2026)

Usage:
    python run.py --config configs/config.yaml --mode DRY_RUN
    python run.py --config configs/config.yaml --mode LIVE  # requires Tor + real exchange
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.agenticaita.config import AgenticAITAConfig
from src.agenticaita.azte import AdaptiveZScoreTriggerEngine
from src.agenticaita.cbd import CorrelationBreakDiversification
from src.agenticaita.igp import InferenceGatingProtocol
from src.agenticaita.memory import EpisodicMemory
from src.agenticaita.market_data import MarketDataFeed
from src.agenticaita.ollama_client import OllamaClient
from src.agenticaita.exchange import build_exchange_adapter
from src.agenticaita.agents.analyst import AnalystAgent
from src.agenticaita.agents.risk_manager import RiskManagerAgent
from src.agenticaita.agents.executor import ExecutorAgent
from src.agenticaita.pipeline import SequentialDeliberativePipeline
from src.agenticaita.monitor import MonitoringPoller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGENTICAITA — Deliberative Multi-Agent Autonomous Trading System"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["DRY_RUN", "LIVE"], help="Override system.mode in config")
    parser.add_argument("--assets", help="Override asset list path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


async def build_system(cfg: AgenticAITAConfig) -> tuple[MonitoringPoller, list]:
    """Construct and wire all system components."""
    # Core components
    memory = EpisodicMemory(cfg.database.path)
    await memory.initialize_db()

    azte = AdaptiveZScoreTriggerEngine(cfg.azte, db=memory)
    cbd = CorrelationBreakDiversification(cfg.cbd)
    igp = InferenceGatingProtocol(global_cooldown_s=cfg.igp.global_cooldown_s)

    market_data = MarketDataFeed(
        exchange_id=cfg.exchange.adapter,
        ohlcv_limit=cfg.market_data.ohlcv_limit,
        l2_depth=cfg.market_data.l2_depth,
    )

    ollama = OllamaClient(cfg.ollama, db=memory)

    # Verify Ollama availability
    if not await ollama.health_check():
        logging.warning(f"Ollama not reachable at {cfg.ollama.base_url} — LLM calls will fail")

    exchange = build_exchange_adapter(cfg.exchange.adapter)

    # Agents
    analyst = AnalystAgent(llm=ollama)
    risk_manager = RiskManagerAgent(cfg=cfg.risk_manager, llm=ollama)
    executor = ExecutorAgent(exchange=exchange, db=memory, mode=cfg.system.mode)

    # Pipeline
    pipeline = SequentialDeliberativePipeline(
        cfg=cfg, igp=igp, cbd=cbd, memory=memory,
        market_data=market_data,
        analyst=analyst, risk_manager=risk_manager, executor=executor,
    )

    # Monitor
    monitor = MonitoringPoller(cfg=cfg, azte=azte, cbd=cbd, pipeline=pipeline,
                               memory=memory, market_data=market_data)

    closeables = [memory, market_data, ollama]
    return monitor, closeables


async def main() -> None:
    args = parse_args()

    cfg = AgenticAITAConfig.from_yaml(args.config)

    if args.mode:
        cfg.system.mode = args.mode
    if args.log_level:
        cfg.system.log_level = args.log_level

    setup_logging(cfg.system.log_level)
    logger = logging.getLogger("agenticaita")

    logger.info(f"AGENTICAITA starting | mode={cfg.system.mode} | model={cfg.ollama.model}")

    if cfg.system.mode == "LIVE":
        logger.warning("⚠  LIVE MODE: real orders will be placed. Ensure Tor is running.")

    monitor, closeables = await build_system(cfg)
    monitor.load_assets(args.assets)

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown(sig):
        logger.info(f"Signal {sig.name} received — shutting down")
        monitor.stop()

    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda s=s: _shutdown(s))

    try:
        await monitor.poll_loop()
    finally:
        logger.info("Closing resources...")
        for c in closeables:
            try:
                await c.close()
            except Exception:
                pass
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
