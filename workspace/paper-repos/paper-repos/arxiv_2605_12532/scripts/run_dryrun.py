#!/usr/bin/env python3
"""
scripts/run_dryrun.py — Main entry point for AGENTICAITA dry-run session.

Paper: AGENTICAITA (arxiv:2605.12532)
Runs the full agentic pipeline in DRY_RUN mode (no real order placement).

Usage:
    python scripts/run_dryrun.py --config configs/default.yaml --assets assets.txt
    python scripts/run_dryrun.py --config configs/default.yaml --assets BTC ETH SOL --ticks 100

Examples:
    # Quick smoke test (10 ticks, mock data):
    python scripts/run_dryrun.py --ticks 10

    # Full session on a custom asset list:
    python scripts/run_dryrun.py --assets BTC ETH SOL AVAX --log-level DEBUG
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agenticaita.main import run_system


DEFAULT_ASSETS = [
    "BTC", "ETH", "SOL", "AVAX", "FARTCOIN", "XPL", "CC", "HEMI",
    "DOGE", "ADA", "XRP", "DOT", "ETC", "BCH",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AGENTICAITA in DRY_RUN mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--assets", nargs="*", default=None,
        help="Asset symbols to monitor (space-separated). Overrides config.",
    )
    parser.add_argument(
        "--assets-file", type=str, default=None,
        help="Path to newline-separated file of asset symbols.",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Override database path from config.",
    )
    parser.add_argument(
        "--ticks", type=int, default=None,
        help="Stop after N polling ticks. None = run indefinitely.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_dryrun")

    # Resolve asset list
    assets = DEFAULT_ASSETS
    if args.assets_file:
        with open(args.assets_file) as f:
            assets = [line.strip() for line in f if line.strip()]
    if args.assets:
        assets = args.assets

    logger.info(f"Starting AGENTICAITA DRY_RUN")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Assets:  {assets}")
    logger.info(f"Ticks:   {args.ticks or 'unlimited'}")

    asyncio.run(run_system(
        config_path=args.config,
        assets=assets,
        max_ticks=args.ticks,
    ))


if __name__ == "__main__":
    main()
