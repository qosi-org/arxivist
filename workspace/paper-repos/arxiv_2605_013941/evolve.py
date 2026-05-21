#!/usr/bin/env python3
"""
evolve.py — Run EVOLVEMEM self-evolution loop.

Runs the self-evolution loop (Algorithm 1) on a provided QA set,
starting from the initial configuration and discovering optimal retrieval parameters.

Usage:
    python evolve.py --qa-file data/locomo_qa.jsonl --db-path evolvemem.db \\
                     --config configs/config.yaml --max-rounds 7 \\
                     --output-config outputs/best_config.json
"""

import argparse
import json
import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evolve")


def main():
    parser = argparse.ArgumentParser(description="EVOLVEMEM: Run self-evolution loop")
    parser.add_argument("--qa-file", required=True, help="JSONL file with QA pairs (q, ref, category)")
    parser.add_argument("--db-path", required=True, help="SQLite memory store path")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override max evolution rounds")
    parser.add_argument("--output-config", default="outputs/best_config.json", help="Output path for best config")
    parser.add_argument("--benchmark", choices=["locomo", "membench"], default="locomo",
                        help="Benchmark type for diagnosis prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Limit to first 20 QA pairs for quick test")
    args = parser.parse_args()

    # Seed
    import random
    random.seed(args.seed)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.db_path:
        cfg["memory"]["db_path"] = args.db_path
    if args.max_rounds:
        cfg["evolution"]["max_rounds"] = args.max_rounds

    # Save modified config to temp
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    # Load QA pairs
    qa_pairs = []
    with open(args.qa_file) as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(json.loads(line))

    if args.debug:
        qa_pairs = qa_pairs[:20]
        logger.info(f"DEBUG mode: using {len(qa_pairs)} QA pairs")
    else:
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {args.qa_file}")

    # Initialize EvolveMem
    from src.evolvemem.evolvemem import EvolveMem
    from src.evolvemem.evolution.diagnosis import DiagnosisModule

    em = EvolveMem(config_path=tmp_config)
    em.diagnosis = DiagnosisModule(em.llm_client, benchmark=args.benchmark)

    # Run evolution
    logger.info("Starting self-evolution loop...")
    best_config = em.evolve(qa_pairs, update_config=True)

    # Save best config
    os.makedirs(os.path.dirname(args.output_config) or ".", exist_ok=True)
    em.save_state(args.output_config)
    logger.info(f"Best config saved to {args.output_config}")
    logger.info(f"Best config: {best_config}")

    os.unlink(tmp_config)


if __name__ == "__main__":
    main()
