#!/usr/bin/env python3
"""
answer.py — Interactive single-query inference with EVOLVEMEM.

Usage:
    python answer.py --query "What books does Alice like?" \\
                     --db-path evolvemem.db \\
                     --config configs/config.yaml \\
                     --category 2
"""

import argparse
import logging

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="EVOLVEMEM: Single-query inference")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--db-path", required=True, help="SQLite memory store path")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--evolved-config", default=None, help="Evolved config JSON path")
    parser.add_argument("--category", type=int, default=None, help="LoCoMo question category (1-5)")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved memories")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["memory"]["db_path"] = args.db_path

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    from src.evolvemem.evolvemem import EvolveMem
    em = EvolveMem(config_path=tmp_config)

    if args.evolved_config:
        em.load_state(args.evolved_config)

    if args.verbose:
        retrieved = em.retriever.retrieve(args.query, category=args.category)
        print("\n--- Retrieved memories ---")
        for i, unit in enumerate(retrieved, 1):
            print(f"{i}. [{unit.memory_type}] {unit.content}")
        print("---\n")

    answer = em.answer(args.query, category=args.category)
    print(f"Answer: {answer}")

    os.unlink(tmp_config)


if __name__ == "__main__":
    main()
