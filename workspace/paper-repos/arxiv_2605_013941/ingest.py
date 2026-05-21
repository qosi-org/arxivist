#!/usr/bin/env python3
"""
ingest.py — Ingest conversation sessions into EVOLVEMEM memory store.

Reads a JSONL file of sessions and extracts typed memory units
into the SQLite store.

Usage:
    python ingest.py --sessions-file data/sessions.jsonl \\
                     --db-path evolvemem.db \\
                     --config configs/config.yaml
"""

import argparse
import json
import logging

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="EVOLVEMEM: Ingest conversation sessions")
    parser.add_argument("--sessions-file", required=True, help="JSONL file of sessions")
    parser.add_argument("--db-path", required=True, help="SQLite memory store path")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--scope", default="user:default|workspace:default|session:default",
                        help="Memory scope identifier")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["memory"]["db_path"] = args.db_path

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    sessions = []
    with open(args.sessions_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    logger.info(f"Loaded {len(sessions)} sessions from {args.sessions_file}")

    from src.evolvemem.evolvemem import EvolveMem
    em = EvolveMem(config_path=tmp_config)
    em.ingest_sessions(sessions)

    logger.info(f"Ingestion complete. Store: {em.store.size()} active memory units.")
    os.unlink(tmp_config)


if __name__ == "__main__":
    main()
