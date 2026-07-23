"""
utils/io_helpers.py
-------------------
JSONL streaming utilities and address anonymization helper.

Paper: Kamat (2026), Section 3.1 (data streams), Section 3.2 (anonymization).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Generator, List

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False


class JsonlStreamer:
    """
    Streams records from a JSONL or gzipped JSONL file one line at a time,
    avoiding loading all 1.5M records into memory at once.

    Paper reference: Section 3.1 — pumpfun_buyers.jsonl, pumpfun_launches.jsonl.
    """

    @staticmethod
    def stream(path: str) -> Generator[Dict, None, None]:
        """
        Yield one parsed JSON record per line from *path*.
        Supports both plain .jsonl and .jsonl.gz files.
        Uses orjson when available (10-30x faster than stdlib json).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"See data/README.md for instructions on obtaining pump.fun data."
            )

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if _USE_ORJSON:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)

    @staticmethod
    def write(records: List[Dict], path: str) -> None:
        """Write a list of dicts to a JSONL file at *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")

    def __repr__(self) -> str:
        return "JsonlStreamer()"


class AddressAnonymizer:
    """
    Truncates a Solana base-58 wallet address to first4...last4 characters
    for display, matching the paper's anonymization convention.

    Paper reference: Section 3.2 — e.g., 'F6zvmnwC..cxSi'.
    """

    @staticmethod
    def truncate(address: str) -> str:
        """Return 'XXXX..XXXX' form of a base-58 address."""
        if len(address) <= 8:
            return address
        return f"{address[:4]}..{address[-4:]}"

    def __repr__(self) -> str:
        return "AddressAnonymizer()"
