"""
io/loader.py
------------
Reads pumpfun_buyers.jsonl and pumpfun_launches.jsonl into typed DataFrames.

Paper: Kamat (2026), Section 3.1 — Corpus description and schema.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from red_cohort.utils.io_helpers import JsonlStreamer


# Expected schema columns for validation
_BUYER_COLS = {"mint", "wallet", "blockTime", "sol_in", "rank"}
_LAUNCH_COLS = {"mint", "created_timestamp"}


class DataLoader:
    """
    Loads and validates the two primary data streams described in Section 3.1.

    Args:
        chunk_size: Number of records to accumulate before converting to DataFrame.
                    Controls peak memory usage during streaming.

    Paper reference:
        Section 3.1 — buyer events (1,578,333 records) and launch metadata (1,315,257 records).
    """

    def __init__(self, chunk_size: int = 100_000) -> None:
        self.chunk_size = chunk_size

    def load_buyers(self, path: str) -> pd.DataFrame:
        """
        Stream pumpfun_buyers.jsonl and return a DataFrame with columns:
        {mint, wallet, slot, blockTime, sol_in, tx_sig, rank}

        Rank ties at the same blockTime are broken by tx_sig (lexicographic),
        providing a deterministic canonical order consistent with Solana indexer ordering.
        # TODO: verify tie-breaking strategy with paper author (SIR confidence 0.65)

        Returns:
            pd.DataFrame with dtype-cast columns. ~1.58M rows for the paper corpus.
        """
        streamer = JsonlStreamer()
        records = []
        chunks = []

        for rec in streamer.stream(path):
            records.append({
                "mint":      str(rec.get("mint", "")),
                "wallet":    str(rec.get("wallet", "")),
                "slot":      int(rec.get("slot", 0)),
                "blockTime": int(rec.get("blockTime", 0)),
                "sol_in":    float(rec.get("sol_in", 0.0)),
                "tx_sig":    str(rec.get("tx_sig", "")),
                "rank":      int(rec.get("rank", 0)),
            })
            if len(records) >= self.chunk_size:
                chunks.append(pd.DataFrame(records))
                records = []

        if records:
            chunks.append(pd.DataFrame(records))

        if not chunks:
            raise ValueError(f"No buyer records found in {path}")

        df = pd.concat(chunks, ignore_index=True)
        self.validate_schema(df, "buyers")
        return df

    def load_launches(self, path: str) -> pd.DataFrame:
        """
        Stream pumpfun_launches.jsonl and return a DataFrame with columns:
        {mint, symbol, name, created_timestamp, initial_mcap_sol,
         has_twitter, has_website, has_telegram, description_len}

        Returns:
            pd.DataFrame. ~1.315M rows for the paper corpus.
        """
        streamer = JsonlStreamer()
        records = []
        chunks = []

        for rec in streamer.stream(path):
            records.append({
                "mint":               str(rec.get("mint", "")),
                "symbol":             str(rec.get("symbol", "")),
                "name":               str(rec.get("name", "")),
                "created_timestamp":  int(rec.get("created_timestamp", 0)),
                "initial_mcap_sol":   float(rec.get("initial_mcap_sol", 0.0)),
                "has_twitter":        bool(rec.get("has_twitter", False)),
                "has_website":        bool(rec.get("has_website", False)),
                "has_telegram":       bool(rec.get("has_telegram", False)),
                "description_len":    int(rec.get("description_len", 0)),
            })
            if len(records) >= self.chunk_size:
                chunks.append(pd.DataFrame(records))
                records = []

        if records:
            chunks.append(pd.DataFrame(records))

        if not chunks:
            raise ValueError(f"No launch records found in {path}")

        df = pd.concat(chunks, ignore_index=True)
        self.validate_schema(df, "launches")
        return df

    def load_intra(self, path: str) -> pd.DataFrame:
        """
        Load the Stage-1 checkpoint sniper_cohorts_intra.jsonl.gz,
        bypassing raw buyer ingestion and Stage-1 extraction.

        This file is part of the RED-COHORT-2026-v1 Zenodo release.
        """
        streamer = JsonlStreamer()
        records = list(streamer.stream(path))
        if not records:
            raise ValueError(f"No intra records found in {path}")
        return pd.DataFrame(records)

    def validate_schema(self, df: pd.DataFrame, schema: str) -> bool:
        """Raise ValueError if required columns are missing."""
        required = _BUYER_COLS if schema == "buyers" else _LAUNCH_COLS
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Schema '{schema}' missing columns: {missing}")
        return True

    def __repr__(self) -> str:
        return f"DataLoader(chunk_size={self.chunk_size})"
