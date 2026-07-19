# Data — RED-COHORT-2026

## What you need

Two raw input files (not included in this repo due to size):

| File | Records | Size (est.) | Description |
|---|---|---|---|
| `pumpfun_buyers.jsonl` | 1,578,333 | ~800 MB | Buyer events from pump.fun (15-day window 2026-06-12 to 2026-06-26) |
| `pumpfun_launches.jsonl` | 1,315,257 | ~500 MB | Launch metadata |

## Option 1 — Use the Zenodo checkpoint (recommended for detection only)

The paper releases `sniper_cohorts_intra.jsonl.gz` on Zenodo — the Stage-1
checkpoint that lets you skip raw data ingestion and run only the detection
and analysis pipelines:

    DOI: https://zenodo.org/records/20978742

Download and place at `data/sniper_cohorts_intra.jsonl.gz`, then run:

    python detect.py --from-intra data/sniper_cohorts_intra.jsonl.gz

## Option 2 — Collect your own data via Solana RPC

The paper collected data via a passive Solana RPC observer (read-only listener).
Required fields per buyer event record:

```json
{
  "mint":      "string (Solana base-58 token address)",
  "wallet":    "string (buyer wallet address)",
  "slot":      "integer (Solana slot)",
  "blockTime": "integer (Unix timestamp, seconds)",
  "sol_in":    "float (SOL committed in this buy)",
  "tx_sig":    "string (transaction signature)",
  "rank":      "integer (buyer rank within launch, 1=first)"
}
```

Required fields per launch record:

```json
{
  "mint":               "string",
  "symbol":             "string",
  "name":               "string",
  "created_timestamp":  "integer (Unix seconds)",
  "initial_mcap_sol":   "float",
  "has_twitter":        "boolean",
  "has_website":        "boolean",
  "has_telegram":       "boolean",
  "description_len":    "integer"
}
```

Solana RPC endpoints: https://solana.com/rpc
pump.fun program ID: `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`

## Sample file (smoke tests)

`data/sample_buyers.jsonl` — 1,000 rows of synthetic buyer events in the
correct schema. Use this to verify your pipeline installation works:

    python detect.py --buyers data/sample_buyers.jsonl
