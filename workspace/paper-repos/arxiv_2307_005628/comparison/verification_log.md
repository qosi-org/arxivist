# Verification Log — DNAGPT (arxiv_2307_005628)

## Run metadata
- Architecture verification: 2026-07-24, local CPU (torch 2.5.0+cpu)
- GSR fine-tuning: **pending** — needs official .pth (Google Drive) + real DeepGSR data
- SIR version used: 1
- Manual review required: No

## Part A — architecture verification (executed)

```
pytest tests/ -q        # 8 passed (local CPU)
```
Checks: non-overlap k-mer count (N/k), 19,530-kmer vocab (S1.3), GC content, GSR
classification template, dual-head forward shapes, joint sequence+number forward,
combined loss L = 0.01*MSE + CE (Eq 1), GSR head learns synthetic signal.

Model build:
```
DNAGPT-M -> DNAGPT(layers=12, hidden=768, heads=12, params=116.3M)   # ~ paper 0.1B
```

Entrypoint sanity:
```
python train.py --help        # OK
python train.py --config configs/config_debug.yaml --dry-run   # builds model + tokenizer + trainer
```

## Part B — GSR fine-tuning (pending)

To finalize:
1. Copy each checkpoint's Google Drive ID into DRIVE_IDS in data/download.py (from
   https://github.com/TencentAILabHealthcare/DNAGPT), then
   `python data/download.py --weights dna_gpt0.1b_m`.
2. Assemble real DeepGSR PAS/TIS CSVs (data/README_data.md).
3. `python train.py --config configs/config.yaml`; paste `[done]` acc/mcc back to ArXivist.
4. reproducibility_score.json + benchmark_comparison.md updated with real values.

## Config
- Stock configs/config.yaml (DNAGPT-M, lr 3e-5 `#ASSUMED`-adjacent from Fig S4, 10 ep, bs 8).
- No user modifications recorded.

## Integrity
- No downstream metric has been fabricated. Pending fields are explicitly `null` until the
  user's run supplies real numbers. Architecture facts (116.3M params, 8/8 tests, 19,530
  vocab) are measured, not asserted.
