# RED-COHORT-2026

**Reproducibility package for:**
> Coordinated Sniper Cohorts on Pump.fun: Detection of 1,012 Persistent Wallet Rings and the Limits of Naive Causal Inference for First-Hour Buyer Flow
> Arati Uday Kamat — Independent Researcher · ORCID: 0009-0000-4781-312X
> DOI: [10.5281/zenodo.20978741](https://doi.org/10.5281/zenodo.20978741) | Version 1.0 (2026-06-27)

---

## What this paper does

This paper identifies and characterises **coordinated sniper cohorts** on the Solana pump.fun bonding-curve marketplace — small groups of wallets that systematically appear among the first 10 buyers of many token launches. A two-stage detection pipeline (intra-launch first-buyer-window extraction → cross-launch union-find on co-occurrence graphs) surfaces 1,012 persistent wallet cohorts from 1,578,333 buyer events across 166,098 launches over 15 days.

The paper's central methodological finding is a **causal refutation**: while cohort-touched launches show a naïve +132.3% lift in first-30-minute buyer flow, an activity-matched placebo design yields a *larger* +216.3% lift with no confidence-interval overlap, indicating the association reflects launch-quality selection rather than a coordination-specific causal effect.

---

## Quick start

```bash
# 1. Clone and install
git clone <this-repo> && cd red-cohort-2026
pip install -e .

# 2. Obtain data (see data/README.md)
# Fastest path: download Zenodo Stage-1 checkpoint
# wget https://zenodo.org/records/20978742/files/sniper_cohorts_intra.jsonl.gz -P data/

# 3a. Run detection from Stage-1 checkpoint (no raw JSONL needed)
python detect.py --from-intra data/sniper_cohorts_intra.jsonl.gz

# 3b. Or run from raw data
python detect.py --buyers data/pumpfun_buyers.jsonl

# 4. Produce descriptive stats + figures
python analyze.py --cohorts results/sniper_cohorts.jsonl

# 5. Run full causal analysis
python causal.py \
    --buyers data/pumpfun_buyers.jsonl \
    --launches data/pumpfun_launches.jsonl \
    --cohorts results/sniper_cohorts.jsonl

# 6. Or run everything in one command
python run_all.py \
    --buyers data/pumpfun_buyers.jsonl \
    --launches data/pumpfun_launches.jsonl
```

---

## Installation

### pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda env create -f environment.yaml
conda activate red-cohort
pip install -e .
```

### Docker
```bash
# Build image
docker build -f docker/Dockerfile -t red-cohort .

# Run full pipeline (mount your data directory)
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results red-cohort

# Or use docker-compose
docker-compose -f docker/docker-compose.yml run pipeline
```

---

## Entrypoints

| Script | Purpose | Key flags |
|---|---|---|
| `detect.py` | Stage 1+2 detection → `sniper_cohorts.jsonl` | `--buyers`, `--from-intra`, `--ablation`, `--calibrate` |
| `analyze.py` | Tables 1-3 + Figures 1-3 | `--cohorts`, `--output-dir`, `--top-k` |
| `causal.py` | Causal analysis + placebos + robustness | `--buyers`, `--launches`, `--cohorts`, `--skip-placebo` |
| `run_all.py` | Master: detect → analyze → causal | `--buyers`, `--launches` |

---

## Expected results (paper values)

### Detection (Section 5)
| Metric | Paper value |
|---|---|
| Total cohorts | 1,012 |
| Unique cohort wallets | 2,965 |
| Premium-tier cohorts (≥20 launches) | 22 |
| High-tier cohorts | 153 |
| Mints touched (≥2-wallet strict) | 5,411 |
| Median cohort size | 2 wallets |
| Top cohort (COH-0001) score | 430.44 |
| COH-0001 launches hit | 42 |
| COH-0001 mean first-buyer rank | 2.29 |

### Causal analysis (Section 6.3)
| Outcome | Treated mean | Control mean | Lift | 95% CI |
|---|---|---|---|---|
| First-30-min buyer count | 21.00 | 9.10 | +132.3% | [+127.0%, +137.4%] |
| First-30-min SOL inflow | 5.58 | 2.35 | +136.5% | [+120.9%, +152.2%] |

### Placebo check (Appendix B.1 — the central finding)
| Design | Buyer-count lift | 95% CI |
|---|---|---|
| Real cohort (n=5,411) | +132.3% | [+127.0%, +137.4%] |
| Uniform-random placebo | +152.0% | — |
| **Activity-matched placebo (n=173)** | **+216.3%** | **[+183.8%, +255.2%]** |

Real-cohort lift lies entirely **below** the activity-matched placebo CI lower bound (183.8%). No overlap. This refutes a cohort-specific causal interpretation.

---

## Implementation assumptions

Three assumptions from the SIR require attention:

1. **Score threshold τ (SIR confidence: 0.55 — HIGH RISK)**
   The paper does not disclose the numeric value of τ. This repo defaults to `score_tau: 40.0` in `configs/config.yaml`. Use `--calibrate` in `detect.py` to binary-search for a τ that yields exactly 1,012 cohorts.

2. **Touch indicator in EQ1 (SIR confidence: 0.68 — MEDIUM RISK)**
   Whether the score formula's `1{C touches L}` uses ≥1 or ≥2 cohort wallets is ambiguous. Default is ≥1 (`touch_threshold_score: 1`). The causal analysis uses ≥2 (`touch_threshold_causal: 2`). Set `touch_threshold_score: 2` in config to test the alternative.

3. **blockTime tie-breaking (SIR confidence: 0.65 — MEDIUM RISK)**
   Buyer rank is derived from `blockTime ASC, tx_sig ASC` (lexicographic on transaction signature). The paper does not specify tie-breaking. This is a reasonable canonical convention matching typical Solana indexer behaviour.

---

## Reproducibility notes

- The activity-matched placebo produces only **n=173** treated mints vs 5,411 for real cohorts, because the strict ≥2-wallet touch threshold combined with randomly-matched wallets produces far fewer co-occurrences. This power asymmetry is inherent to the paper's design.
- The raw `pumpfun_buyers.jsonl` and `pumpfun_launches.jsonl` are not in the Zenodo release. Use the Stage-1 checkpoint (`sniper_cohorts_intra.jsonl.gz`) to bypass raw ingestion. See `data/README.md`.
- The Lorenz curve (Figure 2) and tier stratification (Section 6.5) are sensitive to the total cohort count, which depends on τ. Reproduce counts precisely by using `--calibrate`.

---

## Repository structure

```
red-cohort-2026/
├── detect.py              ← Detection entrypoint
├── analyze.py             ← Descriptive stats + figures
├── causal.py              ← Causal analysis
├── run_all.py             ← Master pipeline
├── configs/config.yaml    ← All hyperparameters (annotated with SIR confidence)
├── src/red_cohort/
│   ├── io/                ← DataLoader
│   ├── detection/         ← Stage 1+2 pipeline (EQ1, EQ3)
│   ├── causal/            ← Causal analysis (EQ2, placebo designs)
│   ├── analysis/          ← Descriptive stats + figures
│   └── utils/             ← Config, IO helpers, seed utility
├── data/                  ← Input data (obtain separately — see data/README.md)
├── results/               ← All generated outputs
├── notebooks/             ← Jupyter notebook (Stage 5)
├── docker/                ← Dockerfile + docker-compose.yml
├── requirements.txt
└── environment.yaml
```

---

## Citation

```bibtex
@article{kamat2026sniper,
  title   = {Coordinated Sniper Cohorts on Pump.fun: Detection of 1,012
             Persistent Wallet Rings and the Limits of Naive Causal Inference
             for First-Hour Buyer Flow},
  author  = {Kamat, Arati Uday},
  year    = {2026},
  doi     = {10.5281/zenodo.20978741},
  note    = {Version 1.0. Companion dataset: RED-COHORT-2026-v1 (CC-BY-4.0).
             Pending US patent application No. 64/099,108 (filed 2026-06-25).}
}
```

---

*ArXivist-generated reproducibility package. All empirical claims verified against source artefacts in the SIR (Scientific Intermediate Representation). Known deviations from paper documented in `configs/config.yaml` inline comments.*
