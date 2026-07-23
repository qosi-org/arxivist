# Architecture Plan — RED-COHORT-2026
## Paper: Coordinated Sniper Cohorts on Pump.fun
### ArXivist Stage 3 Output | v1 | 2026-07-14

---

## 1. Framework Selection

**Python 3.10+ | pandas + networkx + scipy + numpy | No GPU required**

This paper is a graph-theoretic detection pipeline, not a neural network. The technology choices follow directly from the two released scripts (`analyze_sniper_cohorts.py`, `gen_p7_artifacts.py`) and the nature of the operations: tabular JSONL ingestion (pandas), co-occurrence graph operations (networkx), and bootstrap CI math (numpy/scipy). `orjson` replaces stdlib `json` for 10-30× faster parsing of 1.5M-record JSONL files.

---

## 2. Repository Layout

```
red-cohort-2026/
│
├── detect.py                  ← Entrypoint: Stage 1+2 detection pipeline
├── analyze.py                 ← Entrypoint: Descriptive stats + figures
├── causal.py                  ← Entrypoint: Causal analysis + placebo checks
├── run_all.py                 ← Master entrypoint: runs all three in sequence
│
├── configs/
│   └── config.yaml            ← All hyperparameters (annotated with SIR confidence)
│
├── src/
│   └── red_cohort/
│       ├── __init__.py
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   └── loader.py      ← DataLoader: reads buyers.jsonl + launches.jsonl
│       │
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── extractor.py   ← Stage 1: IntraLaunchExtractor
│       │   ├── graph.py       ← CoOccurrenceGraph builder + EdgeWeightFilter
│       │   ├── union_find.py  ← CohortSurface (union-find) + CohortSizeFilter
│       │   ├── scorer.py      ← CohortScorer (EQ1) + TierClassifier
│       │   └── pipeline.py    ← DetectionPipeline (orchestrates full Stage 1→2)
│       │
│       ├── causal/
│       │   ├── __init__.py
│       │   ├── sample.py      ← CausalSampleBuilder (treated/control construction)
│       │   ├── estimator.py   ← LiftEstimator (EQ2 + bootstrap CI)
│       │   ├── placebo.py     ← UniformRandomPlacebo + ActivityMatchedPlacebo
│       │   └── robustness.py  ← RobustnessChecker (top-k exclusion, tier strat)
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── descriptive.py ← DescriptiveAnalyzer (Tables 1-3, Lorenz data)
│       │   └── visualizer.py  ← Visualizer (Figures 1-3)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py      ← DetectionConfig, CausalConfig, PipelineConfig dataclasses
│           └── io_helpers.py  ← JsonlStreamer, AddressAnonymizer
│
├── data/
│   ├── README.md              ← How to collect/obtain pumpfun_buyers.jsonl
│   └── sample_buyers.jsonl    ← 10,000-row anonymized sample for smoke tests
│
├── results/                   ← All output artifacts written here
│
├── tests/
│   ├── unit/
│   │   ├── test_extractor.py
│   │   ├── test_graph.py
│   │   ├── test_union_find.py
│   │   ├── test_scorer.py
│   │   ├── test_tier.py
│   │   ├── test_estimator.py
│   │   └── test_placebo.py
│   └── integration/
│       ├── test_pipeline_smoke.py
│       └── test_causal_smoke.py
│
├── notebooks/
│   └── red_cohort_walkthrough.ipynb   ← Stage 5 output (not yet generated)
│
├── docker/
│   └── Dockerfile
│
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
└── README.md
```

---

## 3. Data Flow (Pseudocode)

### Detection Pipeline

```
INGEST
  stream pumpfun_buyers.jsonl (1,578,333 records)
  → buyers_df: DataFrame[N×7] cols: {mint, wallet, slot, blockTime, sol_in, tx_sig, rank}
  stream pumpfun_launches.jsonl
  → launches_df: DataFrame[M×9]

STAGE1_EXTRACT  [IntraLaunchExtractor]
  filter buyers_df → rank <= window_size (default: 10)
  group by mint → per-launch first-buyer index
  → intra_index: DataFrame[~1.66M×5] cols: {mint, wallet, rank, block_time, sol_committed}
  qualifying_mints = 166,098

GRAPH_BUILD  [CoOccurrenceGraph]
  for each mint: generate all (u,v) wallet pairs from intra_index
  accumulate edge weights in dict: weight[(u,v)] += 1
  → G_raw: nx.Graph, |V|=all wallets, |E|=all co-occurrence pairs

EDGE_FILTER  [EdgeWeightFilter]
  retain edges where weight >= min_weight (default: 3)
  → G_filtered: 9,788 qualifying pairs

UNION_FIND  [CohortSurface]
  nx.connected_components(G_filtered)
  → components_raw: 1,161 frozensets at cutoff=3

SIZE_FILTER  [CohortSizeFilter]
  discard components where len > max_cohort_size (default: 12)
  → components_filtered: 1,012 frozensets

SCORE  [CohortScorer — EQ1]
  for each component C:
    n_launches = count of mints where any wallet in C in intra_index
    mean_first_rank = mean(min rank per touched launch)
    total_sol = sum(sol_committed for C wallets across touched launches)
    score = 10*n_launches + 5/mean_first_rank + sqrt(total_sol)
  apply threshold tau (default: 40.0)  ← ⚠ ASSUMED VALUE
  → scored_cohorts

TIER  [TierClassifier]
  premium: n_launches >= 20
  high:    n_launches >= 10  OR  score >= 100
  standard: otherwise
  → cohorts_df: 1,012 rows, sorted by score DESC, id = COH-0001..COH-1012
```

### Causal Analysis Pipeline

```
SAMPLE  [CausalSampleBuilder]
  treated_mints = mints with >=2 cohort wallets in first-10 (strict threshold)
  → n_treated = 5,411
  control_mints = random sample(untouched, size=3×5411, seed=42)
  → n_control = 16,233
  attach outcomes: first_30min_buyer_count, first_30min_sol_inflow, total_buyer_count

LIFT  [LiftEstimator — EQ2]
  lift = (mean_treated - mean_control) / mean_control * 100
  bootstrap 1,000× with replacement → percentile CI [2.5, 97.5]
  → {point_estimate, ci_lower, ci_upper, treated_mean, control_mean}

PLACEBO_DESIGN1  [UniformRandomPlacebo]
  sample 1,012 placebo cohorts uniformly from buyer-event universe
  touch threshold: >=1 wallet
  → placebo_buyer_lift = +152.0%

PLACEBO_DESIGN2  [ActivityMatchedPlacebo]
  for each real cohort C of size k with per-wallet launch counts {a1..ak}:
    sample k non-cohort wallets each matched to ai (within ±100 launches)
  touch threshold: >=2 wallets (same as real cohort)
  → n_placebo_treated = 173, buyer_lift = +216.3% [+183.8%, +255.2%]

ROBUSTNESS  [RobustnessChecker]
  top_k_exclusion: remove COH-0001/0002/0003 → n_treated=5,869, lift=+128.8%
  tier_stratification: standard=+122.8%, high=+131.4%, premium=+79.5%
```

---

## 4. Configuration Schema (`configs/config.yaml`)

```yaml
detection:
  first_buyer_window: 10         # SIR confidence: 0.97
  edge_weight_cutoff: 3          # SIR confidence: 0.97
  max_cohort_size: 12            # SIR confidence: 0.97
  score_tau: 40.0                # ASSUMED: exact tau undisclosed (SIR confidence: 0.55)
                                 # Use --calibrate to binary-search for target cohort count
  touch_threshold_score: 1       # ASSUMED: >=1 for score (SIR confidence: 0.68)
                                 # Set to 2 to match causal analysis threshold
  touch_threshold_causal: 2      # SIR confidence: 0.97
  ablation_cutoffs: [2, 3, 5]    # SIR confidence: 0.97

causal:
  window_minutes: 30             # SIR confidence: 0.97
  control_ratio: 3               # SIR confidence: 0.97
  random_seed: 42                # SIR confidence: 0.97
  bootstrap_iterations: 1000     # SIR confidence: 0.97
  bootstrap_ci_level: 0.95       # SIR confidence: 0.97
  activity_match_tolerance: 100  # SIR confidence: 0.97
  top_k_exclusion: 3             # SIR confidence: 0.97

tier_thresholds:
  premium_min_launches: 20       # SIR confidence: 0.97
  high_min_launches: 10          # SIR confidence: 0.97
  high_min_score: 100.0          # SIR confidence: 0.97

data:
  buyers_path: data/pumpfun_buyers.jsonl
  launches_path: data/pumpfun_launches.jsonl
  output_dir: results/

hardware:
  n_workers: 4
  chunk_size: 100000
```

---

## 5. Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.2.2 | DataFrames for buyer events, launches, cohort records |
| numpy | 1.26.4 | Vectorized score math, bootstrap percentile CI |
| networkx | 3.3 | Co-occurrence graph + connected_components (union-find) |
| scipy | 1.13.1 | Statistical utilities; placeholder for future PSM extension |
| matplotlib | 3.9.0 | Figures 1-3 reproduction |
| pyyaml | 6.0.1 | Config file loading |
| tqdm | 4.66.4 | Progress bars for 1.5M-record streaming |
| orjson | 3.10.3 | Fast JSONL parsing (10-30× faster than stdlib json) |

Dev: pytest, black, ruff, mypy, jupyter

---

## 6. Entrypoints

| Script | Purpose | Key Args |
|---|---|---|
| `detect.py` | Full detection → cohort catalogue | `--buyers`, `--config`, `--output`, `--ablation`, `--cutoff` |
| `analyze.py` | Descriptive stats + all 3 figures | `--cohorts`, `--output-dir`, `--top-k` |
| `causal.py` | Causal analysis + placebos + robustness | `--buyers`, `--launches`, `--cohorts`, `--skip-placebo`, `--seed` |
| `run_all.py` | Master: detect → analyze → causal | `--buyers`, `--launches`, `--output-dir` |

---

## 7. Docker Spec

- **Base image**: `python:3.10-slim` (no GPU required)
- **System deps**: `build-essential`, `git`
- **Working dir**: `/app`
- **Volume mounts**: `/app/data` (input JSONL), `/app/results` (outputs)
- **Default CMD**: `python run_all.py --buyers data/pumpfun_buyers.jsonl --launches data/pumpfun_launches.jsonl`
- **Env**: `PYTHONUNBUFFERED=1`, `PYTHONPATH=/app/src`

---

## 8. Risk Register

| ID | Severity | Component | Issue | Mitigation |
|---|---|---|---|---|
| RISK-01 | 🔴 HIGH | `scorer.py` — tau | Score threshold tau not disclosed in paper. Default 40.0 is assumed. Exact cohort count (1,012) may not reproduce. | Expose `--calibrate` flag for binary-search. Abstract `apply_score_threshold()` as swappable method. |
| RISK-02 | 🟡 MEDIUM | `scorer.py` — touch indicator | Score formula touch definition (>=1 vs >=2 wallets) ambiguous (confidence 0.68). Affects n_launches_hit and tier labels. | Expose `touch_threshold_score` in config. Default=1. Run both variants in --ablation mode. |
| RISK-03 | 🟡 MEDIUM | `placebo.py` — small n=173 | Activity-matched placebo yields only 173 treated mints. CI comparison is power-asymmetric by design. | Implement faithfully. Expose `--placebo-touch-threshold` for >=1 variant. Document asymmetry in output summary. |
| RISK-04 | 🟡 MEDIUM | `loader.py` — blockTime ties | Tie-breaking strategy for equal blockTime not specified. Affects rank assignment near window boundary. | Default: sort (blockTime ASC, tx_sig ASC). Expose as config option. Tag TODO in code. |
| RISK-05 | 🟢 LOW | `graph.py` — memory | Raw pair generation ~7.5M before filtering. Fits in RAM but should stream. | Use itertools.combinations + defaultdict accumulation. Build networkx graph only from filtered edges. |
| RISK-06 | 🟢 LOW | Data access | Raw JSONL not in Zenodo release. Users need Solana RPC access or author contact. | Provide data/README.md with collection instructions. Provide 10K-row sample file. Support `--from-intra` to skip Stage 1. |

---

## Reproduction Target Benchmarks

When run against the released `sniper_cohorts_intra.jsonl.gz`:

| Metric | Paper Value | Tolerance |
|---|---|---|
| Total cohorts | 1,012 | ±50 (tau ambiguity) |
| Median cohort size | 2 | exact |
| Max launches hit (COH-0001) | 42 | exact |
| COH-0001 score | 430.44 | ±0.1 |
| Naive buyer-count lift | +132.3% | ±5pp |
| Activity-matched placebo lift | +216.3% | ±20pp (wide CI) |
