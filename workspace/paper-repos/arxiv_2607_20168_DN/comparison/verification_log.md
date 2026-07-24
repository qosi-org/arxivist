# Verification Log — arxiv_2607_20168, Stage 6 Comparison Run

- **Comparison run timestamp**: 2026-07-24T05:31:52Z
- **ArXivist SIR version used**: 1
- **Architecture plan version used**: 1
- **Paper metrics available (sir.json → evaluation_protocol.reported_results)**: 30
- **User results submitted**: 12 metric-model pairs (mean_ic, icir, t_stat, hit_rate × 3 models)
- **Matched pairs**: 12 / 12

## Input provenance

Produced by:
```
python run_study.py --config configs/config.yaml --study main --debug \
    --output-dir /tmp/qkernel_out --models qkrr-fid,krr-rbf,ridge
python compare_models.py --results-dir /tmp/qkernel_out --output /tmp/qkernel_out/comparison.csv
```

SHA256 of each results file:

| File | SHA256 |
|---|---|
| `qkrr-fid_ic_series.csv` | `6288c9c17ebaebf59d3e10bd33efa192719e0d59363a3dfb01be894ae91c219c` |
| `krr-rbf_ic_series.csv` | `dbb6b11ac14780da54592b1b6a670fbce7fb3ae4791f1c37a8f982f2eea296ce` |
| `ridge_ic_series.csv` | `1dd05ba6100bcecc64e5b4343d5cf2bf08a1a17320fc909e32e04bfb6d0a1e19` |
| `comparison.csv` | `8299be51eb0d9ac8a70bec7f39d23ecd620ec1b583b7d4a6aa3c2476eda056fe` |
| `comparison_pairwise.csv` | `4d2ce4dfdb76da65591c10edf4d54e4a11c9a576db1f0ef1a2305df077b41396` |
| `bandwidth_geometry_diagnostic.csv` | `b0dd553365a77c46c218064b782dfd9c2990c90ec8e3cb600920bf83298fe3f9` |

## User-reported (inferred from CLI flags) config modifications

`--debug`: synthetic 20-60-ticker panel (`data/synthetic.py`) with an artificially injected
return signal, `N=32` training subsample (vs. real `N=1536`), 3 walk-forward windows (vs.
real 170), fixed λ/α/γ (grid index 1) rather than per-window grid search.

## Bug found and fixed during this session's testing

While building the repo, `python run_study.py --debug` initially failed with
`KeyError: 'date'` inside `WalkForwardEngine.iter_windows`. Root cause: a pandas-version-
dependent behavior where `groupby(..., group_keys=False).apply(fn)` silently drops the
grouping column from its output on pandas ≥3.0, even when `fn` returns that column. Fixed
by rewriting `CharacteristicBuilder.standardize()` (and `data/synthetic.py`'s use of it) to
use `groupby().transform()` instead of `.apply()`, which cannot have this failure mode.
Re-verified with an isolated pandas repro before and after the fix; then re-ran the full
`run_study.py` → `compare_models.py` → `run_bandwidth_geometry_diagnostic.py` chain plus
the full notebook via `jupyter nbconvert --execute`, all successful.

## Why this run exists

No real training run against the paper's actual (proprietary, unreleased) dataset was
possible in this session. This Stage 6 run was executed against the only available
results — the pipeline's own debug smoke-test — specifically to (a) validate the Stage 6
reporting pipeline end-to-end, (b) confirm every module in the architecture plan is
actually wired together correctly (not just individually unit-testable), and (c) surface
pre-identified risk factors ahead of any real-data attempt.

## Manual review required: **YES**

**Reasons**:
1. User results come from a debug smoke-test on synthetic data with an artificially
   injected signal, not real China A-share data.
2. Only 3 windows evaluated vs. the paper's 170 (main study) — statistically meaningless.
3. No genuine reproduction attempt against arXiv:2607.20168's real dataset has been
   performed, and cannot be, without the user sourcing equivalent data independently.

## Files written this run

- `comparison/benchmark_comparison.md`
- `comparison/reproducibility_score.json`
- `comparison/hallucination_report.md`
- `comparison/verification_log.md` (this file)
- `sir-registry/arxiv_2607_20168/metadata.json` updated: `has_comparison_report: true`
