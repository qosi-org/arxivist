# Verification Log — arxiv_2607_18311, Stage 6 Comparison Run

- **Comparison run timestamp**: 2026-07-22T18:34:20Z
- **ArXivist SIR version used**: 1 (`sir-registry/arxiv_2607_18311/sir.json`)
- **Architecture plan version used**: 1 (`sir-registry/arxiv_2607_18311/architecture_plan.json`)
- **Paper metrics available (sir.json → evaluation_protocol.reported_results)**: 19
- **User results submitted**: 4 (mae, rmse, mape, r2 — from `evaluate.py --regime in_distribution`)
- **Matched pairs (valid deviation computed)**: 2 (mae, rmse)
- **Unmatched**: 2 (r2 — undefined on a 1-pair split; mape — no paper counterpart for this regime/split)

## Input provenance

- Source: `/tmp/spr_out/eval_results2.json`, produced by:
  ```
  python train.py --config configs/config.yaml --debug --data-dir data/toy --output-dir /tmp/spr_out
  python evaluate.py --config configs/config.yaml --checkpoint /tmp/spr_out/best_model.pt \
      --regime in_distribution --data-dir data/toy --output /tmp/spr_out/eval_results2.json
  ```
- SHA256 of `eval_results2.json`: `d7ea9ccd78dea8fe0f6d6ebcd7e0693f3c1d1691bd294b5e866a78ce8b5fbb72`
- Raw content:
  ```json
  {
    "in_distribution": {
      "mae": 1.0,
      "rmse": 1.0,
      "mape": 100.0,
      "r2": null
    }
  }
  ```

## User-reported (inferred from CLI flags) config modifications

1. `--debug`: `training.max_epochs` overridden 200 → 2; `training.batch_size` overridden 32 → 4.
2. `--data-dir data/toy`: pointed at the bundled synthetic smoke-test fixture (4 tree pairs,
   randomly-generated SPR labels via `random.uniform(0, 5)` in the reproduction notebook's
   demo cell) instead of the real Zenodo dataset (DOI 10.5281/zenodo.20476872, 388 labelled
   pairs with genuine `phangorn::SPR.dist` labels).

## Metrics compared

`mae`, `rmse`, `r2`, `mape` — all against the `in_distribution` regime row of Table 2.

## Why this run exists

No genuine training run against the paper's real dataset has been performed in this
session (Zenodo is not reachable from this sandbox's network allowlist). This Stage 6
run was executed against the only available results — the pipeline's own debug
smoke-test — specifically to (a) validate the Stage 6 reporting pipeline end-to-end, and
(b) surface pre-identified risk factors (parametric/architectural assumptions) ahead of a
real training run. It should be re-run once real results exist.

## Manual review required: **YES**

**Reasons**:
1. User results come from a debug smoke-test on synthetic random-label data, not a real
   training run on the paper's actual dataset.
2. R² is undefined on the test split used (only 1 pair, a single unique target value —
   `sklearn.r2_score` requires ≥2 unique values; `RegressionMetrics.compute()` correctly
   returns `NaN`/`null` rather than fabricating a number).
3. No genuine reproduction attempt against arXiv:2607.18311's real dataset has been
   performed yet.

## Files written this run

- `comparison/benchmark_comparison.md`
- `comparison/reproducibility_score.json`
- `comparison/hallucination_report.md`
- `comparison/verification_log.md` (this file)
- `sir-registry/arxiv_2607_18311/metadata.json` updated: `has_comparison_report: true`
