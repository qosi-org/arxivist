# Verification Log — arxiv_2603_18107 (ARTEMIS)

- **Comparison run at**: 2026-07-19T00:30:00Z
- **ArXivist SIR version used**: 1 (`sir-registry/arxiv_2603_18107/sir.json`, overall confidence 0.64)
- **Architecture plan version used**: 1 (`sir-registry/arxiv_2603_18107/architecture_plan.json`)
- **User experimental results supplied**: **None.** No results (pasted text, CSV, JSON, or file)
  were provided in this conversation for comparison against the paper's reported metrics.
- **User results hash**: `N/A — no input received`
- **User-reported config modifications**: N/A — no run was reported.
- **Paper metrics available for matching** (`sir.json → evaluation_protocol.reported_results`):
  19 entries across RMSE/RankIC/DirAcc/Weighted-R2 for 4 datasets plus the 7-variant DSLOB
  ablation study (Tables 2–3).
- **Metrics matched to user results**: 0 / 19 (no user results to match against).
- **Substitute reference used for this log only**: the Stage 4/5 placeholder-seed DSLOB
  smoke-test runs (`pipeline_state.json → stage4_validation`, `stage5_validation`), which
  exercised the full data-generation → fit → simulate → evaluate → plot pipeline end-to-end on
  a **locally generated placeholder seed series** (`data/make_synthetic_seed_lob.py`). This
  confirms the *code executes correctly*; it is **not** a basis for a reproducibility score,
  for two independent reasons:
  1. **Structural**: DSLOB — the dataset behind the paper's headline claim (64.96% DirAcc) and
     its entire 7-variant ablation study (Table 3) — is generated from a "real high-frequency
     limit order book dataset" that Section 3.4 never names, cites, or provides. There is no
     way to obtain the paper's actual seed data, so no exact reproduction of DSLOB numbers is
     possible in principle, not merely in practice.
  2. **Statistical**: even setting DSLOB aside, the placeholder data used in Stage 4/5 smoke
     tests has different statistical properties than any real market data, and training runs
     used toy-scale dimensions (d_z=32, 3–10 epochs) rather than the paper's (unreported)
     actual configuration.
- **Hallucination analysis**: performed independently of user results (compares
  `architecture_plan.json` and the generated `src/` code against `sir.json` and the paper
  text directly), since this does not require a user run. See `hallucination_report.md`.
- **Manual review required**: **Yes** — carried forward from Stage 1 (SIR confidence 0.64,
  below the 0.65 threshold) and reinforced by: no user-submitted results; DSLOB's structural
  unreproducibility; and a Medium-severity default-config bug found during Stage 5 (see
  `hallucination_report.md` finding H-1) that silently disables one of the paper's two
  physics-informed loss terms when using the repo's shipped defaults.

## How to complete this comparison

1. For **Jane Street / Optiver / Time-IMM** (all publicly available): download per
   `data/README_data.md`, wire up the three preprocessors already implemented in
   `src/arxivist_artemis/data/preprocessing.py`, and run `python train.py --config
   configs/config.yaml --model artemis --dataset <dataset>` for each of the 5 models × 5 seeds.
2. For **DSLOB**: exact reproduction is not possible (structural blocker, see above). At best,
   a qualitatively comparable synthetic dataset could be built by identifying and using a
   real, named high-frequency LOB dataset as the seed (e.g. LOBSTER data), then running
   `data/make_synthetic_seed_lob.py`'s underlying `DSLOBGenerator` pipeline (already validated
   as functionally correct in Stage 4/5) on it instead of the placeholder.
3. Fix the d_z/d_w config mismatch (`hallucination_report.md` finding H-1) before any
   comparison run, or L_MPR will silently be 0 throughout training.
4. Submit the resulting `results/metrics_report.md` (or raw metric values, per seed, per
   dataset/model) back to the ArXivist Results Comparator to regenerate
   `benchmark_comparison.md` and `reproducibility_score.json` with real numbers.
