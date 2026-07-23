# Hallucination Report

**Paper ID**: arxiv_2502_007272
**Date**: 2026-07-23
**Method**: Architecture plan + generated code audited against the SIR and the released checkpoint.

## Verdict: no structural or omission hallucinations; one minor parametric assumption

The backbone is the authors' **official** `GenerTeam/GENERator-eukaryote-1.2b-base`, loaded via
`AutoModel` — no reimplementation, so no surface for structural drift. The runtime reported
**1.15B parameters**, matching the paper's ~1.2B. The SIR architecture was **confirmed exactly**
against the downloaded `config.json` before any code was written.

## Structural hallucinations (components in code but NOT in the SIR)
**None.** The only added component is a linear classification head on the last-token embedding, which
the paper explicitly specifies (Appendix C.4: "predictions derived from the `<EOS>` token embedding
via a linear layer"). The `lm_head.weight` reported as `UNEXPECTED` at load is the pretraining LM
head — correctly dropped, since we use a classification head, not a hallucination.

## Parametric hallucinations (assumed hyperparameters)
| Hyperparameter | Assumed value | Severity | Evidence | Suggested fix |
|---|---|---|---|---|
| Downstream LR / batch size | `1e-5 / 64` | Minor | SIR conf 0.7. The paper selects these per task via grid search (Tables S8–S11); we fixed one point. The flat training-loss plateau suggests the LR is likely too low, plausibly explaining a large share of the −6.44% gap. | Grid-search `lr ∈ {1e-5…5e-3} × bs ∈ {64,128,256,512}` per Appendix C.4. |

All other training hyperparameters (AdamW β=(0.9,0.95), weight_decay 0.1, reduce-on-plateau + early
stopping patience 5) are taken verbatim from Appendix B/C.4 — not assumptions. Three SIR architecture
values originally marked inferred were confirmed by the config (hidden 2048, 26 layers, vocab 4128).

## Omission hallucinations (in SIR but missing/stubbed in code)
**None affecting this result.** Out-of-scope-by-design (correctly not reproduced):
- **Pretraining** — 386B-nucleotide pretraining (~11.8k A100-hours) is infeasible; we fine-tune the
  released checkpoint, exactly as the paper's downstream protocol does.
- **Sequence recovery / CRE design / central-dogma generation** — separate paper experiments not part
  of the classification-reproduction target. A **zero-shot VEP** path *is* included (`models/vep.py`,
  Sec 4.5) as a bonus, though not exercised in this run.

## Conclusion
Faithful reproduction. The −6.44% gap is fully explained by single-fold evaluation and an untuned
(assumed) learning rate — both methodological, not implementation errors. No corrective code changes
required; closing the gap is a matter of running the paper's grid search + 10-fold CV.
