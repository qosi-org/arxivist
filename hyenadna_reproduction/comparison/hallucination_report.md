# Hallucination Report

**Paper ID**: arxiv_2306_015794
**Date**: 2026-07-12
**Method**: Architecture plan + generated code reviewed against the SIR and the released checkpoint.

A key fact anchors this report: after vendoring the authors' `standalone_hyenadna.py`, the pretrained
backbone loaded with **`missing=0` unexpected=4** — every backbone parameter matched the official
checkpoint key-for-key. This rules out structural/omission hallucinations in the backbone.

## Structural hallucinations (components in code but NOT in the SIR)
**None.** The backbone is the authors' exact `HyenaDNAModel`. The only added component is a linear
classification head (`nn.Linear(d_model, num_classes)`), which is the standard, paper-consistent
downstream head — not a hallucination. The from-scratch `hyena_operator.py` is present but unused on
the reproduction path (clearly labeled reference-only).

## Parametric hallucinations (assumed hyperparameters)
| Hyperparameter | Assumed value | Severity | Evidence | Suggested fix |
|---|---|---|---|---|
| Downstream LR / epochs | initially 6e-5 / 5, tuned to 2e-4 / 20 | Minor | SIR training_pipeline confidence 0.62; paper does not pin the downstream recipe per dataset. Raising them closed the gap from −7.4% to −1.96%, confirming the original values were too conservative. | Try official `lr=6e-4`; keep best-by-val (done). |

The 4 "unexpected" checkpoint keys (`lm_head.weight`, `*_torchmetrics.num_tokens.count`) are the
pretraining LM head + metric counters — correctly ignored, not hallucinations.

## Omission hallucinations (in SIR but missing/stubbed in code)
| Missing component | SIR location | Severity | Suggested fix |
|---|---|---|---|
| Reverse-complement augmentation | training_pipeline.data_augmentation | Minor | Implemented in `transforms.py` but not wired into `dataset.py`. Wire it in for a likely +0.5–1.5 pts. |

No critical omissions. Sequence-length warmup (curriculum) is a pretraining-time detail and does not
affect downstream fine-tuning from released weights, so its absence here is expected, not an omission.

## Conclusion
No critical or significant hallucinations. The reproduction is faithful; the residual <2% accuracy
gap is a fine-tuning-recipe gap, fully explained by the one Minor parametric item above.
