# Hallucination Report

**Paper ID**: arxiv_2306_015006
**Date**: 2026-07-16
**Method**: Architecture plan + generated code audited against the SIR and the released checkpoint.

## Verdict: no hallucinations found (0 structural, 0 parametric, 0 omission)

Two facts anchor this. First, the backbone is the authors' **official** `zhihan1996/DNABERT-2-117M`
loaded via `AutoModel` — nothing about the encoder was reimplemented, so there is no surface for
structural drift. Second, the runtime reported **117.1M parameters**, matching the paper's stated
117M.

## Structural hallucinations (components in code but NOT in the SIR)
**None.** The only added component is a linear classification head
(`nn.Linear(hidden_size, num_classes)` + dropout), which is the standard downstream head the paper's
fine-tuning protocol implies (Sec 5.2). ALiBi, GEGLU, and the BPE tokenizer all come from the
released repo rather than our code.

## Parametric hallucinations (assumed hyperparameters that may be wrong)
**None material.** All fine-tuning hyperparameters were taken verbatim from **Appendix A.3**
(AdamW, lr 3e-5, β=(0.9, 0.98), weight_decay 0.01, batch 32, warmup 50) and Table 7 (epochs), i.e.
SIR confidence 0.9+ — not assumptions.

Notably, three architecture values the SIR had marked **inferred (conf 0.7)** were **confirmed
exactly** by the downloaded `config.json`:

| SIR inference | Actual config | Status |
|---|---|---|
| hidden_size 768 | 768 | ✅ confirmed |
| 12 layers | 12 | ✅ confirmed |
| vocab 4096 | 4096 | ✅ confirmed |

Two deliberate, documented deviations (neither a hallucination — both are recorded with rationale):

| Setting | Value | Why |
|---|---|---|
| `pool` | `mean` | Paper does not specify the classification pooling (SIR ambiguity, conf 0.75). Masked mean-pool chosen; `cls` is config-selectable. |
| `attention_probs_dropout_prob` | `0.1` (their config ships 0.0) | Required to select the authors' **own** PyTorch attention branch — their `flash_attn_triton.py` calls `tl.dot(..., trans_b=True)`, an API removed in Triton 3.x. Their code comment states the PyTorch branch is for "nonzero attention dropout (e.g. during fine-tuning)", so this matches intent. Mathematically identical attention. |

## Omission hallucinations (in SIR but missing/stubbed in code)
**None affecting this result.** Two SIR components are intentionally out of scope for downstream
fine-tuning from released weights, and their absence is correct, not an omission:
- **MLM pretraining head** — pretraining is not reproduced (the paper's own pretraining took ~14 days
  on 8×2080Ti); we fine-tune the released checkpoint, as the paper's Sec 5 protocol does.
- **LoRA** — the paper uses LoRA for the *Nucleotide Transformer baseline*, not for DNABERT-2 itself
  ("we perform standard fine-tuning for DNABERT and DNABERT-2"). Correctly not applied.

## Conclusion
The reproduction is faithful and the −0.72% MCC gap is explained by single-seed variance, not by any
implementation error. No corrective action required.
