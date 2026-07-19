# Verification Log

**Comparison run**: 2026-07-16
**Paper ID**: arxiv_2306_015006

## Provenance
- ArXivist SIR version used: 1
- Architecture plan version used: 1
- Model: `zhihan1996/DNABERT-2-117M` (official), loaded via `AutoModel` + `trust_remote_code`
- Reported at runtime: **117.1M parameters** (paper states 117M ✅)
- Tokenizer: official DNABERT-2 BPE, vocab 4096 (paper Sec 3.1 ✅)
- Hardware (user run): Colab GPU (cuda), bf16 autocast

## Data integrity
GUE `prom_300_all` splits loaded and cross-checked against paper Table 12:

| Split | Loaded | Table 12 | Match |
|---|---|---|---|
| train | 47356 | 47356 | ✅ |
| dev | 5920 | 5920 | ✅ |
| test | 5920 | 5920 | ✅ |

## Metrics
- Paper metrics available for this dataset: 1 (MCC)
- User results provided: 2 (MCC, accuracy)
- Matched pairs: 1 (MCC)
- Unmatched (no paper target): 1 (accuracy)

| metric | paper | user | dev % | severity |
|---|---|---|---|---|
| mcc | 86.77 | 86.14 | −0.72 | excellent |

## User-reported config modifications
None — stock `configs/config.yaml` (Appendix A.3 recipe: AdamW, lr 3e-5, wd 0.01, batch 32,
warmup 50, 4 epochs, best-by-val-loss).

## Reproduction repair history (Stage 4 loops)
DNABERT-2's 2023-era remote code required five fixes to run on a 2026 stack. All are documented in
`models/classifier.py`:
1. Wrong HF repo id (`zhihanzhou` → **`zhihan1996`**).
2. GUE hub config names differ from paper task names (`promoter_detection_all` → **`prom_300_all`**);
   added a registry mapping — all 28 datasets verified to resolve.
3. `config.json` omits `pad_token_id`; transformers ≥4.50 dropped the default → inject **3** (read
   from their tokenizer).
4. Meta-device lazy init vs eager `rebuild_alibi_tensor()` → **`low_cpu_mem_usage=False`**.
5. `flash_attn_triton.py` uses `tl.dot(..., trans_b=True)`, removed in Triton 3.x → set
   `attention_probs_dropout_prob=0.1` to select the authors' **own PyTorch attention branch**.

## Integrity
- User results SHA256: `3886f45a36e026c3955510d798c3a7100f58b24ff22c08b39912487176d2bb54`
- Manual review required: No
- Review reasons: none

## Confidence of this comparison
**Medium.** One matched metric on one of 28 GUE datasets, single seed (paper averages 3). Would rise
to **High** with ≥3 matched metrics or a 3-seed average.
