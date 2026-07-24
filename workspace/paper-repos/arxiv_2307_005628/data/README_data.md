# Data & Weights

## Official weights (Google Drive / Weiyun)
DNAGPT checkpoints are `.pth` files hosted by the authors at
[TencentAILabHealthcare/DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT):

| name | params | pretrain | max bp |
|---|---|---|---|
| `dna_gpt0.1b_h.pth` | 0.1B | human | 24,576 |
| `dna_gpt0.1b_m.pth` | 0.1B | 9 species | 24,576 |
| `dna_gpt3b_m.pth` | 3B | all mammals | 3,060 |
| `classification.pth` | — | fine-tuned (AATAAA PAS) | — |
| `regression.pth` | — | fine-tuned (mRNA) | — |

Copy each file's **Google Drive ID** from the repo README into `DRIVE_IDS` in
[`data/download.py`](download.py), then:
```bash
python data/download.py --weights dna_gpt0.1b_m   # -> checkpoints/dna_gpt0.1b_m.pth
```
`train.py` auto-loads `checkpoints/dna_gpt0.1b_m.pth` if present (via
`DNAGPT.from_pretrained`, `strict=False`), else trains from scratch.

## GSR data (real DeepGSR PAS/TIS) — one command

The GSR datasets are the authors' **DeepGSR** release on Zenodo
([10.5281/zenodo.1117159](https://doi.org/10.5281/zenodo.1117159)) — the *exact* FASTA
files the paper uses. `data/download.py` fetches `Data.zip` (~255 MB) and builds the
CSV splits automatically:

```bash
python data/download.py --gsr human_pas_aataaa    # -> data/gsr/human_pas_aataaa/{train,val,test}.csv
```

This reproduces the paper's protocol (S1.4.1): real positives + **hard negatives** (same
motif, non-signal context), 606 bp (PAS) / 603 bp (TIS) with the signal centered at
position 300, shuffled (seed 42) and split **6 : 1.5 : 2.5**. For human PAS(AATAAA) this
gives **11,302 positives + 11,302 negatives** — matching paper Table S5 exactly.

Tasks: `human_pas_aataaa`, `human_tis_atg`, `mouse_pas_aataaa`, `mouse_tis_atg`,
`bovine_pas_aataaa`, `bovine_tis_atg`, `fruitfly_pas_aataaa`, `fruitfly_tis_atg`.

CSV layout (if you prepare your own):
```
data/gsr/<task>/{train,val,test}.csv     # columns: sequence,label
```

**No data?** The loader falls back to a **clearly-labeled, un-gameable synthetic** smoke
test (weak signal + 15% label noise) so the pipeline/tests run — its accuracy is loudly
marked NOT comparable to the paper.

## Note on pretraining
Pretraining used ~200B bp across many V100s — infeasible to reproduce. This repo
loads the official released weights and fine-tunes on GSR instead.
