# Benchmark Data

Downloads go through `data/download.py`. Data lands under `data/` and is git-ignored.

## Genomic Benchmarks (default, paper Table 1)
```bash
python data/download.py --benchmark genomic_benchmarks --task human_nontata_promoters
python data/download.py --benchmark genomic_benchmarks           # all 8 tasks
```
Uses the `genomic_benchmarks` Python API. `human_nontata_promoters` is the config
default (paper Caduceus-Ph: 0.946 accuracy). `demo_human_or_worm` is the highest
(0.973) and a good sanity target.

## Layout
```
data/
└── genomic_benchmarks/<task>/{train,test}/<class>/*.txt
```

## Note on pretraining data
Caduceus is pretrained on the **HG38 human reference genome** (Enformer splits,
~35B tokens). Reproducing pretraining is infeasible here; this repo fine-tunes the
**official released weights** (`kuleshov-group/caduceus-*`) instead.
