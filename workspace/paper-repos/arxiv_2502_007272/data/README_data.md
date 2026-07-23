# Benchmark Data

Downloads go through `data/download.py`. Data lands under `data/` and is git-ignored.

## Genomic Benchmarks (default, paper Table S5)
```bash
python data/download.py --benchmark genomic_benchmarks --task human_nontata_promoters
python data/download.py --benchmark genomic_benchmarks           # all 8 tasks
```
Uses the `genomic_benchmarks` Python API. `human_nontata_promoters` is the config
default (paper: 0.958 accuracy).

## Nucleotide Transformer tasks (paper Table S3/S4, MCC)
```bash
python data/download.py --benchmark nt_tasks --task promoter_all
```
Pulls `InstaDeepAI/nucleotide_transformer_downstream_tasks` via HF `datasets`.

## Layout
```
data/
├── genomic_benchmarks/<task>/{train,test}/<class>/*.txt
└── nucleotide_transformer/<task>/   # HF save_to_disk format
```
