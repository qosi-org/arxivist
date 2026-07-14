# Datasets

All downloads go through `data/download.py` (API/curl based). Data lands under
`data/` and is git-ignored.

## GenomicBenchmarks (recommended, fully public)
```bash
python data/download.py genomic-benchmarks --dataset human_nontata_promoters
# or all 8:
python data/download.py genomic-benchmarks
```
Uses the `genomic_benchmarks` Python API. The 8 tasks match the paper's Table
(GenomicBenchmarks). `human_nontata_promoters` is the config default (paper: 96.6% top-1).

## hg38 pretraining genome (large, ~1 GB)
```bash
python data/download.py hg38 --data-dir data/
```
Curls `hg38.ml.fa.gz` + `sequences_human.bed` from the `basenji_barnyard2` bucket.
Only needed for from-scratch pretraining (the reference path). Heavy for Colab free tier.

## Nucleotide Transformer benchmarks
```bash
python data/download.py nt-benchmarks --dataset enhancers
# or all tasks:
python data/download.py nt-benchmarks
```
Pulls `InstaDeepAI/nucleotide_transformer_downstream_tasks` via HuggingFace `datasets`.
Some tasks may require accepting terms on the HF hub; set `HF_TOKEN` in `.env` if prompted.

## Expected layout
```
data/
├── genomic_benchmarks/<dataset>/{train,test}/<class>/*.txt
├── hg38/{hg38.ml.fa.gz, sequences_human.bed}
└── nucleotide_transformer/<task>/   # HF save_to_disk format
```
