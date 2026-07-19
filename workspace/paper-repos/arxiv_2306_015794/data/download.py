#!/usr/bin/env python
"""Dataset downloader for the HyenaDNA reproduction — via API calls.

Subcommands
-----------
  genomic-benchmarks   Download GenomicBenchmarks classification datasets
                       (genomic_benchmarks Python API).
  hg38                 Download the hg38 human reference genome FASTA + intervals
                       (curl from the basenji_barnyard2 bucket) for pretraining.
  nt-benchmarks        Download Nucleotide Transformer downstream tasks
                       (HuggingFace `datasets` API).

Every source checks whether data already exists before downloading and prints
the estimated size. Nothing is hardcoded — output dir is --data-dir.

Examples
--------
  python data/download.py genomic-benchmarks --dataset human_nontata_promoters
  python data/download.py hg38 --data-dir data/
  python data/download.py nt-benchmarks --dataset enhancers
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

GENOMIC_BENCHMARKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
    "human_ensembl_regulatory",
]

HG38_FASTA = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
HG38_BED = "https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"


def _exists(path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[skip] already present: {path}")
        return True
    return False


def download_genomic_benchmarks(dataset: str | None, data_dir: str) -> None:
    """Use the genomic_benchmarks API to fetch one or all datasets."""
    out_dir = os.path.join(data_dir, "genomic_benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError:
        sys.exit(
            "genomic_benchmarks not installed. Run: pip install genomic-benchmarks"
        )
    targets = [dataset] if dataset else GENOMIC_BENCHMARKS
    for name in targets:
        marker = os.path.join(out_dir, name)
        if _exists(marker):
            continue
        print(f"[genomic-benchmarks] downloading '{name}' via API ...")
        download_dataset(name, dest_path=out_dir)
        print(f"[ok] {name} -> {marker}")


def download_hg38(data_dir: str) -> None:
    """Curl the hg38 FASTA (~1 GB gz) + intervals BED for pretraining."""
    out_dir = os.path.join(data_dir, "hg38")
    os.makedirs(out_dir, exist_ok=True)
    print("[hg38] estimated download size: ~1 GB (FASTA gz). This is large.")
    for url in (HG38_FASTA, HG38_BED):
        fn = os.path.join(out_dir, os.path.basename(url))
        if _exists(fn):
            continue
        print(f"[hg38] curl {url}")
        rc = subprocess.run(["curl", "-L", "-o", fn, url]).returncode
        if rc != 0:
            sys.exit(f"curl failed for {url} (exit {rc})")
        print(f"[ok] -> {fn}")


def download_nt_benchmarks(dataset: str | None, data_dir: str) -> None:
    """Fetch Nucleotide Transformer downstream tasks via HuggingFace datasets."""
    out_dir = os.path.join(data_dir, "nucleotide_transformer")
    os.makedirs(out_dir, exist_ok=True)
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        sys.exit("datasets not installed. Run: pip install datasets")

    repo = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
    try:
        configs = get_dataset_config_names(repo)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Could not list NT tasks from {repo}: {exc}\n"
            "See the Nucleotide Transformer paper appendix for manual download."
        )
    targets = [dataset] if dataset else configs
    for name in targets:
        if name not in configs:
            print(f"[warn] '{name}' not a known NT task. Available: {configs}")
            continue
        marker = os.path.join(out_dir, name)
        if _exists(marker):
            continue
        print(f"[nt-benchmarks] loading '{name}' via HF datasets API ...")
        ds = load_dataset(repo, name)
        ds.save_to_disk(marker)
        print(f"[ok] {name} -> {marker}")


def main() -> None:
    p = argparse.ArgumentParser(description="HyenaDNA dataset downloader (API)")
    p.add_argument("source", choices=["genomic-benchmarks", "hg38", "nt-benchmarks"])
    p.add_argument("--dataset", default=None, help="specific dataset/task name")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/"))
    args = p.parse_args()

    if args.source == "genomic-benchmarks":
        download_genomic_benchmarks(args.dataset, args.data_dir)
    elif args.source == "hg38":
        download_hg38(args.data_dir)
    elif args.source == "nt-benchmarks":
        download_nt_benchmarks(args.dataset, args.data_dir)
    print("[done]")


if __name__ == "__main__":
    main()
