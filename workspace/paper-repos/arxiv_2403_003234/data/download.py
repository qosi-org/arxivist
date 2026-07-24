#!/usr/bin/env python
"""Download Genomic Benchmarks datasets used by Caduceus (Table 1).

Uses the genomic_benchmarks API. Example:
    python data/download.py --benchmark genomic_benchmarks --task human_nontata_promoters
    python data/download.py --benchmark genomic_benchmarks            # all tasks
"""
from __future__ import annotations

import argparse
import os

GENOMIC_BENCHMARKS_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "human_ensembl_regulatory",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
]


def download_genomic_benchmarks(task: str | None, data_dir: str) -> None:
    from genomic_benchmarks.loc2seq import download_dataset

    dest = os.path.join(data_dir, "genomic_benchmarks")
    os.makedirs(dest, exist_ok=True)
    tasks = [task] if task else GENOMIC_BENCHMARKS_TASKS
    for t in tasks:
        print(f"[download] genomic_benchmarks/{t} -> {dest}")
        download_dataset(t, dest_path=dest)
    print("[done] Genomic Benchmarks downloaded.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download Caduceus benchmark datasets")
    p.add_argument("--benchmark", default="genomic_benchmarks", choices=["genomic_benchmarks"])
    p.add_argument("--task", default=None, help="specific task; omit for all")
    p.add_argument("--data-dir", default="data/")
    args = p.parse_args()

    if args.benchmark == "genomic_benchmarks":
        download_genomic_benchmarks(args.task, args.data_dir)


if __name__ == "__main__":
    main()
