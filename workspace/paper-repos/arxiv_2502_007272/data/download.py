#!/usr/bin/env python
"""Benchmark dataset downloader for the GENERator reproduction — via API.

  --benchmark genomic_benchmarks   Genomic Benchmarks (genomic_benchmarks API)
  --benchmark nt_tasks             Nucleotide Transformer tasks (HF datasets API)

Examples
--------
  python data/download.py --benchmark genomic_benchmarks --task human_nontata_promoters
  python data/download.py --benchmark genomic_benchmarks           # all GB tasks
"""
from __future__ import annotations

import argparse
import os
import sys

GENOMIC_BENCHMARKS = [
    "human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl",
    "human_ocr_ensembl", "human_ensembl_regulatory",
    "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
]
NT_TASKS = ["promoter_all", "enhancers"]


def download_genomic_benchmarks(task: str | None, data_dir: str) -> None:
    out_dir = os.path.join(data_dir, "genomic_benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError:
        sys.exit("genomic_benchmarks not installed. Run: pip install genomic-benchmarks")
    for name in ([task] if task else GENOMIC_BENCHMARKS):
        marker = os.path.join(out_dir, name)
        if os.path.isdir(marker):
            print(f"[skip] already present: {marker}")
            continue
        print(f"[genomic-benchmarks] downloading '{name}' via API ...")
        download_dataset(name, dest_path=out_dir)
        print(f"[ok] {name} -> {marker}")


def download_nt_tasks(task: str | None, data_dir: str) -> None:
    out_dir = os.path.join(data_dir, "nucleotide_transformer")
    os.makedirs(out_dir, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed. Run: pip install datasets")
    repo = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
    for name in ([task] if task else NT_TASKS):
        marker = os.path.join(out_dir, name)
        if os.path.isdir(marker):
            print(f"[skip] already present: {marker}")
            continue
        print(f"[nt-tasks] loading '{name}' via HF datasets API ...")
        try:
            ds = load_dataset(repo, name)
            ds.save_to_disk(marker)
            print(f"[ok] {name} -> {marker}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not fetch {name}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="GENERator benchmark downloader (API)")
    p.add_argument("--benchmark", default="genomic_benchmarks", choices=["genomic_benchmarks", "nt_tasks"])
    p.add_argument("--task", default=None, help="specific task; omit for all")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/"))
    args = p.parse_args()

    if args.benchmark == "genomic_benchmarks":
        download_genomic_benchmarks(args.task, args.data_dir)
    else:
        download_nt_tasks(args.task, args.data_dir)
    print("[done]")


if __name__ == "__main__":
    main()
