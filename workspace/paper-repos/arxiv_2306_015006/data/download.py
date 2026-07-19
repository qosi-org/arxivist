#!/usr/bin/env python
"""GUE benchmark downloader — via API.

Downloads the Genome Understanding Evaluation (GUE) datasets introduced by
DNABERT-2 (Sec 4.2). Pulls from the HuggingFace `datasets` hub and writes local
CSVs in the layout expected by src/dnabert2/data/gue.py:

    {data_dir}/GUE/{task}/{subset}/{train,dev,test}.csv

Examples
--------
  python data/download.py --task promoter_detection
  python data/download.py                      # all tasks
"""
from __future__ import annotations

import argparse
import os
import sys

# Task -> (hf_prefix, subsets). Mirrors utils/config.GUE_TASKS; kept local so the
# script is standalone. hf_prefix maps our task names to the GUE dataset's
# BuilderConfig names on the HF hub (e.g. promoter_detection/all -> prom_300_all).
GUE = {
    "promoter_detection":      ("prom_300_",  ["all", "notata", "tata"]),
    "core_promoter_detection": ("prom_core_", ["all", "notata", "tata"]),
    "tf_human":                ("human_tf_",  ["0", "1", "2", "3", "4"]),
    "tf_mouse":                ("mouse_",     ["0", "1", "2", "3", "4"]),
    "epigenetic_marks":        ("emp_",       ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2",
                                               "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac"]),
    "splice":                  ("splice_",    ["reconstructed"]),
    "covid_variant":           ("virus_",     ["covid"]),
}

HF_REPO = "leannmlindsey/GUE"  # community mirror of the GUE benchmark on the HF hub


def _save_split(ds, out_csv: str) -> None:
    import pandas as pd

    seq_col = "sequence" if "sequence" in ds.column_names else ds.column_names[0]
    df = pd.DataFrame({"sequence": list(ds[seq_col]), "label": list(ds["label"])})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[ok] {out_csv} ({len(df)} rows)")


def download_task(task: str, data_dir: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed. Run: pip install datasets")

    prefix, subsets = GUE[task]
    for subset in subsets:
        cfg = f"{prefix}{subset}"
        out_dir = os.path.join(data_dir, "GUE", task, subset)
        if os.path.exists(os.path.join(out_dir, "test.csv")):
            print(f"[skip] already present: {out_dir}")
            continue
        print(f"[GUE] downloading {task}/{subset} (hub config '{cfg}') via HF datasets API ...")
        try:
            for split, hf_split in (("train", "train"), ("dev", "validation"), ("test", "test")):
                try:
                    ds = load_dataset(HF_REPO, name=cfg, split=hf_split)
                except ValueError:
                    # Some configs name the validation split 'dev'.
                    alt = {"validation": "dev"}.get(hf_split, hf_split)
                    ds = load_dataset(HF_REPO, name=cfg, split=alt)
                _save_split(ds, os.path.join(out_dir, f"{split}.csv"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not fetch {cfg}: {exc}")
            print("       See the DNABERT-2 repo (MAGICS-LAB/DNABERT_2) for the official GUE download.")


def main() -> None:
    p = argparse.ArgumentParser(description="GUE dataset downloader (API)")
    p.add_argument("--task", default=None, choices=list(GUE), help="specific task; omit for all")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/"))
    args = p.parse_args()

    tasks = [args.task] if args.task else list(GUE)
    for t in tasks:
        download_task(t, args.data_dir)
    print("[done]")


if __name__ == "__main__":
    main()
