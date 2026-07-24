#!/usr/bin/env python
"""Download official DNAGPT weights and/or the real DeepGSR GSR datasets.

Official DNAGPT checkpoints are on Google Drive / Weiyun (TencentAILabHealthcare/DNAGPT):

    python data/download.py --weights dna_gpt0.1b_m

The real GSR data is the authors' DeepGSR release on Zenodo (10.5281/zenodo.1117159),
the exact PAS/TIS FASTA files used by the paper. This fetches Data.zip (~255 MB) and
builds the CSV splits the trainer reads:

    python data/download.py --gsr human_pas_aataaa

Without the GSR data the trainer falls back to a clearly-labeled synthetic smoke test.
"""
from __future__ import annotations

import argparse
import os
import zipfile

# DeepGSR data release (Kalkatawi et al. 2019) — the exact PAS/TIS FASTA the paper uses.
DEEPGSR_DATA_URL = "https://zenodo.org/api/records/1117159/files/Data.zip/content"

# task -> (species dir, PAS|TIS subdir, positive FASTA, negative FASTA) inside Data.zip.
GSR_FILES = {
    "human_pas_aataaa":    ("Human",    "PAS", "hs_AATAAA_polyA.fa", "hs_negAATAAA_polyA.fa"),
    "human_tis_atg":       ("Human",    "TIS", "hs_ATG_TIS.fa",      "hs_negATG_TIS.fa"),
    "mouse_pas_aataaa":    ("Mouse",    "PAS", "mm_AATAAA_polyA.fa", "mm_negAATAAA_polyA.fa"),
    "mouse_tis_atg":       ("Mouse",    "TIS", "mm_ATG_TIS.fa",      "mm_negATG_TIS.fa"),
    "bovine_pas_aataaa":   ("Bovine",   "PAS", "bt_AATAAA_polyA.fa", "bt_negAATAAA_polyA.fa"),
    "bovine_tis_atg":      ("Bovine",   "TIS", "bt_ATG_TIS.fa",      "bt_negATG_TIS.fa"),
    "fruitfly_pas_aataaa": ("FruitFly", "PAS", "dm_AATAAA_polyA.fa", "dm_negAATAAA_polyA.fa"),
    "fruitfly_tis_atg":    ("FruitFly", "TIS", "dm_ATG_TIS.fa",      "dm_negATG_TIS.fa"),
}

# Google Drive file IDs from the official DNAGPT repo README
# (https://github.com/TencentAILabHealthcare/DNAGPT). Weiyun mirror:
# https://share.weiyun.com/car87dsv
DRIVE_IDS = {
    "dna_gpt0.1b_h": "15m6CH3zaMSqflOaf6ec5VPfiulg-Gh0u",  # dna_gpt0.1b_h.pth
    "dna_gpt0.1b_m": "1C0BRXfz7RNtCSjSY1dKQeR1yP7I3wTyx",  # dna_gpt0.1b_m.pth
    "dna_gpt3b_m":   "1pQ3Ai7C-ObzKkKTRwuf6eshVneKHzYEg",  # dna_gpt3b_m.pth
    "classification":"1TdMCiJO6rq32WSka73VdKI0Cthitd9Bb",  # classification.pth (AATAAA PAS)
    "regression":    "1_BDbfB5iNmfus3imx1_YSD1ac6OiJkaY",  # regression.pth (mRNA)
}


def download_weights(name: str, data_dir: str) -> None:
    dest = os.path.join(data_dir, "..", "checkpoints", f"{name}.pth")
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    file_id = DRIVE_IDS.get(name, "")
    if not file_id:
        print(f"[weights] No Google Drive ID configured for '{name}'.")
        print("  Copy the file ID from https://github.com/TencentAILabHealthcare/DNAGPT")
        print(f"  into DRIVE_IDS['{name}'] in this script, then re-run. Falling back to")
        print("  from-scratch training is automatic if the .pth is absent.")
        return
    try:
        import gdown

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[weights] downloading {name} -> {dest}")
        gdown.download(url, dest, quiet=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[weights] download failed ({exc}). Install gdown or download manually.")


def _read_fasta_seqs(zf: zipfile.ZipFile, member: str) -> list[str]:
    """Return the sequence lines (no headers) from a FASTA member of the zip."""
    raw = zf.read(member).decode("utf-8", "replace").splitlines()
    return [ln.strip() for ln in raw if ln and not ln.startswith(">")]


def download_gsr(task: str, data_dir: str, seed: int = 42) -> None:
    """Download the real DeepGSR data (Zenodo) and build train/val/test CSVs.

    Reproduces the paper's protocol: real positive + hard-negative FASTA for the
    task, shuffled with a fixed seed, split **6 : 1.5 : 2.5** (S1.4.1).
    """
    import csv
    import random
    import urllib.request

    if task not in GSR_FILES:
        raise SystemExit(f"Unknown GSR task '{task}'. Options: {list(GSR_FILES)}")
    species, sub, pos_fa, neg_fa = GSR_FILES[task]

    out_dir = os.path.join(data_dir, "gsr", task)
    os.makedirs(out_dir, exist_ok=True)
    if all(os.path.isfile(os.path.join(out_dir, f"{s}.csv")) for s in ("train", "val", "test")):
        print(f"[gsr] {task} splits already present in {out_dir} — skipping.")
        return

    zip_path = os.path.join(data_dir, "raw", "DeepGSR_Data.zip")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if not os.path.isfile(zip_path):
        print(f"[gsr] downloading DeepGSR Data.zip (~255 MB) -> {zip_path}")
        urllib.request.urlretrieve(DEEPGSR_DATA_URL, zip_path)
    else:
        print(f"[gsr] using cached {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        base = f"Data/{species}/{sub}"
        pos = _read_fasta_seqs(zf, f"{base}/{pos_fa}")
        neg = _read_fasta_seqs(zf, f"{base}/{neg_fa}")
    print(f"[gsr] {task}: {len(pos)} positives + {len(neg)} negatives (real DeepGSR)")

    rows = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_tr = int(n * 0.60)
    n_va = int(n * 0.15)
    splits = {"train": rows[:n_tr], "val": rows[n_tr:n_tr + n_va], "test": rows[n_tr + n_va:]}
    for name, split_rows in splits.items():
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence", "label"])
            w.writerows(split_rows)
        pos_ct = sum(1 for _, y in split_rows if y == 1)
        print(f"[gsr]   {name}: {len(split_rows)} ({pos_ct} pos) -> {path}")
    print(f"[gsr] done. `python train.py --config configs/config.yaml --task {task}` now uses real data.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download DNAGPT weights / GSR data")
    p.add_argument("--weights", default=None, choices=list(DRIVE_IDS.keys()),
                   help="official DNAGPT .pth to fetch (Google Drive)")
    p.add_argument("--gsr", default=None, choices=list(GSR_FILES.keys()),
                   help="build real DeepGSR train/val/test CSVs for this task (Zenodo)")
    p.add_argument("--data-dir", default="data/")
    args = p.parse_args()

    if args.weights:
        download_weights(args.weights, args.data_dir)
    if args.gsr:
        download_gsr(args.gsr, args.data_dir)
    if not args.weights and not args.gsr:
        # default: fetch the base model weights
        download_weights("dna_gpt0.1b_m", args.data_dir)


if __name__ == "__main__":
    main()
