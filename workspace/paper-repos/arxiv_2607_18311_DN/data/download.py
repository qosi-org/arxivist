#!/usr/bin/env python3
"""
Downloads the paper's released dataset from Zenodo (DOI 10.5281/zenodo.20476872):
864 phylogenetic trees + 388 labelled SPR-distance pairs across four bacterial
species (Clostridium, Salmonella, Vibrio, S. pneumoniae).

The paper does not publish an MD5/SHA256 for the archive, so integrity is
checked by file presence + non-zero size only (see README_data.md).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_RECORD_URL = "https://zenodo.org/records/20476872"  # DOI 10.5281/zenodo.20476872
DATA_DIR = Path(__file__).parent
ARCHIVE_PATH = DATA_DIR / "spr_dataset.zip"


def already_downloaded() -> bool:
    return (DATA_DIR / "master_pairs.csv").exists() and (DATA_DIR / "trees").exists()


def main() -> None:
    if already_downloaded():
        print(f"Dataset already present at {DATA_DIR}. Nothing to do.")
        return

    print(
        "This paper's dataset is hosted on Zenodo and is not bundled in this "
        "repo (see architecture_plan risk_assessment: Medium severity -- "
        f"external dataset). Please:\n"
        f"  1. Visit {ZENODO_RECORD_URL} (DOI 10.5281/zenodo.20476872)\n"
        f"  2. Download the archive and place it at {ARCHIVE_PATH}\n"
        f"  3. Re-run this script to extract it.\n"
    )

    if not ARCHIVE_PATH.exists():
        print(f"Archive not found at {ARCHIVE_PATH}. Aborting -- see instructions above.")
        sys.exit(1)

    if ARCHIVE_PATH.stat().st_size == 0:
        print(f"Archive at {ARCHIVE_PATH} is empty (0 bytes). Aborting.")
        sys.exit(1)

    print(f"Extracting {ARCHIVE_PATH} ...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    if not already_downloaded():
        print("Extraction finished, but expected files (master_pairs.csv, trees/) were not found.")
        print("Please check the archive structure against README_data.md.")
        sys.exit(1)

    print(f"Dataset ready at {DATA_DIR}.")


if __name__ == "__main__":
    main()
