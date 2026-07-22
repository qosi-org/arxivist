#!/usr/bin/env python
"""
Data download script.

The paper's real evaluation cohort is proprietary and cannot be downloaded
(see data/README_data.md). This script checks whether a local `data/raw/`
directory already exists; if not, it prints instructions instead of
attempting a download, and exits with a non-error status so it is safe to
call from automation.
"""

from __future__ import annotations

from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent / "raw"


def main() -> None:
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        print(f"Found existing data at {RAW_DATA_DIR}, skipping download instructions.")
        return

    print(
        "\n"
        "============================================================\n"
        " LATTICE (arXiv:2607.14410) — no public dataset to download\n"
        "============================================================\n"
        "The paper's 11-sample melanoma multimodal cohort (54,912 spots)\n"
        "is private and cannot be redistributed (Section 4.1, Appendix G.1).\n"
        "There is no public substitute at this exact resolution/modality set.\n\n"
        "This repository trains on a SYNTHETIC dataset by default -- no\n"
        "download is required. Just run:\n\n"
        "    python train.py --config configs/config.yaml\n\n"
        f"See {Path('data/README_data.md')} for full details, including the\n"
        "expected data schema if you have access to an equivalent private or\n"
        "future public cohort.\n"
        "============================================================\n"
    )


if __name__ == "__main__":
    main()
