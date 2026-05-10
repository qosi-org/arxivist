"""
Download Indian microfinance village network data.
Source: Banerjee et al. (2013), "The Diffusion of Microfinance", Science.
Data hosted on Harvard Dataverse.

Usage:
    python data/download_villages.py

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 5.2, Appendix E.3.
"""
from __future__ import annotations

import os
from pathlib import Path

DATAVERSE_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/"
    "2460959"   # Approximate — verify exact file ID at Harvard Dataverse
)
DATA_DIR = Path(__file__).parent / "villages"
README_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U3BIHX"


def check_existing():
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.npy")):
        print(f"Village data already present in {DATA_DIR}/")
        return True
    return False


def download():
    print("Indian Microfinance Village Network Data")
    print("=" * 50)
    print(f"Source: Banerjee et al. (2013), Science 341(6144):1236498")
    print(f"Available at: {README_URL}")
    print()

    if check_existing():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("INSTRUCTIONS (manual download required):")
    print("1. Visit:", README_URL)
    print("2. Download the adjacency matrix files for all 43 villages.")
    print("3. Save them as: data/villages/village_{i}_adjacency.npy")
    print("   where i = 0..42 (village index).")
    print()
    print("Alternatively, if you have the original MATLAB .mat files:")
    print("  python data/convert_mat_to_npy.py --input /path/to/mat/files")
    print()
    print("Expected file format:")
    print("  np.load('data/villages/village_0_adjacency.npy')  -> shape [N, N], dtype int")
    print()

    # Create placeholder structure
    placeholder = DATA_DIR / "README_data.md"
    placeholder.write_text(
        "# Village Adjacency Data\n\n"
        "Place village adjacency matrices here as:\n"
        "  village_{i}_adjacency.npy  for i = 0..42\n\n"
        f"Download from: {README_URL}\n"
    )
    print(f"Created placeholder README at {placeholder}")


if __name__ == "__main__":
    download()
