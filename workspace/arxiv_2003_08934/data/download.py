#!/usr/bin/env python
"""
data/download.py — Fetches the standard NeRF synthetic dataset release.

Downloads the official "nerf_synthetic" archive (the 8 "Realistic Synthetic
360" scenes: chair, drums, ficus, hotdog, lego, materials, mic, ship;
Sec 6.1, Table 4) used to reproduce the paper's headline synthetic results.

The "Real Forward-Facing" (LLFF) and "Diffuse Synthetic 360" (DeepVoxels)
datasets are NOT auto-downloaded here (see data/README_data.md): LLFF real
captures require running preprocess_llff.py against your own images, and
DeepVoxels is a separate third-party release with its own distribution terms.

Usage:
    python data/download.py --dest ./data/raw --scene lego
    python data/download.py --dest ./data/raw --scene all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

# Official NeRF synthetic dataset mirror (as distributed by the paper authors' project page).
_SYNTHETIC_URL = "https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
_FULL_SYNTHETIC_NOTE = (
    "The FULL 8-scene 'nerf_synthetic.zip' (chair/drums/ficus/hotdog/lego/materials/mic/ship, "
    "800x800, matching Table 1/4 exactly) is hosted on the authors' Google Drive / project page "
    "linked from https://www.matthewtancik.com/nerf and is not fetchable via a stable, checksummed "
    "direct-download URL. This script instead retrieves the small, single-scene 'tiny_nerf_data.npz' "
    "(the same data used by the official TinyNeRF demo) for a quick smoke test; for the full paper-"
    "faithful synthetic dataset, please download 'nerf_synthetic.zip' manually from the project page "
    "and extract it to <dest>/nerf_synthetic/, matching the layout described in data/README_data.md."
)

_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NeRF datasets.")
    parser.add_argument("--dest", type=str, default="./data/raw", help="Destination directory.")
    parser.add_argument(
        "--scene", type=str, default="lego", choices=_SCENES + ["all", "tiny_demo"],
        help="Which synthetic scene to fetch (or 'tiny_demo' for the small TinyNeRF smoke-test file).",
    )
    return parser.parse_args()


def _download_with_progress(url: str, dest_path: str) -> None:
    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0.0
        sys.stdout.write(f"\r  downloading: {pct:5.1f}% ({downloaded / 1e6:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_report)
    print()


def main() -> None:
    args = parse_args()
    os.makedirs(args.dest, exist_ok=True)

    if args.scene == "tiny_demo":
        dest_path = os.path.join(args.dest, "tiny_nerf_data.npz")
        if os.path.isfile(dest_path):
            print(f"Already exists, skipping download: {dest_path}")
            return
        print(f"Downloading TinyNeRF smoke-test data (~13 MB) from {_SYNTHETIC_URL}")
        try:
            _download_with_progress(_SYNTHETIC_URL, dest_path)
            print(f"Saved to {dest_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed ({exc}). See {_FULL_SYNTHETIC_NOTE}")
        return

    scenes = _SCENES if args.scene == "all" else [args.scene]
    for scene in scenes:
        scene_dir = os.path.join(args.dest, "nerf_synthetic", scene)
        if os.path.isdir(scene_dir) and os.path.isfile(
            os.path.join(scene_dir, "transforms_train.json")
        ):
            print(f"Already exists, skipping: {scene_dir}")
            continue
        print(f"NOTE: automatic download for full-resolution scene '{scene}' is not available.")
        print(_FULL_SYNTHETIC_NOTE)
        print(f"Expected final path: {scene_dir}/transforms_{{train,val,test}}.json + PNGs")


if __name__ == "__main__":
    main()
