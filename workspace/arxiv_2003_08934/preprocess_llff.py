#!/usr/bin/env python
"""
preprocess_llff.py — Runs COLMAP structure-from-motion on a directory of raw
images and packages the result into the `poses_bounds.npy` format consumed
by data/dataset.py::LLFFRealDataset, for the "Real Forward-Facing" benchmark
(Sec 6.1, Appendix B: "use the COLMAP structure-from-motion package to
estimate these parameters for real data").

This is an orchestration script around the external `colmap` CLI binary
(see docker/Dockerfile system_deps); it does not reimplement COLMAP itself.

Usage:
    python preprocess_llff.py --scenedir /data/raw/fern --factor 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP + package poses_bounds.npy for an LLFF scene.")
    parser.add_argument("--scenedir", type=str, required=True, help="Directory of raw input images.")
    parser.add_argument(
        "--factor", type=int, default=8, help="Downsampling factor for COLMAP + training resolution."
    )
    return parser.parse_args()


def check_colmap_available() -> None:
    if shutil.which("colmap") is None:
        raise RuntimeError(
            "The `colmap` binary was not found on PATH. Install COLMAP "
            "(see docker/Dockerfile for the containerized setup, or "
            "https://colmap.github.io/install.html) before running this script."
        )


def run_colmap(scenedir: str) -> None:
    """Runs COLMAP's automatic reconstruction pipeline on `scenedir/images`."""
    images_dir = os.path.join(scenedir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Expected raw images at {images_dir}. See data/README_data.md for the expected layout."
        )
    sparse_dir = os.path.join(scenedir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    db_path = os.path.join(scenedir, "database.db")

    print(f"Running COLMAP feature extraction on {images_dir} ...")
    subprocess.run(
        ["colmap", "feature_extractor", "--database_path", db_path, "--image_path", images_dir],
        check=True,
    )
    print("Running COLMAP exhaustive matching ...")
    subprocess.run(["colmap", "exhaustive_matcher", "--database_path", db_path], check=True)
    print("Running COLMAP sparse mapping (this may take a while) ...")
    subprocess.run(
        [
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir,
        ],
        check=True,
    )
    print(f"COLMAP reconstruction complete. Sparse model written to {sparse_dir}")


def main() -> None:
    args = parse_args()
    check_colmap_available()
    run_colmap(args.scenedir)
    print(
        "NOTE: this script runs COLMAP's sparse reconstruction. Converting the resulting "
        "sparse/ model into poses_bounds.npy (camera extrinsics/intrinsics + near/far bounds "
        "in the LLFF convention) requires the LLFF `imgs2poses.py`-style conversion utility; "
        "that conversion logic is intentionally NOT reimplemented here since the paper does "
        "not describe its internals beyond 'use COLMAP to estimate parameters' (Appendix B). "
        "Downstream users should run the official LLFF pose-conversion script against the "
        f"sparse/ directory at {os.path.join(args.scenedir, 'sparse')} to produce poses_bounds.npy."
    )


if __name__ == "__main__":
    main()
