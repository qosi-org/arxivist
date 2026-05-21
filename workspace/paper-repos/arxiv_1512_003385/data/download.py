"""Download CIFAR-10 to the given directory using torchvision (idempotent)."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 via torchvision.")
    parser.add_argument("--dest", type=str, default="./data/cifar10",
                        help="Destination directory.")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 to {dest} (skipped if already present)...")
    CIFAR10(root=str(dest), train=True, download=True)
    CIFAR10(root=str(dest), train=False, download=True)
    print("Done.")


if __name__ == "__main__":
    main()
