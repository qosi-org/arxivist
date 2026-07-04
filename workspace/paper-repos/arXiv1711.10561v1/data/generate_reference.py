"""
generate_reference.py — Pre-generate spectral reference solutions
==================================================================
Generates and saves reference solutions for Schrödinger and Allen-Cahn
equations via pseudo-spectral methods. Run once before evaluation.

Usage:
    python data/generate_reference.py --pde schrodinger --output data/schrodinger_ref.npy
    python data/generate_reference.py --pde allen_cahn  --output data/allen_cahn_ref.npy

For Burgers, the exact solution is generated on-the-fly during evaluate.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinns.data.exact_solutions import AllenCahnSpectralSolution, SchrodingerSpectralSolution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate spectral reference solutions")
    p.add_argument("--pde",    required=True, choices=["schrodinger", "allen_cahn"])
    p.add_argument("--output", required=True, help="Output .npy file path")
    p.add_argument("--modes",  type=int, default=None,
                   help="Number of Fourier modes (default: 256 for NLS, 512 for AC)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"  Reference solution already exists at {out}. Delete to regenerate.")
        return

    print(f"  Generating reference solution for '{args.pde}'...")

    if args.pde == "schrodinger":
        solver = SchrodingerSpectralSolution()
        data   = solver._generate(nx=args.modes or 256)
        np.save(out, data)
        print(f"  Saved Schrödinger reference: shape {data.shape} → {out}")

    elif args.pde == "allen_cahn":
        solver = AllenCahnSpectralSolution()
        data   = solver._generate(nx=args.modes or 512)
        np.save(out, data)
        print(f"  Saved Allen-Cahn reference: shape {data.shape} → {out}")


if __name__ == "__main__":
    main()
