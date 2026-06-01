#!/usr/bin/env python3
"""
run_all.py — Full paper reproduction: QVC + QKE.

Sequentially runs QVC (all depths, 3 datasets) and QKE (Sets I-III),
then saves a combined results JSON and generates all paper figures.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --debug        # fast mode (~2-3 min)
    python scripts/run_all.py --seed 0
    python scripts/run_all.py --config configs/default.yaml

Paper: Havlicek et al. (2018), arXiv:1804.11326v2
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsvm.config import load_config, set_seed
from qsvm.utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Full Havlicek 2018 reproduction")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Fast mode — quick validation, not full reproduction")
    p.add_argument("--skip-qvc", action="store_true")
    p.add_argument("--skip-qke", action="store_true")
    return p.parse_args()


def run_script(script: str, extra_args: list) -> int:
    """Run a script via subprocess; stream output live."""
    cmd = [sys.executable, script] + extra_args
    print(f"\n>>> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.seed = args.seed
    out_dir = args.output_dir or cfg.output.results_dir
    set_seed(cfg.seed)
    ensure_dir(out_dir)

    scripts_dir = Path(__file__).parent
    extra = ["--config", args.config, "--plot"]
    if args.debug:
        extra.append("--debug")
    if args.seed is not None:
        extra += ["--seed", str(args.seed)]

    t0 = time.time()

    # --- QVC ---
    if not args.skip_qvc:
        print("\n" + "=" * 55)
        print("  STAGE: Quantum Variational Classifier")
        print("=" * 55)
        rc = run_script(str(scripts_dir / "train_qvc.py"), extra)
        if rc != 0:
            print(f"[WARNING] QVC script exited with code {rc}")

    # --- QKE ---
    if not args.skip_qke:
        print("\n" + "=" * 55)
        print("  STAGE: Quantum Kernel Estimator")
        print("=" * 55)
        rc = run_script(str(scripts_dir / "train_qke.py"), extra)
        if rc != 0:
            print(f"[WARNING] QKE script exited with code {rc}")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  All stages complete. Total: {elapsed:.1f}s")
    print(f"  Results in: {out_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
