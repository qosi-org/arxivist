#!/usr/bin/env python
"""
evaluate.py
============
Standard ArXivist entrypoint name, delegating to scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint (see scripts/evaluate.py).")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True,
                    choices=["jane_street", "optiver", "time_imm", "dslob"])
    p.add_argument("--out", type=str, default="results/metrics_report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [sys.executable, "scripts/evaluate.py", "--config", args.config,
           "--checkpoint", args.checkpoint, "--dataset", args.dataset, "--out", args.out]
    print(f"[evaluate.py] delegating to: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
