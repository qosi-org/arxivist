#!/usr/bin/env python
"""
train.py
=========
Standard ArXivist entrypoint name, delegating to scripts/train_model.py
(which trains either ARTEMIS or one of the 5 baselines).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ARTEMIS or a baseline (see scripts/train_model.py).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, default="artemis",
                    choices=["artemis", "lstm", "transformer", "ns_transformer", "informer", "chronos2"])
    p.add_argument("--dataset", type=str, default="dslob",
                    choices=["jane_street", "optiver", "time_imm", "dslob"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [sys.executable, "scripts/train_model.py", "--config", args.config,
           "--model", args.model, "--dataset", args.dataset, "--seed", str(args.seed),
           "--out-dir", args.out_dir]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.debug:
        cmd.append("--debug")
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"[train.py] delegating to: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
