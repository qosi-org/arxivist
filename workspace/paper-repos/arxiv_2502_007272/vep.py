#!/usr/bin/env python
"""Zero-shot variant effect prediction with GENERator (paper Sec 4.5).

Prints the VEP log-likelihood ratio for a single SNV. Positive => reference
allele favored (stronger evolutionary constraint).
"""
from __future__ import annotations

import argparse

from src.generator.models.vep import VEPScorer
from src.generator.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GENERator zero-shot VEP")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--sequence", required=True, help="reference DNA context (A/C/G/T)")
    p.add_argument("--pos", type=int, required=True, help="0-based variant position")
    p.add_argument("--ref", required=True, help="reference allele (single base)")
    p.add_argument("--alt", required=True, help="alternative allele (single base)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.hardware.get("device", "auto"))

    scorer = VEPScorer(cfg.model["model_name"], device=str(device))
    score = scorer.score_variant(args.sequence, args.pos, args.ref, args.alt)
    verdict = "reference-favored (likely constrained)" if score > 0 else "alt-tolerated"
    print(f"[vep] {args.ref}{args.pos}{args.alt} | VEP score = {score:.4f} | {verdict}")


if __name__ == "__main__":
    main()
