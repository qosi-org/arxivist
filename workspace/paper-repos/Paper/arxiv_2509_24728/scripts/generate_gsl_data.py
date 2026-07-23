"""
Pre-generate and cache synthetic GSL datasets for all 5 theta* settings.

Usage:
    python scripts/generate_gsl_data.py
    python scripts/generate_gsl_data.py --output_dir data/gsl/ --n_samples 10000 --seed 42

This saves pickled (X, Y) tensors that get_gsl_dataloaders() will find and load
instead of re-generating, saving time when running multiple experiments.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catnat.data.gsl_dataset import GSLDataGenerator
from src.catnat.utils.config import GSLDataConfig, GSLModelConfig

import pickle


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic GSL datasets")
    p.add_argument("--output_dir", type=str, default="data/gsl/",
                   help="Directory to save generated datasets")
    p.add_argument("--n_samples",  type=int, default=10000,
                   help="Number of (x,y) samples per dataset")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed")
    p.add_argument("--theta_star_values", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 0.9],
                   help="List of theta* values to generate datasets for")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = GSLDataConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        data_dir=args.output_dir,
    )
    model_config = GSLModelConfig()   # Uses paper defaults

    generator = GSLDataGenerator(data_config, model_config)

    print(f"Generating {len(args.theta_star_values)} datasets × {args.n_samples} samples each")
    print(f"Output directory: {out_dir.resolve()}")
    print()

    for theta_star in args.theta_star_values:
        cache_path = out_dir / f"gsl_theta{theta_star}_seed{args.seed}.pkl"

        if cache_path.exists():
            print(f"  [SKIP] theta*={theta_star} — already exists at {cache_path}")
            continue

        print(f"  Generating theta*={theta_star}... ", end="", flush=True)
        X, Y = generator.generate(
            n_samples=args.n_samples,
            theta_star=theta_star,
            seed=args.seed,
        )
        with open(cache_path, "wb") as f:
            pickle.dump((X, Y), f)
        print(f"Done. X:{X.shape}, Y:{Y.shape} → {cache_path}")

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
