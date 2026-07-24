#!/usr/bin/env python3
"""
Evaluation entrypoint, reproducing Table 2 (in-distribution / cross-species /
size-extrapolation) and the calibration + deviation-histogram data behind
Fig. 7 and Fig. 8.

Example:
    python evaluate.py --checkpoint outputs/best_model.pt --regime all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402

from spr_gnn.utils.config import Config, resolve_device  # noqa: E402
from spr_gnn.data.dataset import TreePairDataModule  # noqa: E402
from spr_gnn.models.siamese_gin import SiameseGINRegressor  # noqa: E402
from spr_gnn.evaluation.metrics import RegressionMetrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SiameseGINRegressor checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data.data_dir / master_pairs_csv from config (e.g. data/toy for a smoke test)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        choices=["in_distribution", "cross_species", "size_extrapolation", "all"],
        help=(
            "Which evaluation regime to run. NOTE: cross_species and "
            "size_extrapolation require separately-partitioned master CSVs "
            "(train on 2 species / large trees excluded, etc.) that this "
            "script does not construct automatically -- see data/README_data.md."
        ),
    )
    parser.add_argument("--output", type=str, default="results/eval_results.json", help="Where to write results JSON")
    return parser.parse_args()


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    for tree_a, tree_b, target in loader:
        tree_a, tree_b = tree_a.to(device), tree_b.to(device)
        pred = model(tree_a, tree_b).cpu()
        all_preds.append(pred)
        all_targets.append(target)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    metrics = RegressionMetrics()
    result = metrics.compute(preds, targets)
    result["calibration"] = metrics.calibration_data(preds, targets)
    result["deviation_histograms"] = metrics.deviation_histograms(preds, targets)
    return result


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)
    device = resolve_device(config.get("hardware", "device", "cuda_if_available_else_cpu"))

    master_csv = str(Path(args.data_dir) / "master_pairs.csv") if args.data_dir else config.get("data", "master_pairs_csv")
    dm = TreePairDataModule(config)
    dm.setup(master_csv_path=master_csv, seed=config.get("data", "split_seed", 42))

    model = SiameseGINRegressor(
        num_species=dm.num_species,
        species_embedding_dim=config.get("model", "species_embedding_dim", 16),
        node_feature_dim_continuous=config.get("model", "node_feature_dim_continuous", 3),
        num_gin_layers=config.get("model", "num_gin_layers", 2),
        gin_hidden_dim=config.get("model", "gin_hidden_dim", 128),
        mlp_head_dims=config.get("model", "mlp_head_dims", [256, 128, 64, 1]),
        dropout=config.get("model", "dropout", 0.3),
        clamp_min=config.get("training", "clamp_min", 0.0),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = {}
    if args.regime in ("in_distribution", "all"):
        test_loader = dm.test_dataloader()
        r = run_eval(model, test_loader, device)
        # Fig 7/8 arrays are not JSON-serializable numpy -- keep the scalar
        # metrics in the report and drop raw arrays here (they're recomputed
        # on demand by notebooks/reproduce_arxiv_2607_18311.ipynb).
        results["in_distribution"] = {k: v for k, v in r.items() if k in ("mae", "rmse", "mape", "r2")}
        print(f"[in_distribution] MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} MAPE={r['mape']:.2f}% R2={r['r2']:.4f}")

    if args.regime in ("cross_species", "size_extrapolation", "all"):
        print(
            "NOTE: cross_species / size_extrapolation regimes (Table 2 rows 2-3) "
            "require the master CSV to be pre-split by species/size per Sec 5.3; "
            "this script evaluates only the standard test split unless you point "
            "--config at a config whose master_pairs_csv already encodes that split."
        )

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, float) and (obj != obj):  # NaN check without numpy dependency here
            return None
        return obj

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
