"""
data/dataset.py — Freddie Mac Loan Dataset Loader and Graph Constructor.

Loads the Single-Family Loan-Level (SFLL) dataset from Freddie Mac and
constructs the dynamic multilayer network snapshots described in Section 4.

Network types (Section 4.2):
  - Single-layer (area): borrowers connected if zip code shares first 2 digits
  - Single-layer (company): borrowers connected if same mortgage provider
  - Double-layer (area+company): supra adjacency matrix combining both

Uses rolling windows of 6 monthly snapshots (τ=6) with 1-month shift.
Preprocessing: outlier capping (1st/99th pctile), median imputation,
               min-max scaling, binary encoding of categoricals (Section 4.1).

Paper: Zandi et al. (EJOR 2025). DOI: 10.1016/j.ejor.2024.09.025
Data source: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor


# 16 node features used (Table 1)
STATIC_FEATURES = [
    "fico", "if_fthb", "mi_pct", "cnt_units", "if_prim_res",
    "dti", "ltv", "if_corr", "if_sf", "if_purc", "cnt_borr", "if_sc",
]
DYNAMIC_FEATURES = ["current_upb", "if_delq_sts", "mths_remng", "current_int_rt"]
ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES
TARGET_FEATURE = "default"

WINDOW_SIZE = 6        # τ = 6 snapshots per window (Section 4.2)
WINDOW_SHIFT = 1       # rolling windows shift by 1 month


class FreddieWindow:
    """One rolling window of 6 monthly LOB snapshots plus labels.

    Attributes:
        snapshot_feats: List of τ feature tensors, each [nl, 16].
        snapshot_adjs:  List of τ adjacency tensors, each [nl, nl] or [2nl, 2nl].
        labels:         Node labels [nl] (1=default, 0=no default).
        node_ids:       Array of loan identifiers [nl].
    """

    def __init__(
        self,
        snapshot_feats: list[Tensor],
        snapshot_adjs: list[Tensor],
        labels: Tensor,
        node_ids: np.ndarray,
    ) -> None:
        self.snapshot_feats = snapshot_feats
        self.snapshot_adjs = snapshot_adjs
        self.labels = labels
        self.node_ids = node_ids

    def __len__(self) -> int:
        return len(self.node_ids)

    def __repr__(self) -> str:
        return (
            f"FreddieWindow(nodes={len(self.node_ids)}, "
            f"snapshots={len(self.snapshot_feats)}, "
            f"default_rate={self.labels.float().mean():.3f})"
        )


class FreddieDataset:
    """Freddie Mac SFLL dataset with dynamic multilayer network construction.

    Args:
        data_dir: Path to directory with processed CSV/parquet files.
        network_type: One of 'area', 'company', 'double'.
        split: 'train' or 'test'.
        cfg: Full config dictionary.
    """

    def __init__(
        self,
        data_dir: str | Path,
        network_type: str = "double",
        split: str = "train",
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.network_type = network_type
        self.split = split
        self.cfg = cfg or {}
        self.windows: list[FreddieWindow] = []
        self._loaded = False

    def load(self) -> None:
        """Load dataset from disk. Falls back to synthetic data if unavailable."""
        if not self.data_dir.exists():
            warnings.warn(
                f"[FreddieDataset] Data directory '{self.data_dir}' not found. "
                "Generating SYNTHETIC data for testing. "
                "See data/README_data.md for real Freddie Mac data setup.",
                UserWarning,
            )
            self.windows = self._generate_synthetic_windows()
        else:
            self.windows = self._load_from_disk()

        self._loaded = True
        n_default = sum(w.labels.sum().item() for w in self.windows)
        n_total = sum(len(w) for w in self.windows)
        print(
            f"[FreddieDataset] Loaded {len(self.windows)} windows ({self.split}), "
            f"{n_total} node-window observations, "
            f"default rate={n_default/max(n_total,1):.3f}"
        )

    def _load_from_disk(self) -> list[FreddieWindow]:
        """Load pre-processed windows from disk."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "See data/README_data.md for data preparation instructions."
            )

        windows = []
        window_dirs = sorted(split_dir.glob("window_*/"))
        for wdir in window_dirs:
            window = self._load_window(wdir)
            if window is not None:
                windows.append(window)
        return windows

    def _load_window(self, wdir: Path) -> Optional[FreddieWindow]:
        """Load one window directory."""
        try:
            import pandas as pd
            snapshots_feats, snapshots_adjs = [], []
            for t in range(1, WINDOW_SIZE + 1):
                feat_path = wdir / f"snapshot_{t}_features.parquet"
                adj_path = wdir / f"snapshot_{t}_{self.network_type}_adj.npz"

                if not feat_path.exists() or not adj_path.exists():
                    return None

                feats_df = pd.read_parquet(feat_path)
                feats = torch.FloatTensor(feats_df[ALL_FEATURES].values)

                import scipy.sparse as sp
                adj_sp = sp.load_npz(str(adj_path))
                adj = torch.FloatTensor(adj_sp.toarray())

                snapshots_feats.append(feats)
                snapshots_adjs.append(adj)

            labels_df = pd.read_parquet(wdir / "labels.parquet")
            labels = torch.FloatTensor(labels_df[TARGET_FEATURE].values)
            node_ids = labels_df["loan_id"].values

            return FreddieWindow(snapshots_feats, snapshots_adjs, labels, node_ids)

        except Exception as e:
            warnings.warn(f"Failed to load window {wdir}: {e}")
            return None

    def _generate_synthetic_windows(
        self,
        num_windows: int = 13,
        num_nodes: int = 500,
        default_rate: float = 0.05,
    ) -> list[FreddieWindow]:
        """Generate synthetic windows for testing the pipeline.

        Creates random data with realistic shapes matching the paper.
        Synthetic data does NOT reproduce paper results.
        """
        rng = np.random.default_rng(42)
        windows = []

        num_features = len(ALL_FEATURES)
        n = num_nodes

        if self.network_type == "double":
            adj_size = 2 * n  # supra adjacency
        else:
            adj_size = n

        for _ in range(num_windows):
            # Random node features (normalized to [0,1])
            snaps_f = []
            snaps_a = []
            for _ in range(WINDOW_SIZE):
                feats = torch.FloatTensor(rng.uniform(0, 1, (n, num_features)))
                # Sparse random adjacency (area or company type)
                adj_np = (rng.uniform(0, 1, (adj_size, adj_size)) < 0.01).astype(np.float32)
                adj_np = np.maximum(adj_np, adj_np.T)  # symmetric
                np.fill_diagonal(adj_np, 0)
                adj = torch.FloatTensor(adj_np)
                snaps_f.append(feats)
                snaps_a.append(adj)

            # Labels with ~5% default rate (Table A.2: 7426/148520 ≈ 5%)
            labels_np = (rng.uniform(0, 1, n) < default_rate).astype(np.float32)
            labels = torch.FloatTensor(labels_np)
            node_ids = np.arange(n)

            windows.append(FreddieWindow(snaps_f, snaps_a, labels, node_ids))

        return windows

    def __len__(self) -> int:
        assert self._loaded, "Call .load() before accessing dataset."
        return len(self.windows)

    def __getitem__(self, idx: int) -> FreddieWindow:
        assert self._loaded, "Call .load() before accessing dataset."
        return self.windows[idx]

    def __repr__(self) -> str:
        return (
            f"FreddieDataset(split={self.split}, "
            f"network={self.network_type}, "
            f"windows={len(self.windows)}, loaded={self._loaded})"
        )
