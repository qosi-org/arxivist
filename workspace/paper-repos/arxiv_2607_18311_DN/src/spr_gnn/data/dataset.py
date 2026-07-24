"""
Tree-pair dataset and data module (Sec 3 "Pairing and labelling", Sec 4.3).

Expects a master CSV (downloaded via data/download.py from the paper's
Zenodo release) with columns:
    pair_id, tree_a_path, tree_b_path, species, spr_label, split

`split` in {"train", "val", "test"} should already reflect the paper's
70/15/15 partition under seed 42; if the raw Zenodo release does not
include a `split` column, `TreePairDataModule.setup()` will create one
deterministically from `split_seed`.

Per Sec 4.3: "each tree is parsed and converted to a PyTorch Geometric
object once and cached in memory", reducing per-epoch cost from
O(epochs x N) to O(N).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

from spr_gnn.data.newick_parser import NewickToGraph
from spr_gnn.data.node_features import NodeFeatureExtractor


def collate_tree_pairs(
    samples: list[tuple[Data, Data, float]]
) -> tuple[Batch, Batch, torch.FloatTensor]:
    """Collates a list of (tree_a, tree_b, label) into batched PyG Batch objects."""
    trees_a, trees_b, labels = zip(*samples)
    return (
        Batch.from_data_list(list(trees_a)),
        Batch.from_data_list(list(trees_b)),
        torch.tensor(labels, dtype=torch.float32),
    )


class TreePairDataset(Dataset):
    """One row = one labelled (tree_a, tree_b, spr_distance) pair.

    Parses each unique Newick file once and caches the resulting PyG `Data`
    object (Sec 4.3 caching requirement).
    """

    def __init__(self, pairs_df: pd.DataFrame, species_to_id: dict[str, int]):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.species_to_id = species_to_id
        self._parser = NewickToGraph()
        self._extractor = NodeFeatureExtractor()
        self._tree_cache: dict[str, Data] = {}

    def _load_tree(self, nwk_path: str, species: str) -> Data:
        if nwk_path in self._tree_cache:
            return self._tree_cache[nwk_path]
        with open(nwk_path, "r") as f:
            newick_str = f.read()
        graph = self._parser.parse(newick_str)
        edge_index = self._parser.to_bidirectional_edge_index(graph)
        species_id = self.species_to_id[species]
        continuous, node_ids = self._extractor.extract(graph, species_id)
        data = Data(x=continuous, edge_index=edge_index, node_id=node_ids)
        self._tree_cache[nwk_path] = data
        return data

    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> tuple[Data, Data, float]:
        row = self.pairs_df.iloc[idx]
        tree_a = self._load_tree(row["tree_a_path"], row["species"])
        tree_b = self._load_tree(row["tree_b_path"], row["species"])
        return tree_a, tree_b, float(row["spr_label"])

    def __repr__(self) -> str:  # noqa: D105
        return f"TreePairDataset(n_pairs={len(self)})"


class TreePairDataModule:
    """Loads the master pairs CSV and exposes train/val/test DataLoaders.

    Implements the 70/15/15 split under a fixed seed (Sec 4.3). If the
    master CSV does not already contain a `split` column, one is created
    here deterministically.
    """

    def __init__(self, config) -> None:  # config: spr_gnn.utils.config.Config
        self.config = config
        self.species_to_id: dict[str, int] = {}
        self.train_dataset: TreePairDataset | None = None
        self.val_dataset: TreePairDataset | None = None
        self.test_dataset: TreePairDataset | None = None

    def setup(self, master_csv_path: str | None = None, seed: int | None = None) -> None:
        master_csv_path = master_csv_path or self.config.get("data", "master_pairs_csv")
        seed = seed if seed is not None else self.config.get("data", "split_seed", 42)

        path = Path(master_csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Master pairs CSV not found at {path}. Run `python data/download.py` "
                f"first (see data/README_data.md)."
            )
        df = pd.read_csv(path)

        species_list = self.config.get("data", "species")
        self.species_to_id = {name: i for i, name in enumerate(species_list)}

        if "split" not in df.columns:
            train_ratio, val_ratio, test_ratio = self.config.get(
                "data", "train_val_test_split", [0.70, 0.15, 0.15]
            )
            train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)
            relative_val = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(temp_df, test_size=(1 - relative_val), random_state=seed)
        else:
            train_df = df[df["split"] == "train"]
            val_df = df[df["split"] == "val"]
            test_df = df[df["split"] == "test"]

        self.train_dataset = TreePairDataset(train_df, self.species_to_id)
        self.val_dataset = TreePairDataset(val_df, self.species_to_id)
        self.test_dataset = TreePairDataset(test_df, self.species_to_id)

    def _loader(self, dataset: TreePairDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get("data", "num_workers", 4),
            collate_fn=collate_tree_pairs,
        )

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        batch_size = batch_size or self.config.get("training", "batch_size", 32)
        return self._loader(self.train_dataset, batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        batch_size = batch_size or self.config.get("training", "batch_size", 32)
        return self._loader(self.val_dataset, batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        batch_size = batch_size or self.config.get("training", "batch_size", 32)
        return self._loader(self.test_dataset, batch_size, shuffle=False)

    @property
    def num_species(self) -> int:
        return max(len(self.species_to_id), 1)
