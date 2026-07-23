"""
detection/graph.py
------------------
Builds the weighted wallet co-occurrence graph G=(V,E) and applies
the edge-weight filter described in Section 4.2.

Paper: Kamat (2026), Section 4.2 — "Cross-launch persistent-cohort surfacing".

EQ3: w(u, v) = |{L : u ∈ first10(L) ∧ v ∈ first10(L)}|
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, Tuple

import networkx as nx
import pandas as pd


class CoOccurrenceGraph:
    """
    Builds and filters the undirected wallet co-occurrence graph.

    Vertices V are wallet addresses. An edge (u, v) exists when wallets u and v
    both appear among the first-N buyers of at least one launch. Edge weight is
    the integer count of co-occurrences across all launches.

    Paper reference:
        Section 4.2 — "Let G = (V,E) be an undirected graph where vertices V are
        wallet addresses and edges E record co-occurrence."

    Args:
        min_weight: Minimum co-occurrence count to retain an edge (default: 3).
                    "The threshold of 3 is conservative." — Section 4.2.
    """

    def __init__(self, min_weight: int = 3) -> None:
        self.min_weight = min_weight

    def build(self, intra_index: pd.DataFrame) -> nx.Graph:
        """
        Build the raw co-occurrence graph from the intra-launch index,
        then return the filtered graph G_filtered (edges with weight >= min_weight).

        Pair generation uses itertools.combinations for memory efficiency —
        we accumulate edge weights in a dict before constructing the networkx graph,
        avoiding materializing the full raw graph object.

        Paper: at cutoff=3, 9,788 qualifying pairs remain (Appendix A).

        Args:
            intra_index: DataFrame from IntraLaunchExtractor with columns
                         {mint, wallet, rank, block_time, sol_committed}.

        Returns:
            G_filtered: nx.Graph with only edges where weight >= min_weight.
        """
        # Accumulate co-occurrence counts: (wallet_a, wallet_b) → count
        # EQ3: w(u,v) = |{L : u ∈ first10(L) ∧ v ∈ first10(L)}|
        edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for mint, group in intra_index.groupby("mint"):
            wallets = group["wallet"].tolist()
            if len(wallets) < 2:
                continue
            # Generate all unordered pairs within this launch's first-N buyers
            for u, v in itertools.combinations(wallets, 2):
                # Canonical ordering to ensure (u,v) == (v,u)
                key = (min(u, v), max(u, v))
                edge_counts[key] += 1

        # Build filtered graph: only edges with weight >= min_weight
        G = nx.Graph()
        for (u, v), w in edge_counts.items():
            if w >= self.min_weight:
                G.add_edge(u, v, weight=w)

        return G

    def get_edge_weight_distribution(self, G: nx.Graph) -> pd.Series:
        """Return a Series of edge weights (useful for ablation diagnostics)."""
        weights = [data["weight"] for _, _, data in G.edges(data=True)]
        return pd.Series(weights, name="edge_weight")

    def filter_edges(self, G: nx.Graph, min_weight: int) -> nx.Graph:
        """
        Apply a different min_weight to an already-built graph.
        Used by ablation runner to test cutoffs [2, 3, 5] without rebuilding.
        """
        G_filtered = nx.Graph()
        for u, v, data in G.edges(data=True):
            if data["weight"] >= min_weight:
                G_filtered.add_edge(u, v, weight=data["weight"])
        return G_filtered

    def __repr__(self) -> str:
        return f"CoOccurrenceGraph(min_weight={self.min_weight})"
