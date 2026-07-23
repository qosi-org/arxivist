"""
detection/union_find.py
------------------------
Surfaces persistent cohorts as connected components of the filtered
co-occurrence graph, then applies the maximum cohort size filter.

Paper: Kamat (2026), Section 4.2 — "We then run union-find on the filtered
graph to surface connected components."
"""
from __future__ import annotations

from typing import FrozenSet, List

import networkx as nx


class CohortSurface:
    """
    Surfaces connected components from the filtered co-occurrence graph,
    each representing a candidate persistent wallet cohort.

    Uses networkx.connected_components() which internally applies a
    BFS-based union-find equivalent with path compression.

    Args:
        max_size: Maximum cohort size to retain (default: 12).
                  Removes "noise hubs" — Section 4.2 states MAX_COHORT_SIZE=12.
                  At cutoff=3, reduces 1,161 raw components → 1,012 after filter.

    Paper reference:
        Section 4.2 — union-find on G_filtered, then MAX_COHORT_SIZE filter.
    """

    def __init__(self, max_size: int = 12) -> None:
        self.max_size = max_size

    def surface(self, G_filtered: nx.Graph) -> List[FrozenSet[str]]:
        """
        Find all connected components of G_filtered using networkx's
        connected_components (BFS union-find equivalent).

        Args:
            G_filtered: Filtered co-occurrence graph from CoOccurrenceGraph.build().

        Returns:
            List of frozensets, each containing wallet addresses in one cohort.
            Components with size > max_size are excluded.
            Paper: 1,161 raw → 1,012 after MAX_COHORT_SIZE=12 filter.
        """
        raw_components: List[FrozenSet[str]] = [
            frozenset(component)
            for component in nx.connected_components(G_filtered)
            if len(component) >= 2  # cohorts must have at least 2 wallets
        ]
        return self.apply_size_filter(raw_components)

    def apply_size_filter(
        self,
        components: List[FrozenSet[str]],
        max_size: int | None = None,
    ) -> List[FrozenSet[str]]:
        """
        Discard components with more than max_size wallets.

        Args:
            components: Raw list of frozenset cohort candidates.
            max_size: Override; uses self.max_size if None.

        Returns:
            Filtered list of frozensets.
        """
        limit = max_size if max_size is not None else self.max_size
        return [c for c in components if len(c) <= limit]

    def __repr__(self) -> str:
        return f"CohortSurface(max_size={self.max_size})"
