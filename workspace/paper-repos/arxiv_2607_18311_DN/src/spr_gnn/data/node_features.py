"""
Per-node feature extraction (Sec 4.1, Fig. 4).

"Each node carries a four-dimensional feature vector: node degree, a binary
leaf indicator, the topological distance to the root in number of branches,
and a categorical taxonomic identifier." The first three are used directly;
the identifier is mapped through TaxonomicEmbedding elsewhere.
"""
from __future__ import annotations

import networkx as nx
import torch


class NodeFeatureExtractor:
    """Extracts (degree, is_leaf, root_distance) continuous features and a species id per node."""

    def extract(self, graph: nx.DiGraph, species_id: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            graph: rooted directed graph from NewickToGraph.parse(), with
                nodes labelled 0..N-1 and a single in-degree-zero root.
            species_id: integer id identifying which species/source dataset
                this tree belongs to (SIR implementation_assumptions[4]:
                treated as per-species rather than per-isolate -- swap here
                if the released dataset schema says otherwise).

        Returns:
            continuous_features: [N, 3] float tensor (degree, is_leaf, root_distance).
            node_ids: [N] long tensor, all entries set to `species_id` (one
                categorical id per node, broadcast across the tree).
        """
        n = graph.number_of_nodes()
        roots = [node for node, in_deg in graph.in_degree() if in_deg == 0]
        assert len(roots) == 1, f"Expected exactly one root, found {len(roots)}"
        root = roots[0]

        # Undirected degree (paper does not specify directed vs undirected degree;
        # we use total degree over the underlying undirected tree, i.e. in+out on
        # the bidirectional graph, which is the natural reading of "node degree"
        # for an otherwise-undirected phylogeny).
        undirected = graph.to_undirected()
        degrees = dict(undirected.degree())

        # Topological distance to root, in number of branches (BFS on directed tree).
        root_distances = nx.single_source_shortest_path_length(graph, root)

        continuous = torch.zeros((n, 3), dtype=torch.float32)
        node_ids = torch.full((n,), fill_value=species_id, dtype=torch.long)
        for node in graph.nodes():
            is_leaf = float(graph.out_degree(node) == 0)
            continuous[node, 0] = float(degrees[node])
            continuous[node, 1] = is_leaf
            continuous[node, 2] = float(root_distances[node])

        return continuous, node_ids

    def __repr__(self) -> str:  # noqa: D105
        return "NodeFeatureExtractor()"
