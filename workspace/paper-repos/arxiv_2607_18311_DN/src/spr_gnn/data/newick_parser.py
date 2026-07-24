"""
Newick string -> PyTorch Geometric graph conversion (Sec 4.1, Fig. 4).

"Each Newick tree is parsed with Biopython and converted to a directed
graph whose root is the unique in-degree-zero vertex. Then, it is encoded
as a bidirectional graph in PyTorch Geometric so that messages propagate
in both directions." (Sec 4.1)
"""
from __future__ import annotations

import networkx as nx
import torch
from Bio import Phylo
from io import StringIO


class NewickToGraph:
    """Parses a Newick string into a rooted directed graph, then a bidirectional edge_index."""

    def parse(self, newick_str: str) -> nx.DiGraph:
        """Parse a Newick string into a directed graph rooted at the tree root.

        Args:
            newick_str: Newick-formatted tree string.

        Returns:
            A networkx.DiGraph with a single in-degree-zero root node, integer
            node ids as node keys, and `is_leaf` / `name` node attributes.
        """
        tree = Phylo.read(StringIO(newick_str), "newick")
        graph = nx.DiGraph()

        node_ids: dict[object, int] = {}

        def get_id(clade) -> int:
            if id(clade) not in node_ids:
                node_ids[id(clade)] = len(node_ids)
            return node_ids[id(clade)]

        root_id = get_id(tree.root)
        graph.add_node(root_id, is_leaf=tree.root.is_terminal(), name=tree.root.name)

        def add_children(clade) -> None:
            parent_id = get_id(clade)
            for child in clade.clades:
                child_id = get_id(child)
                graph.add_node(child_id, is_leaf=child.is_terminal(), name=child.name)
                graph.add_edge(parent_id, child_id)
                add_children(child)

        add_children(tree.root)

        assert graph.in_degree(root_id) == 0, "Root must have in-degree zero"
        return graph

    def to_bidirectional_edge_index(self, graph: nx.DiGraph) -> torch.LongTensor:
        """Duplicate every directed edge (u->v and v->u) so messages propagate both ways.

        Args:
            graph: a directed graph from `parse()`.

        Returns:
            [2, E] LongTensor edge_index (E = 2 * num_original_edges), PyG convention.
        """
        edges = list(graph.edges())
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        return torch.tensor([src, dst], dtype=torch.long)

    def __repr__(self) -> str:  # noqa: D105
        return "NewickToGraph()"
