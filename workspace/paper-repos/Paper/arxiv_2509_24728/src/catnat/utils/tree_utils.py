"""
Binary tree indexing utilities for catnat.

Implements the hierarchical binary tree structure described in Section 4.2.1.
Precomputes two buffers at init time that enable a fully vectorized
O(K log K) forward pass in CatNat — no Python loops over nodes.

Notation follows the paper exactly:
  - H = log2(K): tree depth
  - K-1 internal nodes, each holding one Bernoulli activation a_i
  - K leaf nodes, each holding one categorical probability p_k
  - Node at hierarchy h is reached by decisions b_1, ..., b_{h-1}
  - b_h = 1 means "go left", b_h = 0 means "go right" (paper convention)

Paper reference: Section 4.2.1, Eqs. 7–10, Figure 3.
"""

import math

import torch
from torch import Tensor


class BinaryTreeIndex:
    """Precomputed index buffers for the catnat binary tree.

    Builds two buffers once at construction time:
      - leaf_paths:     [K, H] int  — binary (L/R) path from root to each leaf
      - node_path_mask: [K-1, H_max] int — for each internal node, its depth-path prefix

    Both buffers are registered as non-learnable tensors and moved with .to(device).

    Args:
        K: Number of categories. Must be a power of 2.

    Raises:
        ValueError: If K is not a power of 2.
    """

    def __init__(self, K: int) -> None:
        self.validate_K(K)
        self.K = K
        self.H = int(math.log2(K))  # tree depth = number of hierarchy levels

        # [K, H]: binary path for each leaf (b_1, ..., b_H)
        # Row k gives the left/right decisions to reach leaf p_k.
        # Paper: leaf indexed by binary string b = [b_1, ..., b_H], Section 4.2.1
        self.leaf_paths: Tensor = self._build_leaf_paths()  # shape [K, H]

        # [K-1, H]: for each internal node i, the binary prefix path to reach it
        # Used to map each internal node to its position in leaf_paths
        self.node_to_leaf_path_col: Tensor = self._build_node_column_index()  # shape [K-1]

        # [K, K-1]: ancestor mask — ancestor_mask[k, i] = 1 iff node i is an ancestor of leaf k
        self.ancestor_mask: Tensor = self._build_ancestor_mask()  # shape [K, K-1]

        # [K, K-1]: the branch taken at each ancestor node on the path to leaf k
        # branch_taken[k, i] = b_h value at node i when traversing to leaf k
        # Used to select a_i vs (1 - a_i) in the product (Eq. 8)
        self.branch_taken: Tensor = self._build_branch_taken()   # shape [K, K-1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def validate_K(K: int) -> None:
        """Raise ValueError if K is not a power of 2.

        Args:
            K: Number of categories.
        """
        if K < 2 or (K & (K - 1)) != 0:
            raise ValueError(
                f"K={K} is not a power of 2. catnat requires K ∈ {{2,4,8,16,32,...}}. "
                "See RISK-01 in architecture_plan.json."
            )

    def get_leaf_paths(self) -> Tensor:
        """Return the [K, H] binary path matrix.

        Returns:
            Tensor of shape [K, H] with values in {0, 1}.
            leaf_paths[k, h] = b_{h+1} for leaf k (0=right, 1=left).
        """
        return self.leaf_paths

    def get_ancestor_mask(self) -> Tensor:
        """Return the [K, K-1] ancestor mask.

        Returns:
            Boolean Tensor of shape [K, K-1].
            ancestor_mask[k, i] = True iff internal node i is an ancestor of leaf k.
        """
        return self.ancestor_mask.bool()

    def get_branch_taken(self) -> Tensor:
        """Return the [K, K-1] branch direction matrix.

        Returns:
            Tensor of shape [K, K-1] with values in {0, 1}.
            branch_taken[k, i] = direction (0 or 1) taken at node i to reach leaf k.
            Meaningful only where ancestor_mask[k, i] = True.
        """
        return self.branch_taken

    def to(self, device: torch.device) -> "BinaryTreeIndex":
        """Move all buffers to the given device in-place.

        Args:
            device: Target torch device.

        Returns:
            self (for chaining).
        """
        self.leaf_paths = self.leaf_paths.to(device)
        self.ancestor_mask = self.ancestor_mask.to(device)
        self.branch_taken = self.branch_taken.to(device)
        self.node_to_leaf_path_col = self.node_to_leaf_path_col.to(device)
        return self

    def __repr__(self) -> str:
        return f"BinaryTreeIndex(K={self.K}, H={self.H})"

    # ------------------------------------------------------------------
    # Private construction methods
    # ------------------------------------------------------------------

    def _build_leaf_paths(self) -> Tensor:
        """Build [K, H] leaf path matrix.

        Leaf k's path is the H-bit binary representation of k,
        with the most-significant bit first (top of tree = first decision).
        b_h = 1 → go left; b_h = 0 → go right. (Paper Section 4.2.1)
        """
        paths = torch.zeros(self.K, self.H, dtype=torch.long)
        for k in range(self.K):
            for h in range(self.H):
                # Extract bit h from k, MSB first
                bit_pos = self.H - 1 - h
                paths[k, h] = (k >> bit_pos) & 1
        return paths

    def _build_ancestor_mask(self) -> Tensor:
        """Build [K, K-1] ancestor mask.

        For each leaf k and each internal node i,
        ancestor_mask[k, i] = 1 iff node i lies on the path from root to leaf k.

        Internal nodes are numbered 0..K-2 in breadth-first order:
          - Node 0: root (h=1)
          - Nodes 1,2: h=2
          - Nodes 3,4,5,6: h=3
          - ...
        """
        mask = torch.zeros(self.K, self.K - 1, dtype=torch.long)
        for k in range(self.K):
            node_idx = 0
            for h in range(self.H):
                mask[k, node_idx] = 1
                direction = self.leaf_paths[k, h].item()
                # Left child of node at BFS index node_idx: 2*node_idx + 1 + direction
                # Right child: 2*node_idx + 2 - direction
                # Standard binary heap indexing: left=2i+1, right=2i+2
                node_idx = 2 * node_idx + 1 + int(direction)
                if node_idx >= self.K - 1:
                    break  # reached leaf level
        return mask

    def _build_branch_taken(self) -> Tensor:
        """Build [K, K-1] branch direction matrix.

        branch_taken[k, i] = the b_h value (0 or 1) taken at node i
        when traversing from root to leaf k.
        Only meaningful where ancestor_mask[k, i] = 1.
        """
        branch = torch.zeros(self.K, self.K - 1, dtype=torch.long)
        for k in range(self.K):
            node_idx = 0
            for h in range(self.H):
                direction = self.leaf_paths[k, h].item()
                branch[k, node_idx] = int(direction)
                node_idx = 2 * node_idx + 1 + int(direction)
                if node_idx >= self.K - 1:
                    break
        return branch

    def _build_node_column_index(self) -> Tensor:
        """Map each internal node to its hierarchy level h.

        Returns [K-1] tensor where entry i = hierarchy level h (0-indexed)
        of internal node i in BFS order.
        """
        levels = torch.zeros(self.K - 1, dtype=torch.long)
        for i in range(self.K - 1):
            # BFS level of node i: floor(log2(i+1))
            levels[i] = int(math.floor(math.log2(i + 1)))
        return levels
