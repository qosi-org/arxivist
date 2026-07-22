"""
Synthetic multimodal spatial dataset.

IMPORTANT: The paper's evaluation cohort (11 private melanoma samples,
54,912 total Visium spots, 5 aligned modality blocks) cannot be publicly
released (Section 4.1, Appendix G.1) and there is no public dataset at this
exact resolution / modality combination (SIR ambiguities[5], confidence 0.8).

This module generates a SYNTHETIC dataset matching the paper's documented
shapes so the full LATTICE pipeline (training, evaluation, notebook) is
runnable end-to-end without access to the private cohort. It does NOT
attempt to reproduce the paper's actual reported numbers (Table 2/3) --
see README.md "Reproducibility Notes" and data/README_data.md for how to
substitute real data.

Synthetic generation strategy:
  - Spot coordinates are laid out on a roughly square Visium-like hex/grid
    approximation (simple 2D grid + jitter) matching `spots_per_sample`.
  - Each modality block is drawn from a modality-specific Gaussian mixture
    over `num_spatial_domains` latent "tissue domains" so that (a) spatially
    nearby spots are more likely to share a domain (via a smooth spatial
    field) and (b) different modalities are correlated but not identical for
    the same domain, so cross-modal alignment and reconstruction are
    non-trivial learning problems.
  - `presence_mask` marks entries as observed with probability
    `modality_presence_prob` per modality, independently per spot, to
    exercise the modality-aware fusion path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSpatialMultimodalDataset(Dataset):
    """Synthetic stand-in for the paper's private multimodal melanoma cohort.

    Args:
        num_samples: number of "slides" (11 in the paper).
        spots_per_sample: spots per slide (4,992 in the paper).
        gene_count_range: (min, max) genes per modality block, sampled per
            sample (paper reports the final intersected gene count G in
            [129, 322]; each modality block has this same width D_b = G,
            so total D = 5 * G, matching Appendix H's "D = 5G" statement).
        num_modality_blocks: number of modality blocks B (5).
        num_spatial_domains: number of latent tissue domains used to
            generate spatially-correlated synthetic signal.
        modality_presence_prob: probability that a given modality's reading
            is "present" (non-imputed) at a given spot.
        seed: RNG seed for reproducible synthetic data generation.
    """

    def __init__(
        self,
        num_samples: int = 11,
        spots_per_sample: int = 4992,
        gene_count_range: tuple[int, int] = (129, 322),
        num_modality_blocks: int = 5,
        num_spatial_domains: int = 6,
        modality_presence_prob: float = 0.9,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.spots_per_sample = spots_per_sample
        self.gene_count_range = gene_count_range
        self.num_modality_blocks = num_modality_blocks
        self.num_spatial_domains = num_spatial_domains
        self.modality_presence_prob = modality_presence_prob
        self.seed = seed

        self.seed = seed
        # IMPORTANT: modality_dims (and hence the model's input-adapter sizes)
        # must be a pure function of `gene_count_range`, NOT of `num_samples`
        # or any other runtime setting. Earlier versions derived g_max from a
        # `num_samples`-sized random draw, which meant a checkpoint trained
        # with e.g. num_samples=2 (as in --debug) had a different feature
        # width than a dataset instantiated with num_samples=11 at eval time,
        # causing a state_dict shape mismatch. Using the fixed upper bound of
        # `gene_count_range` as g_max avoids this entirely: any two
        # SyntheticSpatialMultimodalDataset instances with the same
        # `gene_count_range` produce identically-shaped modality blocks
        # regardless of `num_samples`.
        self._g_max = int(gene_count_range[1])

    def __len__(self) -> int:
        return self.num_samples

    @property
    def modality_dims(self) -> list[int]:
        """Feature dim per modality block for THIS dataset config.

        Fixed to `gene_count_range[1]` (see note in __init__) so it is
        independent of `num_samples`, keeping checkpoints portable across
        dataset instantiations that share the same `gene_count_range`.
        """
        return [self._g_max] * self.num_modality_blocks

    def _spatial_field(self, coords: np.ndarray, domain_id: int, n_domains: int) -> np.ndarray:
        """Smooth spatial scalar field used to bias modality signal by tissue domain."""
        cx = (domain_id % int(math.sqrt(n_domains) + 1)) / max(1, int(math.sqrt(n_domains)))
        cy = (domain_id // int(math.sqrt(n_domains) + 1)) / max(1, int(math.sqrt(n_domains)))
        center = np.array([cx, cy]) * coords.max(axis=0)
        dist = np.linalg.norm(coords - center, axis=1)
        return np.exp(-dist / (coords.max() + 1e-6))

    def __getitem__(self, idx: int) -> dict:
        """Generate one synthetic sample (slide).

        Returns:
            dict with:
                'modality_blocks': list of B float32 tensors [N, G_max]
                'presence_mask': [N, B] float32 tensor
                'coords': [N, 2] float32 tensor
                'domain_labels': [N] int64 tensor (ground-truth synthetic
                    "tissue domain" -- useful only for sanity-checking the
                    synthetic generator itself, NOT a paper-reported label)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Sample index {idx} out of range [0, {self.num_samples})")

        rng = np.random.default_rng(self.seed + idx)
        n = self.spots_per_sample
        g_max = self._g_max
        g = int(rng.integers(self.gene_count_range[0], self.gene_count_range[1] + 1))
        b = self.num_modality_blocks

        # 1. Layout spots on an approximately square grid with jitter (stand-in for Visium array coords)
        side = int(math.ceil(math.sqrt(n)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        coords_full = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        coords = coords_full[:n] + rng.normal(0, 0.15, size=(n, 2)).astype(np.float32)

        # 2. Assign each spot a soft mixture over spatial domains, biased by the spatial field
        fields = np.stack(
            [self._spatial_field(coords, d, self.num_spatial_domains) for d in range(self.num_spatial_domains)],
            axis=1,
        )  # [N, num_domains]
        domain_probs = fields / fields.sum(axis=1, keepdims=True).clip(min=1e-6)
        domain_labels = np.array(
            [rng.choice(self.num_spatial_domains, p=domain_probs[i]) for i in range(n)]
        )

        # 3. Per-modality Gaussian-mixture signal correlated with domain, independent modality-specific noise
        domain_centers = rng.normal(0, 1.5, size=(self.num_spatial_domains, b, g_max)).astype(np.float32)
        modality_blocks = []
        for m in range(b):
            base = domain_centers[domain_labels, m, :]  # [N, g_max]
            noise = rng.normal(0, 0.5, size=(n, g_max)).astype(np.float32)
            block = base + noise
            if g < g_max:
                block[:, g:] = 0.0  # zero-pad beyond this sample's actual gene count
            modality_blocks.append(torch.from_numpy(block.astype(np.float32)))

        # 4. Per-spot, per-modality presence mask
        presence_mask = (
            rng.random((n, b)) < self.modality_presence_prob
        ).astype(np.float32)
        presence_mask = torch.from_numpy(presence_mask)

        return {
            "modality_blocks": modality_blocks,
            "presence_mask": presence_mask,
            "coords": torch.from_numpy(coords),
            "domain_labels": torch.from_numpy(domain_labels.astype(np.int64)),
        }
