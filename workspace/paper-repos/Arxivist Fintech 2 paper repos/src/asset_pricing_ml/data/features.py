"""
data/features.py — Feature construction for Gu, Kelly, Xiu (2020).

Implements the predictor vector z_it = x_t ⊗ c_it from Section 2.1:
    z_it is a P×1 vector with P = P_c × P_x = 94 × 9 + 74 = 920
    where c_it are 94 cross-sectionally ranked characteristics and
    x_t are 8 macro predictors + constant.

Paper reference: Section 2.1, Equation (21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Constructs the 920-dimensional predictor vector z_it from Section 2.1.

    The overarching feature structure is:
        z_it = (x_t ⊗ c_it) concatenated with 74 industry dummies

    This nests the beta pricing model: if beta_it = theta1 @ c_it and
    lambda_t = theta2 @ x_t, then E[r_it+1] = beta_it' @ lambda_t = z_it' @ theta.

    Paper reference: Equation (21) and surrounding discussion.

    Args:
        n_characteristics: Number of stock characteristics (default 94).
        n_macro: Number of macro predictors + constant (default 9 = 8 + 1).
        n_industry: Number of industry dummies (default 74).
    """

    def __init__(
        self,
        n_characteristics: int = 94,
        n_macro: int = 9,
        n_industry: int = 74,
    ):
        self.n_characteristics = n_characteristics
        self.n_macro = n_macro
        self.n_industry = n_industry
        self.total_features = n_characteristics * n_macro + n_industry
        # Validate the 920-feature count from the paper
        assert self.total_features == 920, (
            f"Expected 920 features, got {self.total_features}. "
            f"Check n_characteristics={n_characteristics}, n_macro={n_macro}, "
            f"n_industry={n_industry}."
        )

    def cross_sectional_rank(self, c_it: np.ndarray) -> np.ndarray:
        """Cross-sectionally rank characteristics and map to [-1, 1].

        Paper Section 2.1 footnote 29: "We cross-sectionally rank all stock
        characteristics period-by-period and map these ranks into the [-1, 1]
        interval following Kelly, Pruitt, and Su (2019)."

        Args:
            c_it: Raw characteristics array of shape [N_t, P_c].

        Returns:
            Ranked array of shape [N_t, P_c] with values in [-1, 1].
        """
        N = c_it.shape[0]
        if N == 1:
            return np.zeros_like(c_it)
        # Rank each column separately (cross-sectional ranks)
        ranked = np.zeros_like(c_it, dtype=float)
        for j in range(c_it.shape[1]):
            col = c_it[:, j]
            finite_mask = np.isfinite(col)
            if finite_mask.sum() == 0:
                ranked[:, j] = 0.0
                continue
            ranks = pd.Series(col).rank(method="average", na_option="keep")
            n_valid = finite_mask.sum()
            # Map ranks to [-1, 1]: rank / (N_valid / 2) - 1
            ranked[:, j] = np.where(
                finite_mask,
                ranks.values / (n_valid / 2.0) - 1.0,
                0.0,  # missing → 0 (median rank)
            )
        return np.clip(ranked, -1.0, 1.0)

    def fill_missing(self, c_it: np.ndarray) -> np.ndarray:
        """Replace missing values with cross-sectional median.

        Paper Section 2.1 footnote 30: "We replace with the cross-sectional
        median at each month for each stock, respectively."

        Args:
            c_it: Characteristics array [N_t, P_c]; NaN for missing.

        Returns:
            Array with NaNs replaced by column medians.
        """
        c_filled = c_it.copy()
        for j in range(c_it.shape[1]):
            col = c_it[:, j]
            median = np.nanmedian(col)
            c_filled[np.isnan(c_filled[:, j]), j] = median
        return c_filled

    def build_interaction_features(
        self,
        c_it_ranked: np.ndarray,  # [N_t, 94]
        x_t: np.ndarray,          # [9] = 8 macro + constant
    ) -> np.ndarray:
        """Build Kronecker interaction features: z_it = x_t ⊗ c_it.

        Paper: Equation (21) — interaction between stock characteristics and
        macro predictors motivates the conditional beta pricing model.

        Args:
            c_it_ranked: Cross-sectionally ranked characteristics [N_t, 94].
            x_t: Macro predictor vector [9] (constant first or last).

        Returns:
            Interaction features [N_t, 94*9=846].
        """
        assert c_it_ranked.shape[1] == self.n_characteristics, (
            f"Expected {self.n_characteristics} characteristics, got {c_it_ranked.shape[1]}"
        )
        assert x_t.shape[0] == self.n_macro, (
            f"Expected {self.n_macro} macro predictors, got {x_t.shape[0]}"
        )
        # Kronecker product: for each stock i, z = x_t ⊗ c_it
        # Result shape: [N_t, 94 * 9]
        # Implemented as: c_it_ranked[:, :, None] * x_t[None, None, :] → reshape
        z_interactions = c_it_ranked[:, :, None] * x_t[None, None, :]  # [N, 94, 9]
        return z_interactions.reshape(c_it_ranked.shape[0], self.n_characteristics * self.n_macro)

    def build_full_feature_vector(
        self,
        c_it: np.ndarray,         # [N_t, 94] raw (may have NaN)
        x_t: np.ndarray,          # [8] macro predictors (constant added internally)
        industry_dummies: np.ndarray,  # [N_t, 74]
    ) -> np.ndarray:
        """Build the full 920-dim feature vector z_it for all stocks at time t.

        Full pipeline:
        1. Fill missing characteristics with cross-sectional median
        2. Cross-sectionally rank characteristics to [-1, 1]
        3. Prepend constant to macro vector: x_t = [1, macro...]
        4. Compute Kronecker interactions: c_it ⊗ x_t → [N, 846]
        5. Concatenate industry dummies → [N, 920]

        Paper: Section 2.1, Equation (21)

        Args:
            c_it: Raw characteristics [N_t, 94].
            x_t: Macro predictors [8] (constant added internally).
            industry_dummies: One-hot industry indicators [N_t, 74].

        Returns:
            Full feature matrix z_it of shape [N_t, 920].
        """
        assert c_it.shape[1] == self.n_characteristics
        assert x_t.shape[0] == self.n_macro - 1, (
            f"Expected {self.n_macro-1} raw macro predictors; constant added here."
        )
        assert industry_dummies.shape[1] == self.n_industry

        # Step 1: Fill missing
        c_filled = self.fill_missing(c_it)

        # Step 2: Cross-sectional ranking to [-1, 1]
        c_ranked = self.cross_sectional_rank(c_filled)

        # Step 3: Prepend constant 1 to macro vector
        x_with_const = np.concatenate([[1.0], x_t])  # [9]

        # Step 4: Kronecker interactions [N, 846]
        z_interact = self.build_interaction_features(c_ranked, x_with_const)

        # Step 5: Concatenate industry dummies → [N, 920]
        z_full = np.concatenate([z_interact, industry_dummies.astype(float)], axis=1)

        assert z_full.shape[1] == self.total_features, (
            f"Expected {self.total_features} features, got {z_full.shape[1]}"
        )
        return z_full.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"FeatureBuilder(n_chars={self.n_characteristics}, "
            f"n_macro={self.n_macro}, n_industry={self.n_industry}, "
            f"total={self.total_features})"
        )
