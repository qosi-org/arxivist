"""
Price estimation utilities for SPG-UVM.

Implements Monte Carlo price estimators described in Section 4.1.3:

- Actor price: lower bound estimate using 2^19 MC paths with fixed deterministic policy.
  Reported with 95% confidence interval.
- Critic price: pointwise evaluation V_phi_0(x_0) at the initial state.
  No bias guarantee (depends on value network accuracy).

Reference: Section 4.1.3 of arXiv:2605.06670.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


class PriceEstimator:
    """
    Monte Carlo price estimators for robust option pricing.

    The "actor price" is the robust option price estimated by:
    1. Simulating 2^19 = 524,288 paths under the DETERMINISTIC policy
       (TUVM(m_theta*(x)) or threshold(q_theta*(x))).
    2. Discounting the terminal payoff: e^{-r*T} * g(X_T).
    3. Reporting the sample mean and 95% confidence interval.

    Section 4.1.3 of arXiv:2605.06670.
    """

    def actor_price_with_ci(
        self,
        discounted_payoffs: Tensor,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Estimate actor price from discounted MC payoffs.

        Args:
            discounted_payoffs: e^{-r*T} * g(X_T^{pi}) for each path, shape [M].
            confidence_level:   Default 0.95 (Section 4.1.3).

        Returns:
            (mean, lower_ci, upper_ci): Price estimate with confidence interval.
        """
        payoffs_np = discounted_payoffs.detach().cpu().numpy().astype(np.float64)
        M = len(payoffs_np)
        mean = payoffs_np.mean()
        std = payoffs_np.std(ddof=1)
        sem = std / np.sqrt(M)

        # z-score for the given confidence level (two-tailed)
        from scipy import stats
        z = stats.norm.ppf((1.0 + confidence_level) / 2.0)
        margin = z * sem

        return float(mean), float(mean - margin), float(mean + margin)

    def relative_error(self, estimate: float, reference: float) -> float:
        """
        Compute relative error: |estimate - reference| / reference.

        Args:
            estimate:  Estimated price.
            reference: Reference (benchmark) price.

        Returns:
            Relative error as a fraction (e.g., 0.02 = 2%).
        """
        if abs(reference) < 1e-12:
            return float("nan")
        return abs(estimate - reference) / abs(reference)

    def format_result(
        self,
        mean: float,
        lower: float,
        upper: float,
        reference: Optional[float] = None,
        label: str = "Actor price",
    ) -> str:
        """Format a price estimate for display."""
        ci_half = (upper - lower) / 2.0
        s = f"{label}: {mean:.4f} ± {ci_half:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])"
        if reference is not None:
            rel_err = self.relative_error(mean, reference)
            s += f"  |  Reference: {reference:.4f}  |  Rel. error: {rel_err:.2%}"
        return s


# Type hint fix
from typing import Optional  # noqa: E402
