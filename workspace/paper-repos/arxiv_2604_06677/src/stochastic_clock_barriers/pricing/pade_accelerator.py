"""
pricing/pade_accelerator.py
============================
Padé resummation for the rho-expansion of barrier prices.

Raw Taylor truncation of the rho-expansion u^rho ≈ sum_n C_n * rho^n
is accurate only for small |rho|. For equity-like correlations (|rho| ~ 0.7)
Padé resummation dramatically extends the usable range (Table 6.12: 36-41x
error reduction at |rho| = 0.7-0.9).

Implements general [L/M] Padé approximant (Eq 6.11) and the [1/1] form
given explicitly in the paper (Eq 5.22).

Paper: Section 5.8, Eqs (5.21)-(5.22); Section 6.3.4, Eq (6.11).
arXiv: 2605.06677v1
"""

import numpy as np
from typing import Optional


class PadeAccelerator:
    """Padé approximant builder and evaluator for barrier price rho-expansions.

    Given Taylor coefficients [C_0, C_1, ..., C_N], constructs the [L/M]
    Padé approximant P_{L,M}(rho) whose Taylor expansion matches up to
    order L+M.

    Paper: Section 5.8, Eq (5.22); Section 6.3.4, Eq (6.11).
    """

    def __init__(self, coefficients: list[float]) -> None:
        """
        Parameters
        ----------
        coefficients : list[float]
            Taylor coefficients [C_0, C_1, ..., C_N] from the rho-expansion.
            C_0 corresponds to the ρ=0 baseline price.
        """
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.N = len(coefficients) - 1     # order of available expansion

        self._numerator: Optional[np.poly1d] = None
        self._denominator: Optional[np.poly1d] = None
        self._L: Optional[int] = None
        self._M: Optional[int] = None

    # ------------------------------------------------------------------
    # [1/1] Padé  (Eq 5.22, explicit formula)
    # ------------------------------------------------------------------

    @classmethod
    def pade_11_explicit(cls, C0: float, C1: float, C2: float) -> "PadeAccelerator":
        """Construct [1/1] Padé from first three Taylor coefficients.

        Paper: Eq (5.22):
            P_{[1/1]}(rho) = (u0 + rho*(u1 - u0*u2/u1)) / (1 - rho*u2/u1)

        This is the minimal Padé that captures the leading-order pole
        structure of the rho-expansion.
        """
        obj = cls([C0, C1, C2])
        if abs(C1) < 1e-14:
            raise ValueError(
                "C1 = u1 ~ 0; [1/1] Padé degenerate (no first-order correction). "
                "Use Taylor truncation instead."
            )
        # Numerator coefficients (degree 1): [C0 + rho*(C1 - C0*C2/C1)]
        a0 = C0
        a1 = C1 - C0 * C2 / C1
        # Denominator coefficients (degree 1): [1 - rho*C2/C1]
        b0 = 1.0
        b1 = -C2 / C1

        obj._numerator   = np.poly1d([a1, a0])[::-1]   # ascending powers
        obj._denominator = np.poly1d([b1, b0])[::-1]
        obj._L = 1
        obj._M = 1
        return obj

    # ------------------------------------------------------------------
    # General [L/M] Padé  (Eq 6.11)
    # ------------------------------------------------------------------

    def build(self, L: int, M: int) -> "PadeAccelerator":
        """Build a general [L/M] Padé approximant.

        Solves the linear system to match Taylor coefficients C_0..C_{L+M}.

        Paper: Section 6.3.4, Eq (6.11):
            P_{L,M}(rho) = (a_0 + a_1*rho + ... + a_L*rho^L)
                           / (1 + b_1*rho + ... + b_M*rho^M)

        The matching conditions yield a linear system for {a_i, b_j}.

        Parameters
        ----------
        L, M : int  Numerator and denominator degrees; L+M <= N.
        """
        assert L + M <= self.N, (
            f"L+M={L+M} exceeds available expansion order N={self.N}. "
            "Compute more Taylor coefficients."
        )

        C = self.coefficients

        # Build the Hankel-like system for denominator coefficients b_1..b_M
        # From: C_{L+1} + b_1*C_L + ... + b_M*C_{L+1-M} = 0, etc.
        if M > 0:
            # System: H @ b_vec = -rhs_vec (Berlekamp-Massey style)
            rows = []
            rhs  = []
            for i in range(M):
                row = []
                for j in range(M):
                    idx = L + i - j
                    row.append(C[idx] if 0 <= idx <= self.N else 0.0)
                rows.append(row)
                rhs.append(-C[L + i + 1] if L + i + 1 <= self.N else 0.0)

            H   = np.array(rows, dtype=float)
            rhs = np.array(rhs, dtype=float)
            try:
                b_vec = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                b_vec = np.linalg.lstsq(H, rhs, rcond=None)[0]
            b_coeffs = np.concatenate([[1.0], b_vec])   # b_0=1, b_1..b_M
        else:
            b_coeffs = np.array([1.0])

        # Numerator: a_k = sum_{j=0}^{min(k,M)} b_j * C_{k-j}
        a_coeffs = np.zeros(L + 1)
        for k in range(L + 1):
            for j in range(min(k, M) + 1):
                if k - j <= self.N:
                    a_coeffs[k] += b_coeffs[j] * C[k - j]

        self._numerator   = a_coeffs     # ascending powers of rho
        self._denominator = b_coeffs
        self._L = L
        self._M = M
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, rho: float) -> float:
        """Evaluate P_{L,M}(rho).

        Parameters
        ----------
        rho : float  Correlation value in (-1, 1).

        Returns
        -------
        float  Approximated barrier price.
        """
        if self._numerator is None:
            raise RuntimeError("Call build(L, M) or pade_11_explicit() first.")

        num = sum(a * rho**k for k, a in enumerate(self._numerator))
        den = sum(b * rho**k for k, b in enumerate(self._denominator))

        if abs(den) < 1e-12:
            raise ValueError(
                f"Padé denominator near zero at rho={rho} (den={den:.2e}). "
                "A pole is very close to this point — use Taylor truncation instead."
            )
        return float(num / den)

    def pole_distance(self, rho_target: float) -> float:
        """Distance from rho_target to the nearest real pole of the Padé.

        Paper: Section 5.8 — 'one monitors pole proximity and falls back
        to Taylor truncation when the pole is too close'.
        """
        if self._denominator is None or len(self._denominator) <= 1:
            return np.inf

        # Find roots of denominator polynomial
        b = self._denominator
        if len(b) == 2:           # [1/1]: single pole at -b[0]/b[1]
            poles = np.array([-b[0] / b[1]])
        else:
            # Companion matrix eigenvalues
            # poly with coefficients in ascending order -> numpy descending
            poles = np.roots(b[::-1])

        real_poles = poles[np.isreal(poles)].real
        if len(real_poles) == 0:
            return np.inf
        return float(np.min(np.abs(real_poles - rho_target)))

    def is_safe(self, rho_target: float, threshold: float = 0.1) -> bool:
        """Return True if no real pole lies within `threshold` of rho_target.

        Paper: Section 5.8 — pole safety diagnostic before using Padé value.
        """
        return self.pole_distance(rho_target) > threshold

    def evaluate_safe(
        self,
        rho: float,
        taylor_fallback: bool = True,
        threshold: float = 0.1,
    ) -> tuple[float, str]:
        """Evaluate Padé, falling back to Taylor if a pole is too close.

        Returns
        -------
        price  : float  Approximated price
        method : str    'pade' or 'taylor_<order>'
        """
        if self.is_safe(rho, threshold):
            return self.evaluate(rho), f"pade_{self._L}_{self._M}"
        elif taylor_fallback:
            # Fall back to Taylor truncation at order N
            price = sum(c * rho**n for n, c in enumerate(self.coefficients))
            return float(price), f"taylor_{self.N}"
        else:
            raise ValueError(
                f"Padé pole too close to rho={rho} (distance={self.pole_distance(rho):.4f}). "
                "Set taylor_fallback=True or choose a different [L/M]."
            )

    def error_diagnostic(self, rho: float) -> float:
        """Estimate truncation error magnitude from the last coefficient.

        Paper: Eq (5.25): ||e^(N)(t, ·)||_inf ≲ |rho|^{N+1} * ||L1 u_N||_inf
        Here we use |C_N * rho^N| as a proxy for the last term magnitude.
        """
        N = len(self.coefficients) - 1
        return abs(self.coefficients[N]) * abs(rho)**N

    def __repr__(self) -> str:
        L = self._L if self._L is not None else "?"
        M = self._M if self._M is not None else "?"
        return f"PadeAccelerator([{L}/{M}], N={self.N})"
