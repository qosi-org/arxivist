"""
STUB: General matrix-valued case (Section 5.7), eq. (5.50)-(5.51).

The paper explicitly states: "Wellposedness for fully coupled FBSDEs of this type is,
to the best of our knowledge, an open problem ... We leave both the wellposedness of
(5.50)-(5.51) and the verification of MDC in the general matrix-valued setting to
future work."

SIR ambiguity (see sir-registry/arxiv_2607_10934/sir.json -> ambiguities[0]):
no known algorithm is proven to solve this system. The class below is an UNVERIFIED
Picard-iteration heuristic provided only for exploratory use. It must never be treated
as a validated reproduction of a paper result, and it is excluded from the default
verification suite (run_verification.py only invokes it under --case general_fbsde_stub).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class GeneralMatrixFBSDEHeuristic:
    """
    STUB: heuristic, unverified Picard-iteration approximation to (5.50)-(5.51).

    This does NOT reproduce a paper result. It exists only so downstream users have a
    concrete (if unreliable) starting point for exploring the open problem, per
    ArXivist's hallucination-prevention rule: genuinely unknown components must be
    implemented as clearly-labeled stubs, never silently guessed.
    """

    n_assets: int

    def solve_heuristic(
        self,
        C: np.ndarray,
        sigma_path_fn: Callable[[float], np.ndarray],
        T: float,
        n_steps: int,
        n_iter: int = 5,
    ) -> dict:
        warnings.warn(
            "GeneralMatrixFBSDEHeuristic.solve_heuristic is an UNVERIFIED stub for an "
            "explicitly open problem in the source paper (Section 5.7). Its output has "
            "no theoretical guarantee of correctness -- see SIR ambiguities[0].",
            UserWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "STUB: General matrix-valued FBSDE (5.50)-(5.51) has no known wellposedness "
            "result in the source paper. Replace this stub with a validated numerical "
            "FBSDE scheme before relying on it for anything beyond exploration."
        )
