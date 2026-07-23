"""
catnat: Natural Parameterization for Categorical Random Variables.

Reproduces: "Beyond Softmax: A Natural Parameterization for Categorical Random Variables"
Manenti & Alippi, ICML 2026. arXiv: 2509.24728v2.

Primary public API:
    CatNat               — the core parameterization (drop-in softmax replacement)
    NaturalActivation    — ν(x) activation function (Eq. 12)
    SoftmaxParam         — softmax baseline
    SparseMaxParam       — sparsemax baseline
    build_parameterization — factory for all four parameterizations
"""

from .catnat import CatNat, SoftmaxParam, SparseMaxParam, build_parameterization
from .activations import NaturalActivation, SigmoidActivation, build_activation

__all__ = [
    "CatNat",
    "SoftmaxParam",
    "SparseMaxParam",
    "NaturalActivation",
    "SigmoidActivation",
    "build_parameterization",
    "build_activation",
]
