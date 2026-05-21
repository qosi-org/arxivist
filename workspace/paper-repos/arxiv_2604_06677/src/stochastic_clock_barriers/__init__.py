"""
stochastic_clock_barriers
=========================
Semi-analytic pricing library for barrier options under stochastic-clock
volatility models, with leverage corrections via rho-expansion and Padé
resummation.

Paper: "Extrema, Barrier Options, and Semi-Analytic Leverage Corrections
       in Stochastic-Clock Volatility Models" — Tristan Guillaume (2026)
arXiv: 2605.06677v1
"""

from .models import CIRClock, SquaredOUClock
from .pricing import SingleBarrierPricer, DoubleBarrierPricer, LeverageExpansion
from .monte_carlo import MonteCarloPricer

__version__ = "1.0.0"
__all__ = [
    "CIRClock",
    "SquaredOUClock",
    "SingleBarrierPricer",
    "DoubleBarrierPricer",
    "LeverageExpansion",
    "MonteCarloPricer",
]
