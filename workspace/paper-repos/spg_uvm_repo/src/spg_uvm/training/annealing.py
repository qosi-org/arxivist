"""
Sigmoid annealing schedule for SPG-UVM.

Anneals exploration temperature (lambda or gamma) and learning rate
from initial to final value over training epochs using a sigmoid curve.

Reference: Section 4.1.2 of arXiv:2605.06670 ("sigmoid-type decay").
Figure 1 shows the schedule shape; the exact formula is not given
(see ASSUMED comment below).

ASSUMED: We parameterize the schedule as:
    value(l) = v_final + (v_initial - v_final) * sigmoid(steepness * (midpoint - l))

where l is the current epoch (0-indexed), midpoint = total_epochs / 2,
and steepness controls the transition sharpness.
This matches the sigmoidal shape in Figure 1.
"""
from __future__ import annotations

import math


class SigmoidAnnealer:
    """
    Smooth sigmoid decay from an initial value to a final value.

    Used for:
      - Lambda (continuous policy temperature): 1.0 -> 0.01
      - Gamma (bang-bang entropy coefficient):  0.01 -> 0.0
      - Learning rate:                          5e-3 -> 1e-4

    All three share the same sigmoid schedule shape (Section 4.1.2).

    Args:
        v_initial:    Starting value at epoch 0.
        v_final:      Ending value at epoch total_epochs.
        steepness:    Controls transition sharpness.
                      Higher = sharper transition around midpoint.
                      ASSUMED: 0.15 (matches Figure 1 visual).
    """

    def __init__(
        self,
        v_initial: float,
        v_final: float,
        steepness: float = 0.15,
    ) -> None:
        self.v_initial = v_initial
        self.v_final = v_final
        self.steepness = steepness

    def get_value(self, epoch: int, total_epochs: int) -> float:
        """
        Get the annealed value at the current epoch.

        ASSUMED formula (see module docstring):
            value(l) = v_final + (v_initial - v_final) * sigmoid(k * (mid - l))

        Args:
            epoch:        Current epoch index (0-based).
            total_epochs: Total number of epochs for this time step.

        Returns:
            Annealed scalar value.
        """
        if total_epochs <= 1:
            return self.v_final

        midpoint = total_epochs / 2.0
        # sigmoid(k * (mid - l)): starts near 1 at l=0, falls to near 0 at l=total
        sig = 1.0 / (1.0 + math.exp(self.steepness * (epoch - midpoint)))
        return self.v_final + (self.v_initial - self.v_final) * sig

    def get_schedule(self, total_epochs: int) -> list:
        """Return full schedule as a list (useful for plotting / logging)."""
        return [self.get_value(e, total_epochs) for e in range(total_epochs)]

    def __repr__(self) -> str:
        return (
            f"SigmoidAnnealer(v_initial={self.v_initial}, v_final={self.v_final}, "
            f"steepness={self.steepness})"
        )
