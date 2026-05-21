"""Rotation angle logic for Quantum-SMOTE (Algorithm 2).

Implements the three-branch angular distance to rotation-angle mapping
described in the architecture plan and SIR.
"""
from __future__ import annotations

import math


class AngleCalculator:
    """Map angular distance to a rotation angle.

    The logic follows the paper's Algorithm 2:
    - if angular_distance > pi/2: use |(pi/2 - angular_distance) / split_factor|
    - if angular_distance < 0: use |((pi/2 - angular_distance) * Uniform(0.5, 1)) / split_factor|
    - else: use Uniform(0, angular_distance) / split_factor

    After branch selection, angle_increment = 0.0174533 * loop_ctr is added.
    """

    @staticmethod
    def compute(angular_distance: float, split_factor: int, loop_ctr: int) -> float:
        """Compute the final rotation angle.

        Parameters
        ----------
        angular_distance : float
            Output from the compact swap test, in radians.
        split_factor : int
            Hyperparameter controlling the magnitude of the rotation.
        loop_ctr : int
            Synthetic loop index; used to add the per-iteration increment.

        Returns
        -------
        float
            Rotation angle in radians.
        """
        if split_factor <= 0:
            raise ValueError("split_factor must be a positive integer")
        if loop_ctr < 0:
            raise ValueError("loop_ctr must be non-negative")

        half_pi = math.pi / 2.0

        if angular_distance > half_pi:
            angle = abs((half_pi - angular_distance) / float(split_factor))
        elif angular_distance < 0.0:
            # Algorithm 2 explicitly references Uniform(0.5, 1)
            # We use a deterministic midpoint for reproducibility only if a
            # caller needs a pure function; however, to preserve the paper's
            # stochastic branch, the midpoint is sampled via the stdlib RNG.
            import random

            u = random.uniform(0.5, 1.0)
            angle = abs(((half_pi - angular_distance) * u) / float(split_factor))
        else:
            import random

            u = random.uniform(0.0, float(angular_distance))
            angle = u / float(split_factor)

        angle_increment = 0.0174533 * float(loop_ctr)
        return float(angle + angle_increment)
