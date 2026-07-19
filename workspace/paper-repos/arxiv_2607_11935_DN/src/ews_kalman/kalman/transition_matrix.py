"""
State-transition matrix for the order-3 Taylor-expansion TVP-Kalman filter.

The paper (Section 2.2) names this matrix only as "the Taylor expansion
transition matrix of order 3" without giving numeric entries. This module
implements the standard textbook form: the discretised kinematic/Taylor
chain matrix for a 4-state polynomial trend model
x_k = [beta_k, beta'_k, beta''_k, beta'''_k]^T, i.e. the same structure as a
constant-jerk kinematic model (position/velocity/acceleration/jerk), which is
the standard convention for state-space "local polynomial trend" models in
the time-varying-parameter literature (SIR ambiguity #1, confidence 0.55).
"""

from __future__ import annotations

import math

import numpy as np


def build_taylor_transition_matrix(dt: float, order: int = 3) -> np.ndarray:
    """Build the (order+1)x(order+1) discretised Taylor-expansion transition
    matrix F, such that x_k = F @ x_{k-1} propagates a value and its first
    `order` derivatives forward by one time step dt.

    F[i, j] = dt^(j-i) / (j-i)!  for j >= i, else 0

    For order=3 this gives the standard 4x4 matrix:
        [[1, dt, dt^2/2, dt^3/6],
         [0, 1,  dt,     dt^2/2],
         [0, 0,  1,      dt],
         [0, 0,  0,      1]]

    Args:
        dt: time step (paper uses dt = 1/12 year).
        order: highest derivative tracked (paper's state vector
            [beta, beta', beta'', beta'''] corresponds to order=3).

    Returns:
        (order+1) x (order+1) transition matrix F.
    """
    n = order + 1
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            power = j - i
            F[i, j] = dt**power / math.factorial(power)
    return F
