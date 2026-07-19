"""
kyle_liquidity — a numerical reproduction/verification package for:

    "Multidimensional stochastic liquidity in Kyle's model of informed trading"
    Ekren, Nikitopoulos, Vy (arXiv:2607.10934)

This package does NOT train a neural network — the paper is a stochastic-control /
probability-theory paper. Instead, it numerically simulates the paper's equilibrium
construction (market depth M*_t, price P*_t, posterior covariance Sigma*_t, insider
strategy X*_t) for every benchmark case solved in closed form in Section 5, and checks
the simulated paths against those closed forms.

ArXivist paper_id: arxiv_2607_10934
"""

__version__ = "0.1.0"
