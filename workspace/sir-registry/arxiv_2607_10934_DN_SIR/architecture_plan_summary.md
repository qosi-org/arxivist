# Architecture Plan Summary — arxiv_2607_10934

**Paper**: Multidimensional stochastic liquidity in Kyle's model of informed trading (Ekren, Nikitopoulos, Vy)

## Framework
NumPy/SciPy scientific computing, not a neural-network framework. PyTorch is nominally
"primary" (schema requires pytorch/jax/tensorflow) but is only touched by the explicitly-labeled
`fbsde_stub.py` heuristic for the open Section 5.7 problem — every other module is pure NumPy.

## What gets built
- `depth.py` — closed-form martingale depth M*_t for the four solved benchmark cases
  (Kyle 1985, Back-Pedersen 1998, Collin-Dufresne-Fos 2016, common-eigenbasis/BCEL 2020)
- `filtering.py` — Kalman-Bucy simulator producing price P*_t and posterior covariance Sigma*_t
- `strategy.py` — insider's equilibrium trading strategy dX*_t
- `doob_meyer.py` — numerical illustration of the matrix Doob-Meyer decomposition
- `verification.py` — runs every benchmark case, checks it against its closed form, reports
  an empirical (non-rigorous) MDC health check via min-eigenvalue tracking
- `fbsde_stub.py` — clearly labeled STUB for the open general matrix case (Section 5.7)

## Entrypoints
`run_verification.py` (replaces train.py — there is nothing to train), `compare_to_paper.py`
(replaces evaluate.py), `inference.py` (single-path demo).

## Top risks
1. Section 5.7 general matrix case: no wellposedness theory exists — implemented only as an
   unverified, opt-in heuristic stub.
2. MDC has no general numerical certificate — only an empirical min-eigenvalue health check.
3. No numerical BSDE solver is specified for Section 5.5 (scalar stochastic liquidity) — out
   of scope for the default suite, documented as future work.

Full detail in `architecture_plan.json`.
