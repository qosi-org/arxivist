# Notebooks

## `reproduction_walkthrough.ipynb`

A guided, small-scale walkthrough of the paper's pipeline:

1. Run one baseline stock (demo scale: 60k steps vs. the paper's 1M)
2. Reconstruct metaorders and fit the impact curve (Eq. 1-3), plot Figure-3-style
3. Compare the measured δ against the GGPS theory prediction (Eq. 4)
4. A mini baseline-vs-no-splitting ablation (Table 1's headline comparison)

Runs in a few minutes on a laptop CPU. For the paper's actual scale (2000
stocks, 1e6 steps each), use `train.py` / `run_counterfactual_suite.py`
from the repo root instead — see the main `README.md`.

**Verified**: this notebook was executed end-to-end (with reduced step
counts) as part of generating this repository, with zero runtime errors.
Its as-shipped fitted δ will not closely match the paper's numbers — see
the main README's *Reproducibility Notes* for why, and which config
parameters to tune first.
