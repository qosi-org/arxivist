# Q-Ising: Dynamic Treatment on Networks

**Paper**: [Dynamic Treatment on Networks](https://arxiv.org/abs/2605.06564)  
**Authors**: Bengusu Nar, Jiguang Li, Veronika Ročková, Panos Toulis  
**Institution**: Booth School of Business, University of Chicago  
**arXiv**: 2605.06564 (May 2026)

---

## What This Paper Does

Q-Ising addresses the problem of dynamic treatment allocation on social networks from a single observational panel. Existing methods either handle network interference statically or handle dynamic decisions without network structure. Q-Ising integrates both by running a three-stage pipeline:

1. **Stage 1 (Ising Inference)**: A Bayesian dynamic Ising model estimates how adoption spreads through the network, producing counterfactual "no-treatment" adoption probabilities as latent states.
2. **Stage 2 (Offline CQL)**: Conservative Q-Learning learns a bin-level treatment policy from offline transitions, avoiding overestimation of unsupported actions.
3. **Stage 3 (Ensemble Policy)**: MCMC posterior draws generate an ensemble of policies via majority vote, providing interpretable uncertainty quantification over treatment decisions.

The method comes with a finite-sample regret bound decomposing into standard offline-RL error, network abstraction error, and Ising estimation error.

---

## Quick Start

```bash
# 1. Install
git clone <repo>
cd q_ising
pip install -e .

# 2. Run SBM experiment (Section 5.1)
python train.py --config configs/sbm_default.yaml

# 3. Run with MCMC ensemble
python train.py --config configs/sbm_default.yaml --ensemble

# 4. Debug mode (fast check)
python train.py --config configs/sbm_default.yaml --debug
```

---

## Installation

**pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**conda:**
```bash
conda env create -f environment.yaml
conda activate q-ising
pip install -e .
```

**Docker:**
```bash
docker-compose -f docker/docker-compose.yml up train
```

---

## Village Experiment Data

The Indian microfinance village experiment (Section 5.2) uses data from:

> Banerjee, Chandrasekhar, Duflo, Jackson (2013). *The Diffusion of Microfinance*. Science 341(6144).

```bash
python data/download_villages.py   # Follow the printed instructions
```

Data is publicly available at Harvard Dataverse. Place adjacency matrices at `data/villages/village_{i}_adjacency.npy`.

---

## Expected Results

### SBM Experiment (Section 5.1, Figure 1 left)

| Policy | Mean Adoption Rate |
|--------|--------------------|
| Random | ~0.08 |
| Degree | ~0.07 |
| LIR | ~0.07 |
| Degree-Bin | ~0.09 |
| Plain DQN | ~0.15 |
| **Q-Ising** | **~0.20** |

*(approximate; exact values depend on random seed and SBM realization)*

### Microfinance Villages (Section 5.2, Table 2)

Q-Ising matches or improves over Degree-Bin in 30+ of 42 villages. Improvement is negatively correlated with village modularity (correlation ≈ −0.5).

---

## Implementation Assumptions & Reproducibility Notes

The following assumptions were made where the paper is underspecified:

| Component | Assumption | Confidence | Notes |
|-----------|-----------|-----------|-------|
| EMVS solver | L1-penalized logistic regression (sklearn) | 0.62 | Paper describes EM approach; exact updates follow Ročková & George (2014) |
| HMC library | PyMC with NUTS sampler | 0.65 | Paper cites Hoffman & Gelman (2014); could also be NumPyro/Stan |
| Q-network activation | ReLU | 0.80 | Standard for CQL/d3rlpy; not stated in paper |
| SBM bin assignment | Spectral clustering | 0.78 | Paper uses 4-block SBM; ground-truth blocks used as bins |
| LIR score formula | degree(i) / mean_neighbor_degree(i) | ~0.60 | Proxy for Liu et al. (2017); verify against original |

**Known deviations from paper:**
- The EMVS implementation uses a logistic regression proxy rather than the full Ročková & George (2014) EM algorithm. For a faithful EMVS implementation, see the original R package `EMVS`.
- The MCMC ensemble uses PyMC's continuous Bernoulli relaxation for the spike-and-slab prior, which approximates but does not exactly replicate discrete spike-and-slab sampling.

---

## Project Structure

```
q_ising/
├── train.py                     Main training entrypoint
├── evaluate.py                  Evaluation entrypoint
├── configs/
│   ├── sbm_default.yaml         SBM experiment (Section 5.1)
│   └── village_default.yaml     Village experiment (Section 5.2)
├── src/q_ising/
│   ├── data/
│   │   ├── network.py           NetworkData container
│   │   ├── panel.py             ObservationalPanel + Transition
│   │   └── sis_simulator.py     SIS dynamics simulator
│   ├── models/
│   │   ├── ising.py             DynamicIsingModel (EMVS + MCMC)
│   │   └── state_constructor.py Q-Ising state construction
│   ├── training/
│   │   ├── cql_trainer.py       CQL via d3rlpy
│   │   └── ensemble_trainer.py  Ensemble majority-vote policy
│   ├── evaluation/
│   │   ├── baselines.py         5 reference policies
│   │   └── metrics.py           PolicyEvaluator
│   └── utils/
│       ├── config.py            Typed config + seeding
│       ├── community_detection.py igraph edge-betweenness
│       └── sbm_generator.py     SBM graph generator
└── data/
    └── download_villages.py     Village data download instructions
```

---

## Citation

```bibtex
@article{nar2026dynamic,
  title={Dynamic Treatment on Networks},
  author={Nar, Bengusu and Li, Jiguang and Ro\v{c}kov\'{a}, Veronika and Toulis, Panos},
  journal={arXiv preprint arXiv:2605.06564},
  year={2026}
}
```

---

*Generated by [ArXivist](https://github.com/anthropics/arxivist) — Stage 4 Code Generator*
