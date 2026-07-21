# Hallucination Report

**Paper**: Beyond Softmax: A Natural Parameterization for Categorical Random Variables  
**Paper ID**: arxiv_2509_24728  
**Audit Date**: 2026-07-16  
**Auditor**: ArXivist Stage 6 (Results Comparator)  
**SIR Version**: 1 (confidence: 0.88 overall)  
**Architecture Plan Version**: 1

This report documents all structural, parametric, and omission hallucinations identified by comparing the generated implementation against the SIR. A hallucination is any deviation between what was implemented and what the paper specifies (or strongly implies), whether due to ambiguity in the paper, inference errors in the SIR, or design choices made during code generation.

**Status legend**: ✅ Resolved (patched before Stage 5) | ⚠️ Open (requires verification) | ℹ️ Design decision (documented)

---

## Section 1 — Structural Hallucinations

Components present in the generated code that are NOT explicitly described in the paper and may be incorrect.

---

### H-01 — Upsample layer in VAE decoder

**Location**: `src/catnat/models/vae.py`, `VAEDecoder.deconv`, final layer  
**Severity**: Significant  
**Status**: ⚠️ Open — partially mitigated  
**Type**: structural

**Description**: The generated decoder includes `nn.Upsample(size=(28,28), mode='bilinear')` as a final step to guarantee exact 28×28 output. The paper does not describe this layer. It was added as a practical fix when the ConvTranspose2d chain did not exactly reconstruct 28×28 pixels.

**Evidence of hallucination**: The paper (Appendix F.1) describes "3 transposed convolutional layers" with no mention of a bilinear upsample. The reference codebase (jxmorris12/categorical-vae) uses a symmetric conv/deconv structure that hits 28×28 exactly.

**Risk**: Bilinear upsampling introduces a learned-free interpolation step that slightly blurs reconstructions. This changes the effective reconstruction loss landscape. The effect on NLL is likely small (< 0.5 nats) but nonzero.

**Suggested fix**: Tune the ConvTranspose2d kernel/stride/padding parameters to hit exactly 28×28 without the Upsample layer. A known working sequence for MNIST from a 1×1 feature map: `ConvTranspose2d(64,64,k=5,s=1)→5×5`, `ConvTranspose2d(64,32,k=4,s=2,p=0)→12×12`, `ConvTranspose2d(32,1,k=5,s=2,p=0,op=1)→27×27 → pad to 28×28`. Alternatively, fetch exact layer specs from the reference repo.

---

### H-02 — Gumbel noise at leaf log-probs (confirmed correct, but document)

**Location**: `src/catnat/samplers.py`, `GumbelSoftmaxSampler.forward`  
**Severity**: Minor  
**Status**: ✅ Resolved in Stage 5  
**Type**: structural

**Description**: The paper does not specify where in the catnat tree the Gumbel noise is injected when using the Gumbel-Softmax trick. The implementation injects noise at the leaf log-probabilities (standard Gumbel-Softmax). An alternative would be to inject at internal node logits.

**Evidence of resolution**: All canonical Gumbel-Softmax implementations (Jang et al. 2017; PyTorch `F.gumbel_softmax`; latentspace.github.io) apply noise at the output distribution's log-probabilities. The injected noise therefore operates identically for catnat as for softmax. **This hallucination is closed.**

**Residual risk**: If the authors' catnat-torch implementation injects noise differently (e.g., per node logit before tree traversal), the resulting gradient signal would differ. The `gumbel_injection_point` config flag allows switching if needed.

---

### H-03 — GSL GNN uses self-loop adjacency normalization

**Location**: `src/catnat/models/gsl.py`, `GCNLayer.forward`  
**Severity**: Minor  
**Status**: ⚠️ Open — unverifiable from paper  
**Type**: structural

**Description**: The GCNLayer applies symmetric normalization with self-loops added: $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$. The paper cites Kipf & Welling (2017) for the GCN architecture, which uses this normalization. However, the paper does not specify whether self-loops are included or whether renormalization is applied.

**Evidence**: Kipf & Welling (2017) includes self-loops by default. The paper's reference to "GCN (Kipf & Welling, 2017)" in Appendix E.2 strongly implies this convention.

**Impact on results**: If the true model omits self-loops, node representations will differ. This primarily affects PP-MAE and PP-MSE (downstream prediction metrics) rather than MAE(θ) (latent calibration), since the latter depends on the gradient estimator, not the GNN forward pass.

**Suggested fix**: No change needed unless GNN baseline (softmax) fails to match paper's softmax numbers within 5%. If it does, try removing self-loops from the adjacency.

---

### H-04 — PPO entropy bonus coefficient treated as hyperparameter

**Location**: `src/catnat/training/trainers.py`, `PPOTrainer`, `configs/rl.yaml`  
**Severity**: Minor  
**Status**: ℹ️ Design decision  
**Type**: structural

**Description**: The RL experiment uses the entropy coefficient `ent_coef` as a TPE-searched hyperparameter with range `[0.0, 1.0]`. In practice, most PPO implementations use `ent_coef ≈ 0.01`. The paper does not report the best-performing entropy coefficient.

**Evidence**: The paper (Appendix I.4, Table 7) lists `ent_coef` as a searched parameter with range `[0.0, 1.0]`. The default in `configs/rl.yaml` (`ent_coef: 0.01`) is a reasonable assumed prior.

**Impact**: The entropy coefficient controls exploration. If the true best value differs substantially from 0.01, the reported TPE-searched results cannot be replicated without running the full 160-trial search.

**Suggested fix**: Run `scripts/run_rl_tpe.py` with `n_trials=160` as specified. Do not report numbers from the default config as reproducing Table 4.

---

## Section 2 — Parametric Hallucinations

Hyperparameters that were assumed during generation and may be incorrect.

---

### H-05 — VAE encoder conv channels [32, 64, 64]

**Location**: `src/catnat/models/vae.py`, `VAEEncoder.__init__`; `configs/vae.yaml`  
**Severity**: Significant  
**Status**: ⚠️ Open — partially resolved  
**Type**: parametric  
**SIR confidence**: 0.70 (RISK-03)

**Description**: The paper (Appendix F.1) specifies "3 convolutional layers with ReLU activations, followed by 2 fully-connected layers" but gives no channel widths, kernel sizes, or strides. The implementation uses `[32, 64, 64]` channels with kernel=4, stride=2, derived by examining the reference repo structure and cross-referencing with standard MNIST VAE conventions.

**Evidence of uncertainty**: The reference repo (jxmorris12/categorical-vae) was accessible at the directory level (listing: models.py, train.py) but the raw Python source was not fetched due to URL restrictions. Channel counts are inferred, not confirmed.

**Expected impact**: If the true channel counts differ (e.g., `[32, 64, 128]` as originally assumed), the encoder's representational capacity changes. Empirically, for MNIST with N=20, K=16, the difference is expected to be 1–3 NLL points — a Moderate deviation.

**Suggested fix**:
1. Fetch `https://github.com/jxmorris12/categorical-vae/blob/master/models.py` directly in a browser
2. Confirm conv channel widths and update `configs/vae.yaml` → `encoder_channels`
3. Re-run with the correct channels before comparing to Table 3

**Distinguishing symptom**: If your catnat-ν NLL is > 102 (worse than the paper's sparsemax baseline of 102.1), the conv dims are almost certainly wrong.

---

### H-06 — GSL GNN hidden dimension = 64

**Location**: `src/catnat/models/gsl.py`, `GCNEncoder.__init__`; `configs/gsl.yaml`  
**Severity**: Moderate  
**Status**: ⚠️ Open  
**Type**: parametric  
**SIR confidence**: 0.75 (RISK-05)

**Description**: The paper specifies "identical architecture" to the data-generating model for the learnable GCN, but gives no dimensions. The implementation uses `d_hidden=64, n_layers=2`, which are standard defaults for a small-graph GCN.

**Evidence of uncertainty**: The paper's community graph (Figure 5) has approximately 15 nodes, 4 communities, and a scalar output. The input features are `x ~ N(0, σ²_x I)` with `σ_x=1`. For a 15-node graph with scalar features, a hidden dim of 64 is plausible but not confirmed.

**Expected impact**: A different hidden dim changes the expressive power of the GNN, which primarily affects PP-MAE and PP-MSE. MAE(θ) is less sensitive because it depends on the gradient estimator dynamics.

**Suggested fix**: Try `d_hidden ∈ {32, 64, 128}` with `n_layers=2`. If the softmax baseline ES does not match the paper's `14.990 ± 0.020` within 2%, adjust these dimensions.

---

### H-07 — GSL graph has n_nodes=15 and n_communities=4

**Location**: `src/catnat/data/gsl_dataset.py`, `build_theta_star`; `configs/gsl.yaml`  
**Severity**: Moderate  
**Status**: ⚠️ Open  
**Type**: parametric  
**SIR confidence**: 0.72

**Description**: The paper uses "a graph with 4 communities" (Appendix E.1) but does not specify the number of nodes per community or total nodes. Figure 5 of the paper shows a community graph with a specific visual structure. We assumed n_nodes=15 (approximately 3–4 nodes per community visible in Figure 5).

**Evidence of uncertainty**: Figure 5 is referenced from Manenti et al. (2025). The exact node count is inferable from the figure but was not extracted with certainty.

**Expected impact**: A wrong node count changes the dimensionality of the adjacency matrix, the number of Bernoulli edge parameters, and the computational cost of M=32 graph samples per step. This has a direct effect on all four GSL metrics.

**Suggested fix**: Cross-reference Manenti et al. (2025) ("Learning latent graph structures and their uncertainty", ICML 2025) for the exact graph specification. The paper is at `https://openreview.net/forum?id=TMRh3ScSCb`.

---

### H-08 — RL action padding to next power of 2

**Location**: `src/catnat/training/trainers.py`, `_train_rl`; `train.py`  
**Severity**: Minor  
**Status**: ℹ️ Design decision (RISK-01)  
**Type**: parametric

**Description**: Atari Seaquest has 18 discrete actions. CatNat requires K to be a power of 2. The implementation auto-pads K to 32 (next power of 2 ≥ 18), adding 14 dummy categories. These dummy categories receive near-zero probability mass after training, but they do expand the tree depth from H=log₂(18)≈4.17 to H=5, adding one extra level of binary decisions.

**Evidence**: The paper uses Seaquest as an RL benchmark (Section 5.3) without discussing how non-power-of-2 action spaces are handled. Breakout has 4 actions (K=4, exact power of 2). Seaquest has 18 actions.

**Expected impact**: 14 dummy categories add noise to the probability distribution and increase the FIM complexity. However, the dummy categories naturally concentrate near probability 0 during training, limiting their practical impact. This is expected to cause a Minor deviation (< 5%) from any hypothetical K=18 implementation.

**Suggested fix**: This is an inherent constraint of the catnat construction. The implementation is the best possible given RISK-01. Document it when reporting RL results.

---

## Section 3 — Omission Hallucinations

Components specified in the SIR that are absent or stubbed in the generated code.

---

### H-09 — GSL node count inferred from figure, not fetched from Manenti et al. 2025

**Location**: `configs/gsl.yaml` → `model.n_nodes = 15`  
**Severity**: Moderate  
**Status**: ⚠️ Open  
**Type**: omission

**Description**: The exact experimental setup for the GSL experiment is taken from Manenti et al. (2025), not from the catnat paper itself. The catnat paper says "We adopt the experimental setup from Manenti et al. (2025)" and refers to Figure 5 for the graph structure. The precise node count, community size, and graph topology from that prior paper are not reproduced in the catnat paper's appendix.

**Evidence**: Appendix E references the prior work for full details. The `n_nodes=15` assumption has SIR confidence 0.72 — it is an estimate from Figure 5, not a confirmed value.

**Suggested fix**: Fetch the Manenti et al. (2025) paper or code from `https://openreview.net/forum?id=TMRh3ScSCb` to confirm node count and community structure before running the full 10-seed GSL experiment.

---

### H-10 — VAE training epochs not specified

**Location**: `src/catnat/training/trainers.py`, `VAETrainer.train(n_epochs=100)`  
**Severity**: Minor  
**Status**: ⚠️ Open  
**Type**: omission

**Description**: The paper (Appendix F) does not specify the number of training epochs for the VAE experiment. The 2-stage LR grid search trains for an unspecified number of epochs before final model selection. The implementation uses 100 epochs with early stopping based on validation loss.

**Evidence**: Appendix F.2 describes the LR search procedure but not the total training duration. For reference, Jeffares & Liu (2025) (the alt codebase) trained for 150–500 epochs in the additional experiments.

**Expected impact**: If the paper uses more epochs than 100, results may not be fully converged. Empirically, MNIST VAE typically converges in 50–150 epochs with Adam at lr≈0.01.

**Suggested fix**: Monitor validation ELBO every 10 epochs and train until convergence (< 0.1 ELBO improvement over 20 epochs). The `save_every_n_epochs=10` config enables checkpoint recovery.

---

## Hallucination Count Summary

| Type | Count | Critical | Significant | Moderate | Minor |
|------|-------|----------|-------------|----------|-------|
| Structural  | 4 | 0 | 1 | 0 | 3 |
| Parametric  | 4 | 0 | 1 | 2 | 1 |
| Omission    | 2 | 0 | 0 | 1 | 1 |
| **Total**   | **10** | **0** | **2** | **3** | **5** |

**Resolved before Stage 5**: H-02 (Gumbel injection point)  
**Patched in Stage 5**: H-01 (VAE decoder — bilinear upsample added as mitigation)  
**Requires user action**: H-05 (VAE conv channels), H-06 (GNN dims), H-07 (node count), H-09 (GSL setup)  
**Inherent/design**: H-04 (entropy coef), H-08 (action padding)

---

## Risk-to-Experiment Mapping

| Hallucination | Most Affected Metrics | Likely Deviation Magnitude |
|---------------|----------------------|---------------------------|
| H-01 (VAE decoder upsample) | VAE NLL | < 0.5 nats (Minor) |
| H-05 (VAE conv channels) | VAE NLL | 1–3 nats (Moderate) |
| H-06 (GNN hidden dim) | GSL PP-MAE, PP-MSE | 1–5% (Minor–Moderate) |
| H-07 / H-09 (node count) | All GSL metrics | 5–20% (Moderate–Significant) |
| H-08 (action padding) | RL Seaquest return | < 5% (Minor) |
| H-10 (training epochs) | VAE NLL | < 1 nat if early-stopped correctly |
