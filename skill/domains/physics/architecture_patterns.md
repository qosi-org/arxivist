# Domain: Physics — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Physics**.

---

## Framework selection

**JAX** — strongly preferred for physics-informed neural networks and operator learning.
Excellent autodiff, JIT compilation, and vectorisation (vmap) are critical for physics.
**PyTorch** — use if paper explicitly uses it or if the model is a standard neural network
applied to physics data rather than a physics-native architecture.
**FEniCS / FEniCSx** — for FEM-based papers.
**ASE (Atomic Simulation Environment)** — for molecular simulation papers.
**MDAnalysis** — for molecular dynamics analysis.

## Standard physics module patterns

**Physics-Informed Neural Network (PINN):**
```
(x, t) → MLP [tanh activations] → u(x,t)
Loss = MSE_data + λ_pde * MSE_residual + λ_bc * MSE_boundary
```
Residual computed via automatic differentiation of the network output.

**Fourier Neural Operator (FNO):**
```
Input_function [B, N, C] → Lifting [B, N, D] →
  [FNO_layer: FFT → spectral_conv → iFFT + W(v)] × L →
Projection [B, N, C_out]
```
Spectral convolution operates on Fourier modes up to `modes` (a hyperparameter).

**Equivariant GNN for molecules (E(3)-equivariant):**
```
Atoms + Positions → Radial_basis(distances) → Message_passing → Energy/Forces
```
Use e3nn library for equivariant operations. Never implement SO(3) operations from scratch.

## Config schema for physics papers

```yaml
pde:
  equation: "navier_stokes"    # descriptive name
  dimension: 2
  domain: "unit_square"
  bc_type: "periodic"
  nu: 1.0e-3                   # viscosity (or other PDE parameter)

solver:
  spatial_resolution: 64       # grid points per dimension
  timestep: 0.01
  n_steps: 1000
  integrator: rk4

model:
  architecture: fno
  modes: 12                    # Fourier modes
  width: 32
  n_layers: 4
  activation: gelu

evaluation:
  metric: relative_l2          # standard for PDE papers
  reference: "high_res_solver" # what the reference solution is
```
