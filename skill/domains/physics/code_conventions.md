# Domain: Physics — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **Physics**.

---

## Physical units and dimensional consistency

Always implement a units system or at minimum document the unit convention:
```python
# Document units explicitly in config and code
# e.g. "All lengths in metres, times in seconds, energies in eV"
# Never mix unit systems silently
```

## Automatic differentiation for PDE residuals

```python
import jax
import jax.numpy as jnp

def pde_residual(params, model, x, t):
    """Compute PDE residual via autodiff — no finite differences."""
    u = lambda x, t: model.apply(params, jnp.stack([x, t]))
    u_t = jax.grad(u, argnums=1)(x, t)
    u_xx = jax.grad(jax.grad(u, argnums=0), argnums=0)(x, t)
    # e.g. heat equation: u_t - alpha * u_xx = 0
    return u_t - config.pde.alpha * u_xx
```

## Energy conservation checking (MD papers)

```python
def check_energy_conservation(trajectory, tolerance=1e-4):
    """Monitor energy drift over simulation."""
    energies = [compute_total_energy(frame) for frame in trajectory]
    drift = (max(energies) - min(energies)) / abs(energies[0])
    if drift > tolerance:
        logger.warning(f"Energy drift {drift:.2e} exceeds tolerance {tolerance:.2e}")
    return drift
```

## Relative L2 error (standard physics metric)

```python
def relative_l2_error(pred, target):
    """Standard evaluation metric for PDE papers."""
    return torch.norm(pred - target) / torch.norm(target)
```

## Requirements

```
jax>=0.4.20
jaxlib>=0.4.20
flax>=0.7.5          # for JAX neural networks
optax>=0.1.7         # for JAX optimisers
torch>=2.1.0         # if PyTorch-based
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
# Domain-specific:
# ase>=3.22.1         # atomic simulation
# mdanalysis>=2.6.0   # MD analysis
# fenics-dolfinx      # FEM (complex install — document in README)
# e3nn>=0.5.1         # equivariant operations
```

## What Must NOT be done

- Do NOT use finite differences for gradients when autodiff is available
- Do NOT ignore units — undocumented unit conventions are a major source of errors
- Do NOT use float32 for long MD simulations — energy drift accumulates; use float64
- Do NOT hardcode physical constants — use scipy.constants or a config file
