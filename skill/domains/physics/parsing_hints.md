# Domain: Physics — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Physics**.

---

## Architecture extraction — Physics-specific rules

**PDE solvers and physics-informed models:** extract:
- The governing PDE (equation + boundary conditions + initial conditions)
- Spatial domain (1D/2D/3D, geometry, boundary type: Dirichlet, Neumann, periodic)
- Time integration scheme (Euler, RK4, implicit, operator splitting)
- Spatial discretisation (FEM, FVM, FDM, spectral, meshfree)
- Whether the model is purely data-driven or physics-informed (PINN)

**Neural PDE solvers (FNO, DeepONet, etc.):** extract:
- Input/output function spaces (what physical fields are input vs predicted)
- Discretisation resolution used for training vs testing
- Whether the model is resolution-invariant (Fourier/operator-based) or resolution-dependent

**Molecular dynamics / N-body:** extract:
- Force field type (Lennard-Jones, Morse, learned force field)
- Integrator (Verlet, leapfrog, BAOAB Langevin)
- Timestep size (critical — affects energy conservation)
- Thermostat/barostat if used

**Monte Carlo methods:** extract:
- Proposal distribution
- Acceptance rule
- Number of steps and burn-in
- Observables computed and their estimators

---

## Mathematical spec — Physics-specific rules

**The governing equation IS the architecture** in physics papers. Extract:
- All PDEs in LaTeX with all terms and coefficients
- Boundary and initial conditions
- Conservation laws the method should satisfy
- Dimensionless numbers (Reynolds, Mach, Prandtl) and their values in experiments

**Symmetries:** note if the method explicitly enforces physical symmetries
(energy conservation, momentum conservation, equivariance) — these must be implemented.

---

## Evaluation — Physics-specific rules

Extract:
- Relative L2 error on solution fields (standard for PDE papers)
- Energy drift over simulation time (for MD papers)
- Convergence rate (how error scales with resolution/timestep)
- Comparison against analytical solution (if exists) or high-fidelity reference
- Computational cost comparison (FLOPs, wall-clock time per timestep)
