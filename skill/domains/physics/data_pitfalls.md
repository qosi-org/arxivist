# Domain: Physics — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical reproducibility traps in Physics papers

**Reference solution provenance** — the "ground truth" in PDE papers is usually a
high-resolution numerical solution, not an analytical one. If the reference solver,
its resolution, and its parameters are not specified, the ground truth itself is not
reproducible. Flag if the reference solution generation procedure is undescribed.

**Simulation initial conditions** — deterministic simulations are fully determined by
their initial conditions. If the paper does not specify the initial condition exactly
(e.g. "random turbulent initialisation" without a seed), results cannot be reproduced
exactly. Flag and add to ambiguities.

**Dataset generation vs pre-generated datasets** — some papers train on datasets they
generated via simulation; others use pre-generated datasets. If the paper generated its
own training data, the data generation code is as important as the model code. Extract
the data generation procedure in detail.

**Units and non-dimensionalisation** — papers often work in non-dimensional units without
stating the reference scales. Results reported in one unit system cannot be compared to
results in another. Flag any quantity whose units are ambiguous.

**Hardware-dependent floating point** — GPU and CPU floating point operations produce
slightly different results due to non-associative floating point arithmetic and different
CUDA kernel implementations. Long simulations can amplify these differences. Note if
the paper specifies hardware.

**Periodic vs open boundary conditions** — for the same PDE and the same model, periodic
boundaries produce different solutions than open boundaries. This is almost never ambiguous
in the text but is sometimes mislabelled in code. Always verify BC implementation against
the paper's domain description.

**Thermodynamic ensemble** — molecular dynamics results depend on whether the simulation
runs in NVE (microcanonical), NVT (canonical), or NPT (isothermal-isobaric) ensemble.
Different ensembles give different thermodynamic averages for the same system.
