# Quantum CVA — Reproduction of arXiv:2607.12990

Reproduction of **"A Noise-Aware Quantum Algorithm for Credit Valuation
Adjustments on Real Quantum Hardware"** (Borràs Espert, Gómez Casanova, de
Pedro Sánchez, Hernández Santana, Serrano Molinero; BBVA Quantum /
Universitat de València / Universidad Autónoma de Madrid / Basque Quantum,
July 2026).

The paper develops an end-to-end, noise-aware quantum workflow for Credit
Valuation Adjustment (CVA): market calibration → finite-grid discretisation →
variational quantum encoding (QCBM + controlled-rotation ansätze) → amplitude
estimation, and introduces **CABIQAE** (Contrast-Aware Bayesian Iterative
Quantum Amplitude Estimation), which folds hardware-calibrated Grover-contrast
decay directly into the Bayesian likelihood and depth-scheduling logic.

## What's implemented

| Paper section | Module |
|---|---|
| 2.1 (classical CVA, GBM, Black-Scholes, CDS bootstrap) | `src/quantum_cva/finance/classical_cva.py` |
| 2.2.1 (quantum encoding, finite grid) | `src/quantum_cva/finance/grid_encoding.py`, `src/quantum_cva/circuits/cva_oracle.py` |
| 2.2.2, Appendix A.3 (**CABIQAE**) | `src/quantum_cva/estimation/cabiqae.py` |
| 3.2.2 (hardware calibration, contrast model) | `src/quantum_cva/estimation/contrast_calibration.py` |
| 3.2.3 (QCBM, CRCA ansätze) | `src/quantum_cva/circuits/state_preparation.py`, `controlled_rotations.py` |
| Appendix A.2 (hardware execution) | `src/quantum_cva/hardware/backend_manager.py` |
| 4.5 (error-budget decomposition) | `src/quantum_cva/evaluation/error_decomposition.py` |

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# 1. Train QCBM + CRCA circuit blocks against the finite-grid CVA benchmark
python train.py --config configs/config.yaml --output-dir checkpoints/

# 2. Run one CABIQAE estimation trajectory
python inference.py --config configs/config.yaml --checkpoint-dir checkpoints/ --epsilon 0.01

# 3. Full CABIQAE vs BIQAE vs BAE vs DCS comparison (noiseless or hardware-replay)
python evaluate.py --config configs/config.yaml --regime noiseless --num-trajectories 300

# 4. (Optional) Fit the Grover-contrast decay model from a hardware/simulated sweep
python run_hardware_calibration.py --config configs/config.yaml --circuit validation
```

Or via Docker:
```bash
docker compose -f docker/docker-compose.yml up --build
```

## Repository layout

```
configs/config.yaml            All hyperparameters (# ASSUMED comments flag unstated values)
src/quantum_cva/
  circuits/                    QCBM state prep, CRCA blocks, CVA oracle + Grover iterate
  estimation/                  CABIQAE, BIQAE, BAE, DCS, hardware contrast calibration
  finance/                     Classical CVA benchmark, finite-grid construction
  hardware/                    IBM Quantum / Q-CTRL execution wrapper
  training/                    Variational parameter trainer (parameter-shift + Adam)
  evaluation/                  Error-budget decomposition, trajectory metrics
  utils/                       Config loading, global seeding
train.py / evaluate.py / inference.py / run_hardware_calibration.py   Entrypoints
tests/                         34 unit tests covering every module
data/                          LSEG data requirements + synthetic-fallback docs
docker/                        Dockerfile + docker-compose.yml
```

## Known issue (found via an actual full-scale training run)

Running `train.py` end-to-end at the real six-qubit paper scale and then
computing both `CVA_cont_MC` and `CVA_tab_Delta` surfaced a real gap in
**synthetic-fallback mode**: `train.py::build_finite_grid` hardcodes a flat
`default_incr_vec = 2e-4` for the quantum-circuit target, but there's no
single shared, CDS-calibrated survival curve feeding *both* the finite-grid
benchmark and a continuous Monte Carlo benchmark. That makes the two
benchmarks' default-probability assumptions inconsistent with each other in
synthetic mode, so **Section 4.5's error-budget percentages should not be
trusted until this is fixed** (either by wiring a shared survival curve into
`build_finite_grid`, or by using real LSEG data via `data/download.sh`,
which sidesteps the issue entirely since both quantities are then built from
the one real calibrated curve).

This does **not** affect the CABIQAE/BIQAE/DCS amplitude-estimation
comparison (`evaluate.py`), which operates directly on the trained circuit's
amplitude and is independent of this gap. A full training + evaluation run
(400 SPSA steps on `R_v`, 30 trajectories per regime) reproduced the paper's
core algorithmic finding: CABIQAE substantially outperforms noise-naive
BIQAE once Grover contrast decays (~30% vs ~62% median relative error in a
hardware-replay regime), matching Table 6's qualitative pattern.

## Reproducibility notes (read before trusting exact numbers)

This paper is a **quantum-computing** paper, not a classical deep-learning
paper, so some of this scaffold's usual assumptions (batch size, mixed
precision, GPU) don't apply — see `configs/config.yaml` for what's fixed vs.
`# ASSUMED`. The most consequential gaps between the paper text and this
implementation, in order of impact:

1. **Optimizer / learning rate / iteration counts are not named in the
   paper** (Section 3.2.3 gives only the loss functions). This repo defaults
   to Adam with parameter-shift gradients (exact, since every gate here is a
   Pauli rotation) — see `training/trainer.py`. Iteration counts are read
   off the paper's own Figure 8/9 axes, not stated numerically.
2. **Q-CTRL's Performance Management is a black box.** The paper explicitly
   states this managed service's internal transpilation/error-suppression is
   not observable. `hardware/backend_manager.py::run_with_qctrl` is a thin,
   honest wrapper: with real credentials it calls the actual service; without
   them it raises `HardwareUnavailableError` pointing you to
   `ContrastCalibrator.build_hardware_replay_model`, which reproduces the
   paper's **published** calibration statistics (Tables 17–18) instead of
   fabricating a fake hardware response.
3. **Market data is proprietary (LSEG Workspace).** See `data/README_data.md`
   — `train.py` runs out-of-the-box using representative synthetic curves;
   swap in real LSEG pulls via `data/download.sh` for a fully faithful
   market calibration.
4. **`R_v`'s snake-topology repetition rule** for L=2 is inferred from the
   paper's L=1 diagram (Figure 5) plus its stated "compose several
   repetitions" rule — validated in `tests/test_circuits.py` by checking the
   resulting parameter-count scaling.

Full detail, confidence scores per section, and all logged ambiguities are in
`sir-registry/arxiv_2607_12990/sir.json` (Stage 1) and
`architecture_plan.json` (Stage 3, `risk_assessment` field).

## Testing

```bash
pytest tests/ -v          # 34 tests, all passing
ruff check src/ *.py       # clean
```

## Citation

```
Borràs Espert, G., Gómez Casanova, F., de Pedro Sánchez, L., Hernández Santana, S.,
& Serrano Molinero, P. (2026). A Noise-Aware Quantum Algorithm for Credit Valuation
Adjustments on Real Quantum Hardware. arXiv:2607.12990.
```

This is an independent reproduction generated by the ArXivist pipeline; it is
not affiliated with the paper's authors or BBVA. See `data/README_data.md`
for data-licensing caveats.
