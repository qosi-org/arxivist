# Quantum Horizon — Reproduction of arXiv:2606.14484

Reproduction of **"Quantum Horizon: An evaluation of quantum computing as a
threat to Bitcoin and Ethereum"** (Iosif M. Gershteyn, Jacob A. Alber, 2026).

The paper separates two quantum algorithms public discussion routinely
conflates: Shor's algorithm breaks the elliptic-curve signatures that
authorize spending, while Grover's algorithm does *not* meaningfully
threaten proof-of-work mining. It folds hardware scaling, falling resource
requirements, fault-tolerance readiness, and expert surveys into a bimodal
Monte-Carlo forecast for when a cryptographically-relevant quantum computer
(CRQC) will arrive, quantifies exposed Bitcoin/Ethereum supply, and shows a
prompt migration beats even an optimistic CRQC arrival.

## What's implemented

| Paper section | Module | Figure/Table |
|---|---|---|
| §3.2 systemic break-year forecast | `src/quantum_horizon/timeline/` | Figure 2 |
| §2.2 mining-competitiveness model | `src/quantum_horizon/mining/` | Figure 1 |
| §4 Bitcoin exposure, §5 Ethereum exposure | `src/quantum_horizon/exposure/` | Figure 3 |
| §4.3 mempool-sniping race | `src/quantum_horizon/attacks/` | — |
| §7.4 Mosca's-inequality migration race | `src/quantum_horizon/migration/` | Figure 4 |
| §6 top-20 readiness survey | `src/quantum_horizon/survey/` | Table 3, Figure 5 |

**Note: this paper has no trained model.** Every module here is either a
Monte-Carlo simulation with fixed swept parameter ranges, a deterministic
calibrated formula, or a data-reconciliation calculation. No optimizer, no
loss function, no learned parameters anywhere in this reproduction.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Run everything (all 5 analyses, all 5 figures)
python run_all.py --config configs/config.yaml --output-dir results/

# Or run individually:
python run_timeline_forecast.py --config configs/config.yaml --sensitivity-sweep
python run_mining_analysis.py --config configs/config.yaml
python run_exposure_analysis.py --config configs/config.yaml --chain all
python run_migration_race.py --config configs/config.yaml
python run_readiness_survey.py --config configs/config.yaml
```

Or via Docker:
```bash
docker compose -f docker/docker-compose.yml up --build
```

## Repository layout

```
configs/config.yaml            All calibration constants (# ASSUMED comments flag unstated values)
src/quantum_horizon/
  timeline/                    Physics estimator, survey estimator, blended Monte-Carlo forecast
  mining/                      Grover-vs-PoW mining competitiveness model
  exposure/                    Bitcoin + Ethereum exposure reconciliation
  attacks/                     Mempool-sniping race model
  migration/                   Mosca's-inequality migration-race decision rule
  survey/                      Top-20 readiness survey loader (transcribed data, not a model)
  utils/                       Config loading, seeding, Figure 1-5 plotting
run_timeline_forecast.py / run_mining_analysis.py / run_exposure_analysis.py /
run_migration_race.py / run_readiness_survey.py / run_all.py    Entrypoints
tests/                         45 unit tests, most asserting against the paper's own reported numbers
data/                          Source citations + transcribed Table 3
docker/                        Dockerfile + docker-compose.yml
```

## Reproducibility notes (read before trusting exact numbers)

This paper is unusually well-sourced for its *headline numbers* (every
figure traces to a named citation) but unusually narrative for its
*methodology* — several sub-models are described only as swept ranges plus
output percentiles, with no explicit formula given. Where this repo had to
fill a gap, I verified the result lands close to the paper's own stated
output rather than just assuming a formula would work:

1. **Timeline forecast (§3.2)** — the paper gives swept ranges for hardware
   doubling time, resource-requirement halving time, and fault-tolerance
   lag, plus output percentiles, but never the sampling distribution shape.
   This repo's calibration (`survey_estimator_mode_year=2036`,
   `survey_estimator_sigma=0.45`, `physics_calibration_2026_physical_qubits_required=10000000`)
   was **tuned to match the paper's reported outputs**, not derived from the
   paper's stated inputs alone — see `timeline/physics_estimator.py` and
   `survey_estimator.py` docstrings for exactly what was tuned and why.
   Result: median 2046.3 (paper: 2046-2047), P(by 2040)=30.8% (paper: ~30%),
   P(by 2050)=63.2% (paper: ~60%), 80% range 2034-2058 (paper: ~2032-2060).
2. **Mempool-race model (§4.3)** — the paper reports 41%/30% success
   probabilities without showing the timing calculation. This repo's
   exponential race-condition formula was chosen because it reproduces the
   stated 41% best-case almost exactly under the paper's stated best-case
   inputs (9-min derivation, zero propagation, 1 confirmation) — see
   `attacks/mempool_race.py`'s docstring for why a naive linear formula
   does *not* reproduce this and what was used instead.
3. **Mining "51% of network" figure (§2.2)** — this repo's straightforward
   sqrt(K)-scaling reconstruction gives ~4.5×10¹⁴ machines needed, versus
   the paper's stated ~7×10¹³ — a ~6x discrepancy. Everything else in the
   mining model matches the paper closely (hashrate at 100GHz: 20.69 TH/s
   vs. paper's ~21 TH/s), so this is flagged as an open discrepancy rather
   than silently forced to match — the paper doesn't give enough detail to
   know which assumption differs.
4. **Exposure-reconciliation methods (§4.2, §5.1)** — both Bitcoin and
   Ethereum models combine multiple named sources into a headline range
   without a stated formula; this repo uses a simple mean (Bitcoin) and
   tuned `contract_fraction`/`beacon_overcount_correction_factor` constants
   (Ethereum) that land within the paper's stated ranges.
5. **Top-20 readiness survey (§6)** is explicitly *not* a model per the
   paper's own Appendix A — it's transcribed directly from Table 3 into
   `data/table3_readiness_ratings.csv`, with unit tests checking against
   the paper's stated qualitative findings (no coin reaches 5; Bitcoin and
   Dogecoin near the bottom; XRP/Solana/Zcash lead).

Full detail and confidence scores per section are in
`sir-registry/arxiv_2606_14484/sir.json` (Stage 1, overall confidence 0.64
— below the usual 0.65 comfort threshold, reflecting how much of this
paper's methodology is narrative rather than formulaic) and
`architecture_plan.json` (Stage 3, `risk_assessment` field).

## Testing

```bash
pytest tests/ -v          # 45 tests, all passing
ruff check src/ *.py       # clean
```

Most tests assert against the paper's own reported numbers with a
tolerance band (since several models are ASSUMED/tuned reconstructions, not
exact reproductions) — e.g. `test_systemic_forecast_matches_paper_ballpark`
checks the combined forecast against wide bands around the paper's stated
percentiles rather than exact equality.

## Citation

```
Gershteyn, I. M., & Alber, J. A. (2026). Quantum Horizon: An evaluation of
quantum computing as a threat to Bitcoin and Ethereum. arXiv:2606.14484.
```

This is an independent reproduction generated by the ArXivist pipeline; it
is not affiliated with the paper's authors. The authors' own code is cited
at https://github.com/imgcode/quantum-horizon but was not available to this
reproduction pipeline — see `data/README_data.md` for all source citations
used instead.
