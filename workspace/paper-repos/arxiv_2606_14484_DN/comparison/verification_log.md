# Verification Log
**Paper ID**: arxiv_2606_14484
**Comparison run timestamp**: 2026-07-21T15:20:37Z
**ArXivist SIR version used**: 1 (`sir-registry/arxiv_2606_14484/sir.json`, overall_sir_confidence 0.64)
**Architecture plan version used**: 1 (`architecture_plan.json`)

## Paper metrics vs. matched
- Total metrics in `evaluation_protocol.reported_results`: 15
- Metrics matched to a user-reported value: 13
- Metrics unmatched: 2 (2026 Bitcoin network hashrate; Ethereum dormant-and-exposed fraction)

## Metrics compared (in order reported by user)
1. Logical qubits to break secp256k1 (2026 frontier)
2. Physical qubits to break secp256k1 (2026 frontier, upper bound)
3. Best demonstrated physical qubits (2026)
4. P(CRQC by 2035)
5. P(CRQC by 2040)
6. P(CRQC by 2050)
7. Bitcoin exposed at rest (% of supply)
8. Bitcoin irreducibly-at-risk (M BTC)
9. Bitcoin migratable exposure (M BTC)
10. Ethereum at-rest exposure (top-down / bottom-up / reconciled midpoint)
11. Mempool-sniping: best-case and realistic success probability
12. Slow-clock (trapped-ion) feasibility — qualitative, confirmed matches paper ("impossible")
13. Median break-year and 80% credible range — qualitative, confirmed matches paper (~2046 vs. paper's ~2046-2047; 2034-2058 vs. paper's ~2032-2060)
14. Single quantum-machine hashrate @ 100 GHz (TH/s) and comparison to one ASIC
15. Migration finish year across prompt/delayed/severe-delay scenarios, and at-risk-scenario count (1/9)

## User-reported config modifications
None reported by the user. Assumed defaults from `configs/config.yaml` were used throughout
(notably `survey_weight: 0.5`, `reconciliation_method: simple_mean`,
`realistic_propagation_delay_minutes: 2.0`). This assumption is noted as a limitation — if the
user in fact changed any of these, the root cause analysis in `benchmark_comparison.md`
attributing the P(CRQC by 2035) miss to modeling assumptions rather than config drift should be
revisited.

## Traceability
- User results input: pasted text + one embedded table (coin readiness ratings) + one embedded
  table (migration-race scenario sweep), provided directly in chat, not as a file. No SHA256 is
  computed since the raw input was conversational text rather than an uploaded artifact.
- Source files referenced for ground truth: `sir.json` (`evaluation_protocol.reported_results`,
  `confidence_annotations`), `architecture_plan.json` (`risk_assessment`, `module_hierarchy`),
  `data/table3_readiness_ratings.csv`.

## Registry update
`sir-registry/arxiv_2606_14484/metadata.json`: `has_comparison_report` set to `true`.
