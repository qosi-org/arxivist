# Data — arXiv:2606.14484 reproduction

Unlike a typical ML paper, this repository doesn't need a large dataset —
every model here consumes a handful of scalar calibration constants drawn
directly from the paper's cited sources. All of these are already
transcribed into `configs/config.yaml`.

## Table 3 readiness ratings

`table3_readiness_ratings.csv` is a direct transcription of the paper's
Table 3 (Section 6) — signature scheme, exposure model, post-quantum status,
and ordinal rating (1-5) for 16 of the paper's ~19-20 surveyed
cryptocurrencies (the paper excludes a few small/host-inherited entries from
its own count; this transcription keeps the ones with independently
meaningful ratings). This is **not derivable from any formula** — the paper
itself states in Appendix A that "the market survey of §6 is a sourced field
assessment, not a model."

If you want to extend this to the full ~19-20 coin set, add rows following
the same schema; `tests/test_survey.py` includes drift-detection checks
(e.g. "no coin reaches 5", "Bitcoin and Dogecoin near the bottom") that will
catch obvious transcription errors.

## Calibration sources (all in `configs/config.yaml`)

| Constant | Value | Source |
|---|---|---|
| Mining benchmark | 13.8 GH/s @ 66.7 MHz | Aggarwal et al. 2017, arXiv:1710.10377 |
| 2026 logical/physical qubit requirement (aggressive) | 1,200-1,450 / <500,000 | Babbush, Gidney, et al. 2026, arXiv:2603.28846 |
| 2026 logical/physical qubit requirement (conservative) | — / ~317,000,000 | Webber et al. 2022, arXiv:2108.12371 |
| 2017 ECC resource estimate | 2,330 logical qubits | Roetteler et al. 2017, arXiv:1706.06752 |
| RSA-2048 algorithmic improvement | 20x reduction | Gidney 2025, arXiv:2505.15917 |
| Bitcoin exposure (2025-26) | Glassnode 30.2%, Coinbase ~6.9M, CoinDesk ~7M | Glassnode Research May 2026; see paper citations 15-17 |
| Bitcoin exposure (2020 baseline) | ~25% | Deloitte Netherlands 2020 |
| Ethereum exposure (2021 baseline) | ~65% (full ledger scan) | Deloitte Netherlands 2021 |
| SHA-2/3 preimage fault-tolerant cost | ~2^166 logical-qubit-cycles | Amy et al. 2016, arXiv:1603.09383 |
| Expert survey timeline | mode ~2038-2040 | Global Risk Institute / Mosca-Piani 2024/2025 reports |

## What's NOT reproducible from the paper alone

Several sub-models are described only narratively (ranges + headline
outputs, no explicit formula) — see `README.md` "Reproducibility Notes" and
`sir-registry/arxiv_2606_14484/sir.json`'s `ambiguities` section for the
full list of what's ASSUMED vs. paper-stated. The paper's own code is cited
at https://github.com/imgcode/quantum-horizon but was not available to this
reproduction pipeline.
