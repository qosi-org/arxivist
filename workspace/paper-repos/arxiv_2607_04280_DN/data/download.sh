#!/usr/bin/env bash
# ArXivist-generated data setup script for arxiv_2607_04280.
#
# This paper's "data" is entirely SIMULATED — the LOB ABM generates its own
# market data as it runs (Section 2). There is nothing to download.
#
# The one external reference used for comparison is the Tokyo Stock Exchange
# (TSE) benchmark statistic delta=0.489, c=0.842, from Sato & Kanazawa (2025,
# Physical Review Letters). That single summary statistic is already baked
# into configs/config.yaml (evaluation.tse_benchmark_mean_delta /
# tse_benchmark_mean_c) — the underlying per-stock TSE dataset is not public
# and is not required to run this repository.

set -e

echo "No external data download required for arxiv_2607_04280."
echo "This repository generates its own simulated market data at runtime."
echo ""
echo "To produce data, run one of:"
echo "  python train.py --config configs/config.yaml --debug     # smoke test, ~seconds"
echo "  python train.py --config configs/config.yaml              # full baseline run"
echo "  python run_counterfactual_suite.py --config configs/config.yaml"
echo ""
echo "See data/README_data.md for details."
