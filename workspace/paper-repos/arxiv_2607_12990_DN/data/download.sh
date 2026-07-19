#!/usr/bin/env bash
# download.sh -- documents the LSEG Workspace pulls required to reproduce
# arXiv:2607.12990's classical market calibration from raw data (Table 8).
#
# LSEG Workspace access is proprietary and cannot be automated by this
# script. This file exists as DOCUMENTATION of the exact data needed; run
# the equivalent pulls manually in your own LSEG Workspace / Eikon session
# and save the results as CSVs in data/raw/ with the column layouts below.

set -euo pipefail

echo "=============================================================="
echo " LSEG Workspace data requirements (arXiv:2607.12990, Table 8)"
echo "=============================================================="
echo ""
echo "This script cannot auto-download proprietary LSEG data. Please pull"
echo "the following manually and save as CSVs in data/raw/:"
echo ""
echo "1) data/raw/equity_closes.csv"
echo "   Columns: date, EURO_STOXX_50, SMI"
echo "   Range:   2021-03-13 to 2026-03-13 (daily closes)"
echo ""
echo "2) data/raw/implied_vol_surface.csv"
echo "   Columns: underlying, expiry, atm_implied_vol"
echo "   EURO STOXX 50: 35 expiries (2026-2035); SMI: 18 expiries (2026-2030)"
echo ""
echo "3) data/raw/eur_ois_curve.csv"
echo "   Columns: tenor, mid_rate, discount_factor"
echo "   Pillars: 7D through 60Y"
echo ""
echo "4) data/raw/iberdrola_cds_curve.csv"
echo "   Columns: tenor, par_spread_bps"
echo "   Tenors: 6M through 30Y"
echo ""
echo "Once these files exist, set data.use_synthetic_fallback: false in"
echo "configs/config.yaml. Otherwise, train.py uses the paper's own"
echo "published Table 4 calibration outputs directly (see README_data.md)."
