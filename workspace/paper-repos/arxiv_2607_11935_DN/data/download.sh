#!/usr/bin/env bash
# download.sh -- documents the NASA AIRS Giovanni-portal pulls required to
# reproduce arXiv:2607.11935's observational analysis from raw data.
#
# The Giovanni portal (https://giovanni.gsfc.nasa.gov/giovanni/) requires an
# interactive session (or NASA Earthdata API credentials) and cannot be
# scripted unattended by this file. This script exists as DOCUMENTATION of
# the exact pulls needed; perform the equivalent queries manually and save
# results as CSVs in data/raw/ with the layout below.

set -euo pipefail

echo "=================================================================="
echo " NASA AIRS data requirements (arXiv:2607.11935, Section 2.1)"
echo "=================================================================="
echo ""
echo "Portal: https://giovanni.gsfc.nasa.gov/giovanni/"
echo "Dataset: AIRS/AMSU/HSB (Aqua mission), monthly averages"
echo "Date range: September 2002 - April 2026 (284 months)"
echo ""
echo "Variables:"
echo "  T (surface skin temperature): SurfSkinTemp_A"
echo "  q (surface water-vapor mixing ratio): H2O_MMR_Surf_A"
echo ""
echo "Regions and required output files (data/raw/):"
echo "  1) airs_arctic.csv    -- lat 65N to 90N,  lon -180 to 180"
echo "  2) airs_tropics.csv   -- lat 10S to 10N,   lon -180 to 180"
echo "  3) airs_monsoon.csv   -- lat 5N to 25N,    lon 60E to 100E"
echo ""
echo "Each CSV must have columns: date, T, q"
echo "(one row per month, spatially averaged over the region's grid cells)"
echo ""
echo "Once these files exist, set data.use_synthetic_fallback: false in"
echo "configs/config.yaml. Otherwise, AIRSDataLoader generates a synthetic"
echo "fallback series calibrated to roughly match the paper's Table 1"
echo "|beta| magnitudes per region (see README_data.md)."
