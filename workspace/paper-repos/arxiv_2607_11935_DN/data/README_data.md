# Data — arXiv:2607.11935 reproduction

The paper's observational analysis uses monthly NASA AIRS (Atmospheric
Infrared Sounder) data, freely available via the Giovanni portal, but
requiring an interactive pull (or NASA Earthdata API credentials) that
cannot be scripted unattended.

## Option 1: Use real NASA AIRS data

1. Go to https://giovanni.gsfc.nasa.gov/giovanni/
2. For each of the three regions, extract monthly time series (Sept 2002 –
   Apr 2026, 284 months) of:
   - `SurfSkinTemp_A` (surface skin temperature) → column `T`
   - `H2O_MMR_Surf_A` (surface water-vapor mixing ratio) → column `q`
3. Region bounding boxes (see `configs/config.yaml`):
   - Arctic: 65°N–90°N
   - Tropics: 10°S–10°N
   - Indian Monsoon: 60°E–100°E, 5°N–25°N
4. Save each region as `data/raw/airs_<region>.csv` with columns `date, T, q`
   (e.g. `data/raw/airs_arctic.csv`, `data/raw/airs_tropics.csv`,
   `data/raw/airs_monsoon.csv`).
5. Set `data.use_synthetic_fallback: false` in `configs/config.yaml`.

See `download.sh` for the exact variable names and a documented (manual)
retrieval checklist.

## Option 2: Synthetic fallback (default)

With `data.use_synthetic_fallback: true` (the default), `AIRSDataLoader`
generates a synthetic T, q series per region calibrated only to roughly
match the paper's reported per-region |beta| magnitude (Table 1: Arctic
~0.11 weak/noisy coupling, Tropics ~0.49 near Clausius-Clapeyron and stable,
Monsoon ~0.48 near C-C but decelerating). **This is illustrative synthetic
data, not real AIRS observations** — it lets the full pipeline (Kalman
filter, classical EWS, lead-lag analysis, Tables 1-2, Figures 1-2) run
end-to-end and sanity-check the code without live data access, but exact
correlation values, transition counts, and lead-lag months will not match
the paper's real-data numbers.

## Files

- `download.sh` — documents the exact Giovanni portal variables/regions/date-range needed.
- `README_data.md` — this file.

No raw data files are committed to this repository (see `.gitignore`).
