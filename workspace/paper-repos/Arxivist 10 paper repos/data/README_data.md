# Data Notes

This paper does **not** use any external/downloaded dataset. All "data" is
generated on-the-fly by the simulated limit order book market environment
described in Section 5 (`src/rlte/env/`):

- **Noise traders** (Poisson-process order flow, Section 5.1)
- **Tactical traders** (imbalance-reactive order flow, Section 5.2)
- **Strategic trader** (large TWAP buyer/seller, Section 5.3)

There is therefore no `download.sh` / `download.py` needed. Instead, this
directory documents the two artifacts that *are* estimated empirically from
simulation before training/evaluation, per Appendix A.2:

1. **Long-term average order book shape** (`v_tilde` in Appendix A.2): the
   average volume at each price level, estimated by running the noise-trader
   (and noise+tactical) simulation for a long time and averaging. Used to
   initialize the order book at the start of each episode and to normalize
   volume features (Appendix B.3).
2. **Traded volume statistics** (Appendix A.3, Table 4): average number of
   events and traded volume per 150s episode, used only for sanity-checking
   the simulator's calibration against Table 4 in the paper.

A helper script `data/estimate_equilibrium.py` (STUB in this scaffold) should
run `NoiseTraders`/`TacticalTraders` for a long warm-up period and save the
resulting average shape as a `.npy` file for `FeatureNormalizer` to consume,
matching Figure 8a/8b in the paper. This is not yet implemented in this
reference scaffold (`FeatureNormalizer` currently falls back to a flat
placeholder shape if none is supplied -- see `execution_env.py`).
