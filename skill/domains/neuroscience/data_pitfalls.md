# Domain: Neuroscience — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical reproducibility traps in Neuroscience papers

**Subject-level data leakage** — the most common and damaging pitfall. If trials from
the same subject appear in both train and test sets, the model learns subject-specific
patterns rather than general neural codes. Always split at the subject level, not the
trial level, unless the paper explicitly evaluates within-subject.

**Preprocessing fitted on full dataset** — ICA decomposition, normalisation, and
baseline correction fitted on train+test before splitting introduce test-set information
into training. Preprocessing must be fitted on training data only and applied to test.

**Epoch rejection bias** — if bad epochs are rejected based on the full dataset's
statistics before splitting, the rejection is implicitly informed by test data. Implement
rejection thresholds derived from training data only.

**MRI/fMRI data sharing restrictions** — neuroimaging datasets (OpenNeuro, UK Biobank,
HCP) have data use agreements. Some require registration or institutional approval.
Note data access requirements in `data/README_data.md` for all restricted datasets.

**Atlas version mismatch** — brain parcellation atlases (AAL, Desikan-Killiany, Schaefer)
have multiple versions with different region counts. fMRI connectivity results are not
comparable across atlas versions. Always extract and record the exact atlas version.

**BIDS format assumptions** — most modern neuroimaging datasets use BIDS format, but
the exact BIDS version and optional fields vary. MNE-BIDS and Nilearn expect specific
BIDS structures. If the paper uses a non-standard organisation, flag it.

**Stimulus timing precision** — EEG experiments are sensitive to stimulus onset timing
jitter (< 1ms matters). If the paper used specialised hardware triggers for timing and
the reproduction uses software triggers, ERP latencies and amplitudes may differ.

**Session-to-session variability** — EEG electrode impedance, placement, and amplifier
settings change between recording sessions. Results from different recording sessions
of the same subject cannot be directly compared without normalisation.
