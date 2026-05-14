# Verification Log
**Paper ID**: arxiv_2605_volatility_sig  
**ArXivist Pipeline Run**: 2026-05-13T00:00:00Z  
**SIR Version**: 1  
**Architecture Plan Version**: 1  
**Notebook Version**: 1  

---

## Pipeline Execution Record

| Stage | Name | Status | Duration | Confidence |
|-------|------|--------|----------|-----------|
| 1 | PaperParser | ✓ Complete | — | 0.86 |
| 2 | SIRRegistry | ✓ Complete | — | 0.95 |
| 3 | ArchitecturePlanner | ✓ Complete | — | 0.90 |
| 4 | CodeGenerator | ✓ Complete | — | 0.88 |
| 5 | NotebookGenerator | ✓ Complete | — | 0.92 |
| 6 | ResultsComparator | ✓ Baseline only | — | N/A (no user results) |

---

## Source Document

- **Input**: PDF uploaded by user, 42 pages, arXiv:2507.23392v4
- **PDF processing**: Full text extracted from all 42 pages via document context
- **Figures**: 3 figures (IV surface plots); content inferred from captions and surrounding text
- **Tables**: 5 tables (Tables 2.1–2.3, 5.1–5.2, 6.1); all numerical values extracted verbatim

---

## Paper Metrics Extracted

| ID | Experiment | Metric | Value | Source |
|----|-----------|--------|-------|--------|
| M1 | heston_uncorr | loss_at_convergence | 1.05e-4 | Section 5.1 explicit |
| M2 | heston_corr | loss_at_convergence | 1.46e-3 | Section 5.2 explicit |
| M3 | rough_bergomi | loss_at_convergence | 3.5e-4 | Section 6 explicit |
| M4 | heston_uncorr | l_star[0] | 0.201202133 | Section 5.1 explicit |
| M5 | heston_uncorr | l_star[2] | 1.08471290 | Section 5.1 explicit |
| M6 | heston_uncorr | all 15 l_star components | listed | Section 5.1 explicit |
| M7 | heston_corr | all 15 l_star components | listed | Section 5.2 explicit |
| M8 | rough_bergomi | all 15 l_star components | listed | Section 6 explicit |
| M9 | rough_bergomi | calibration_time | 17-19 min | Section 6 explicit |
| M10 | heston_uncorr | Table 5.1 (all 20 IV errors) | listed | Table 5.1 explicit |
| M11 | heston_corr | Table 5.2 (all 20 IV errors) | listed | Table 5.2 explicit |
| M12 | rough_bergomi | Table 6.1 (all 20 IV errors) | listed | Table 6.1 explicit |

---

## Implementation Assumptions (all ASSUMED values)

| ID | Parameter | Assumed Value | Confidence | Location |
|----|-----------|---------------|------------|----------|
| A1 | T_steps_per_unit | 252 | 0.55 | config.yaml |
| A2 | box_bounds | [-10, 10] | 0.45 | config.yaml |
| A3 | l0_init | zeros | 0.60 | optimizer.py |
| A4 | x0_is_variance | true | 0.60 | config.yaml |
| A5 | fbm_method | cholesky | 0.75 | primary_process.py |
| A6 | cholesky_reg_eps | 1e-8 | 0.90 | compute.py |
| A7 | seed | 42 | 1.00 | config.yaml (irrelevant for comparison) |
| A8 | num_workers | 4 | 1.00 | config.yaml (hardware only) |

---

## Code Artifacts Generated

| File | Lines | Key paper section |
|------|-------|------------------|
| src/volsig/models/signature_vol.py | ~150 | Section 4.3 (full algorithm) |
| src/volsig/models/heston.py | ~200 | Section 2.1 (ASV expansion) |
| src/volsig/models/rough_bergomi.py | ~220 | Section 2.2 (VIX calibration) |
| src/volsig/models/primary_process.py | ~130 | Sections 5–6 (primary process) |
| src/volsig/signatures/compute.py | ~280 | Sections 3–4 (signatures, Q matrix) |
| src/volsig/pricing/black_scholes.py | ~130 | Throughout (BS formula) |
| src/volsig/pricing/mc_pricer.py | ~160 | Proposition 4.2 |
| src/volsig/calibration/optimizer.py | ~140 | Section 4.2 (loss, optimizer) |
| src/volsig/utils/config.py | ~140 | — |
| src/volsig/utils/plotting.py | ~110 | Figures 5.1, 5.2, 6.1 |
| train.py | ~180 | Section 4.3 algorithm |
| evaluate.py | ~70 | Tables 5.1, 5.2, 6.1 |
| calibrate_heston_asv.py | ~70 | Section 2.1 |
| calibrate_rbergomi_vix.py | ~90 | Section 2.2 |

**Total**: ~2050 lines of Python (excluding notebooks and configs)  
**Syntax validation**: All 21 Python files pass `ast.parse()` ✓

---

## Syntax Validation Results

```
✓ src/volsig/__init__.py
✓ src/volsig/models/__init__.py
✓ src/volsig/models/signature_vol.py
✓ src/volsig/models/rough_bergomi.py
✓ src/volsig/models/heston.py
✓ src/volsig/models/primary_process.py
✓ src/volsig/signatures/__init__.py
✓ src/volsig/signatures/compute.py
✓ src/volsig/calibration/__init__.py
✓ src/volsig/calibration/optimizer.py
✓ src/volsig/pricing/__init__.py
✓ src/volsig/pricing/black_scholes.py
✓ src/volsig/pricing/mc_pricer.py
✓ src/volsig/utils/config.py
✓ src/volsig/utils/__init__.py
✓ src/volsig/utils/plotting.py
✓ src/__init__.py
✓ train.py
✓ evaluate.py
✓ calibrate_heston_asv.py
✓ calibrate_rbergomi_vix.py
```

---

## User Results Submission Record

*This section will be populated when user submits results.*

| Submission | Timestamp | Experiment | Loss | Source file |
|------------|-----------|-----------|------|------------|
| — | PENDING | — | — | — |

---

## Audit Notes

1. The paper's ℓ* vectors were transcribed verbatim from Sections 5.1, 5.2, and 6. These serve as exact numerical targets for validation.
2. The paper uses a consumer GPU (RTX 3080 Ti) — results on different hardware will differ only due to floating-point ordering, not algorithmic differences (NumPy operations are deterministic given the same random seed on the same hardware class).
3. The paper does not state whether mixed precision (float16/32) was used. We assume float64 throughout, which may cause minor numerical differences but should not affect the order of magnitude of results.
4. The rough Bergomi experiment (Section 6) uses H=0.2 for the primary process and H=0.1 for the market model. This intentional mismatch is correctly implemented in `patch_experiment()` in `train.py`.
