# Domain: Economics — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Economics**.

---

## Framework selection

**Default: Python with statsmodels and linearmodels** for most empirical economics.
**Use R (via rpy2 or as a separate script)** if the paper explicitly uses R packages
(fixest, rdrobust, synthdid, ivreghdfe) that have no Python equivalent.
**Use PyMC or Stan** for structural Bayesian estimation.

## Module structure for economics papers

```
src/{project}/
├── estimation/
│   ├── ols.py             ← OLS and variants (robust SE, clustered SE)
│   ├── iv.py              ← IV/2SLS/LIML
│   ├── panel.py           ← Fixed effects, random effects, DiD
│   ├── rdd.py             ← Regression discontinuity
│   └── gmm.py             ← GMM estimator
├── data/
│   ├── dataset.py         ← Data loading with vintage tracking
│   └── cleaning.py        ← Merge, winsorise, construct variables
├── inference/
│   ├── standard_errors.py ← HC0-HC3, clustered, bootstrap
│   └── hypothesis.py      ← Hypothesis tests, multiple testing corrections
├── simulation/            ← For structural and DSGE papers
│   └── model.py
└── tables/
    └── latex_output.py    ← Reproduce paper tables in LaTeX format
```

**Always generate LaTeX table output.** Economics papers are judged by their tables.
The reproduction is complete when the generated tables match the paper's tables.

---

# Domain: Economics — Code Conventions (Stage 4 Enrichment)

## Standard error implementation

Always implement the exact standard error type the paper uses:

```python
import statsmodels.formula.api as smf

# Clustered standard errors
result = smf.ols("y ~ x1 + x2", data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cluster_var"]}
)

# Two-way clustered SE (cluster on two dimensions)
# Use linearmodels for panel data with two-way clustering
from linearmodels.panel import PanelOLS
```

## DiD implementation

```python
def difference_in_differences(df, outcome, treatment, time, post, controls=None):
    """
    Standard DiD estimator.
    Verify: parallel trends assumption, no anticipation effects.
    """
    df["DiD"] = df[treatment] * df[post]
    formula = f"{outcome} ~ DiD + {treatment} + {post}"
    if controls:
        formula += " + " + " + ".join(controls)
    return smf.ols(formula, data=df).fit(cov_type="HC3")
```

## Table reproduction

Always produce a `tables/` directory with:
- One `.tex` file per table in the paper
- A `compile_tables.py` script that generates all tables from estimation results
- Tables formatted to match the paper's style as closely as possible

## What Must NOT be done

- Do NOT use `random_state` to shuffle panel data — observations must stay in their
  original temporal and cross-sectional positions
- Do NOT use ML train/test splits for causal inference — the identification strategy
  defines what is valid
- Do NOT omit control variables from a specification — all controls affect coefficient estimates

---

# Domain: Economics — Evaluation Standards (Stage 6 Enrichment)

## What "reproduction" means in economics

Economics reproduction is coefficient-level, not metric-level. The comparison is:
- Coefficient estimate (β) — should match to 2–3 decimal places
- Standard error — should match to 2 decimal places
- t-statistic and p-value — should produce the same significance conclusion
- R² — should match to 3 decimal places

## Deviation thresholds

| Quantity | Excellent | Good | Moderate | Notes |
|----------|-----------|------|----------|-------|
| Coefficient | ≤ 0.5% | ≤ 2% | ≤ 5% | Relative to magnitude |
| Standard error | ≤ 1% | ≤ 3% | ≤ 8% | |
| R² | ≤ 0.001 | ≤ 0.003 | ≤ 0.01 | Absolute |
| F-statistic (first stage) | ≤ 5% | ≤ 15% | ≤ 30% | Relative |

**If significance conclusion differs (p < 0.05 vs p > 0.05), always classify as Significant
regardless of coefficient deviation.** The inference is wrong even if the point estimate is close.

## Root causes specific to economics

1. **Different data vintage** — GDP and macro data are revised; using revised vs real-time data
   changes regression coefficients, sometimes substantially.
2. **Sample period mismatch** — one missing year or quarter changes all fixed effects estimates.
3. **Standard error formula mismatch** — HC2 vs HC3 vs clustered produces different SEs.
4. **Missing control variables** — omitted variables change coefficient estimates through OVB.
5. **Proprietary data** — many economics papers use data that cannot be obtained freely.
   If the dataset is proprietary, reproducibility requires the exact same data access.

---

# Domain: Economics — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

**Data revisions** — macroeconomic series (GDP growth, inflation, unemployment) are revised
multiple times after initial release. Papers using real-time data cannot be reproduced with
final revised data. Flag whenever macro series are used without specifying vintage.

**Proprietary microdata** — Census microdata, administrative records (tax data, health records,
social security data), and firm-level surveys are not publicly available. For such papers,
note in `data/README_data.md` that exact reproduction requires data access approval.

**Sample construction criteria** — papers routinely exclude observations based on undisclosed
rules (trimming outliers, excluding firms below a size threshold). These rules
materially affect results. Flag any exclusion criterion not fully specified.

**Panel attrition** — if a panel dataset loses observations over time, the remaining sample
is selected. Papers often do not address attrition. Flag if the panel is unbalanced without
explanation of the attrition pattern.

**Matched datasets** — when papers merge multiple data sources, the merge keys and merge
type (inner, left, outer) are rarely fully specified. Different merge assumptions can change
the sample size substantially and bias results.
