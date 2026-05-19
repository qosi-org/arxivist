# Domain: Economics — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical data-level reproducibility traps in Economics papers

**Data revisions** — macroeconomic series (GDP growth, inflation, unemployment) are revised
multiple times after initial release. Papers using real-time data cannot be reproduced with
final revised data. Flag whenever macro series are used without specifying vintage.

**Proprietary microdata** — Census microdata, administrative records (tax data, health records,
social security data), and firm-level surveys are not publicly available. For such papers,
note in `data/README_data.md` that exact reproduction requires data access approval and
provide a synthetic data generation script as a substitute.

**Sample construction criteria** — papers routinely exclude observations based on undisclosed
rules (trimming outliers, excluding firms below a size threshold). These rules materially
affect results. Flag any exclusion criterion not fully specified.

**Panel attrition** — if a panel dataset loses observations over time, the remaining sample
is selected. Papers often do not address attrition. Flag if the panel is unbalanced without
explanation of the attrition pattern.

**Matched datasets** — when papers merge multiple data sources, the merge keys and merge type
(inner, left, outer) are rarely fully specified. Different merge assumptions can change the
sample size substantially and bias results.

**Deflation methodology** — nominal vs real values depend on the price index used and the
base year. If the paper uses real values, extract the deflator and base year explicitly.
