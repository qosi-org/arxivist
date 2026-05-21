# RESULTS_COMPARISON

This document compares the paper's reported Table 1 results against the empirical results produced by the local Quantum-SMOTE implementation.

Paper baselines were extracted from `QSMOTR_SIR.json` (`evaluation_protocol.results_table`) and cross-checked against the attached table image. Empirical values were parsed from the terminal output provided in the prompt.

## Comparison Table

| Model | Condition | Metric | Paper Result | My Result | Difference (Delta) |
|---|---|---:|---:|---:|---:|
| Random Forest | Without Synthetic | Accuracy (Test) | 0.7840 | 0.7797 | -0.0043 |
| Random Forest | Without Synthetic | F1-Score | 0.5750 | 0.5359 | -0.0391 |
| Random Forest | Without Synthetic | PR-AUC | 0.6270 | 0.5952 | -0.0318 |
| Random Forest | Without Synthetic | ROC-AUC | 0.8110 | 0.8154 | +0.0044 |
| Random Forest | 30% Minority + Synthetic | Accuracy (Test) | 0.8010 | 0.8085 | +0.0075 |
| Random Forest | 30% Minority + Synthetic | F1-Score | 0.6340 | 0.7051 | +0.0711 |
| Random Forest | 30% Minority + Synthetic | PR-AUC | 0.7580 | 0.8260 | +0.0680 |
| Random Forest | 30% Minority + Synthetic | ROC-AUC | 0.8540 | 0.8786 | +0.0246 |
| Random Forest | 40% Minority + Synthetic | Accuracy (Test) | 0.8220 | 0.8195 | -0.0025 |
| Random Forest | 40% Minority + Synthetic | F1-Score | 0.7640 | 0.7502 | -0.0138 |
| Random Forest | 40% Minority + Synthetic | PR-AUC | 0.8880 | 0.8710 | -0.0170 |
| Random Forest | 40% Minority + Synthetic | ROC-AUC | 0.9050 | 0.8963 | -0.0087 |
| Random Forest | 50% Minority + Synthetic | Accuracy (Test) | 0.8460 | 0.8490 | +0.0030 |
| Random Forest | 50% Minority + Synthetic | F1-Score | 0.8350 | 0.8410 | +0.0060 |
| Random Forest | 50% Minority + Synthetic | PR-AUC | 0.9400 | 0.9417 | +0.0017 |
| Random Forest | 50% Minority + Synthetic | ROC-AUC | 0.9290 | 0.9297 | +0.0007 |
| Logistic Regression | Without Synthetic | Accuracy (Test) | 0.7660 | 0.7996 | +0.0336 |
| Logistic Regression | Without Synthetic | F1-Score | 0.5240 | 0.5983 | +0.0743 |
| Logistic Regression | Without Synthetic | PR-AUC | 0.6040 | 0.6220 | +0.0180 |
| Logistic Regression | Without Synthetic | ROC-AUC | 0.8150 | 0.8338 | +0.0188 |
| Logistic Regression | 30% Minority + Synthetic | Accuracy (Test) | 0.7590 | 0.8256 | +0.0666 |
| Logistic Regression | 30% Minority + Synthetic | F1-Score | 0.5370 | 0.7408 | +0.2038 |
| Logistic Regression | 30% Minority + Synthetic | PR-AUC | 0.6330 | 0.8433 | +0.2103 |
| Logistic Regression | 30% Minority + Synthetic | ROC-AUC | 0.8120 | 0.8953 | +0.0833 |
| Logistic Regression | 40% Minority + Synthetic | Accuracy (Test) | 0.7000 | 0.8309 | +0.1309 |
| Logistic Regression | 40% Minority + Synthetic | F1-Score | 0.6080 | 0.7739 | +0.1659 |
| Logistic Regression | 40% Minority + Synthetic | PR-AUC | 0.6740 | 0.8848 | +0.2108 |
| Logistic Regression | 40% Minority + Synthetic | ROC-AUC | 0.7690 | 0.9104 | +0.1414 |
| Logistic Regression | 50% Minority + Synthetic | Accuracy (Test) | 0.7340 | 0.8558 | +0.1218 |
| Logistic Regression | 50% Minority + Synthetic | F1-Score | 0.7420 | 0.8514 | +0.1094 |
| Logistic Regression | 50% Minority + Synthetic | PR-AUC | 0.7790 | 0.9481 | +0.1691 |
| Logistic Regression | 50% Minority + Synthetic | ROC-AUC | 0.8070 | 0.9382 | +0.1312 |

## Summary

The empirical run reproduces the paper's main trend: adding Quantum-SMOTE improves minority-class performance over the no-synthetic baseline, and the strongest gains are observed at the 50% target setting. For Random Forest, the 50% condition is especially close to the paper's reported values, with all four metrics within about 0.01 of the baseline results. The 40% condition is also broadly aligned, but with slightly lower RF scores than the paper. Logistic Regression shows a stronger deviation from the paper, with the empirical results substantially exceeding the reported values across all SMOTE settings. That difference is larger than typical run-to-run variance and likely reflects implementation or preprocessing differences rather than simulation noise alone. Overall, the experiment successfully reproduces the paper's qualitative findings and matches the Random Forest 50% condition very closely, but the Logistic Regression reproduction is not numerically faithful to the original table.
