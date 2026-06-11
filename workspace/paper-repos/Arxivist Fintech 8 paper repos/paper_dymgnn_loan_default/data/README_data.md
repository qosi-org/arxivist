# Data Setup — Freddie Mac SFLL Dataset

## Data Source

This paper uses the **Single-Family Loan-Level (SFLL) dataset** from the Federal
Home Loan Mortgage Corporation (Freddie Mac).

**Citation**: FreddieMac (2022). Single family loan-level dataset.
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

## How to Obtain

1. Visit https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Register and agree to the data license terms (free for research use)
3. Download loan-level data for origination years **2009 and 2010**
4. You need both the **Origination** and **Monthly Performance** files

## Data Periods Used (Section 4.2)

| Split     | Period                   | Description                     |
|-----------|--------------------------|----------------------------------|
| Training  | Jan 2012 – Jun 2013      | 18 months, 13 rolling windows    |
| Test      | Jul 2013 – Dec 2013      | 6 months, 1 window               |
| Origination | 2009–2010              | Source loan cohort               |

## Features Used (Table 1)

### Static (acquired at origination)
`fico`, `if_fthb`, `mi_pct`, `cnt_units`, `if_prim_res`, `dti`, `ltv`,
`if_corr`, `if_sf`, `if_purc`, `cnt_borr`, `if_sc`

### Dynamic (change monthly)
`current_upb`, `if_delq_sts`, `mths_remng`, `current_int_rt`

### Target
`default` — 90+ days arrears within next 12 months

## Preprocessing (Section 4.1)

- **Outliers**: Capped at 1st and 99th percentile
- **Null values**: Median imputation
- **Numerical features**: Min-max normalization
- **Categorical features**: Binary encoding (one-vs-rest)

## Expected Directory Structure

After downloading and pre-processing, place files as:
```
data/freddie_mac/
├── train/
│   ├── window_01/
│   │   ├── snapshot_1_features.parquet
│   │   ├── snapshot_1_area_adj.npz
│   │   ├── snapshot_1_company_adj.npz
│   │   ├── snapshot_1_double_adj.npz
│   │   ├── ...  (snapshots 2-6)
│   │   └── labels.parquet
│   ├── window_02/ ... window_13/
└── test/
    └── window_01/
        └── ...
```

## Network Construction (Section 4.2)

- **Area layer**: Two borrowers connected if their zip codes share the same first 2 digits
- **Company layer**: Two borrowers connected if they used the same mortgage provider
- **Double layer**: Supra adjacency matrix combining both layers;
  interlayer edges connect each borrower's node to their twin in the other layer

## Synthetic Fallback

If data is unavailable, the `FreddieDataset` class **automatically generates
synthetic random data** for pipeline testing. Results on synthetic data will
not match the paper but are sufficient for:
- Validating the full pipeline
- Running the Jupyter notebook
- Debugging model architecture
- Unit testing

Simply run `python train.py --config configs/config.yaml --debug` — the
fallback activates automatically.

## Network Statistics (Table 2)

| Set        | Single-area nodes | Single-area edges | Double nodes | Double edges |
|------------|:-----------------:|:-----------------:|:------------:|:------------:|
| Training   | 148,520           | 16,368,244        | 297,040      | 108,151,460  |
| Validation | 82,180            | 4,725,842         | 164,360      | 32,625,866   |
| Test       | 96,490            | 6,761,051         | 192,980      | 45,358,308   |
