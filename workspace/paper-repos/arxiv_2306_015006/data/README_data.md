# GUE Benchmark Data

All downloads go through `data/download.py` (HF `datasets` API). Data lands under
`data/GUE/` and is git-ignored.

## Download
```bash
python data/download.py --task promoter_detection   # one task (all its subsets)
python data/download.py                             # all 7 tasks / 28 datasets
```

## Layout
```
data/GUE/{task}/{subset}/{train,dev,test}.csv   # columns: sequence,label
```

## Tasks (DNABERT-2 Table 1 / Table 12)
| Task | Subsets | Metric | Classes |
|---|---|---|---|
| promoter_detection | all, notata, tata | MCC | 2 |
| core_promoter_detection | all, notata, tata | MCC | 2 |
| tf_human | 0–4 | MCC | 2 |
| tf_mouse | 0–4 | MCC | 2 |
| epigenetic_marks | H3, H3K14ac, … (10) | MCC | 2 |
| splice | reconstructed | MCC | 3 |
| covid_variant | covid | F1 | 9 |

## Note
If the HF mirror config names differ from the paper's, consult the official
[MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2) GUE release and
place CSVs in the layout above; the loader reads them directly.
