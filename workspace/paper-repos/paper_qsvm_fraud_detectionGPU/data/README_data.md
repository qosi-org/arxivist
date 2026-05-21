# Dataset: Kaggle Credit Card Fraud Detection

**Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Paper reference:** Section IV-A

## Dataset Stats (from paper)
- Total transactions: **284,807** (2-day window)
- Fraudulent: **492** (0.172%)
- Features: V1–V28 (PCA-derived), Time, Amount → **30 features**
- Target: `Class` (0 = legitimate, 1 = fraud)

## Download Options

### Option 1: Kaggle CLI (recommended)
```bash
# Install kaggle package and configure credentials
pip install kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

python scripts/download_data.py --output-dir data/raw/
```

### Option 2: Manual Download
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download**
3. Unzip and place `creditcard.csv` at `data/raw/creditcard.csv`

### Option 3: Kaggle CLI directly
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

## Expected File Location
```
data/
└── raw/
    └── creditcard.csv   ← place here
```

## Debug Mode (no Kaggle account needed)
For quick testing without the full dataset, the debug config uses only 300 samples.
You still need `creditcard.csv` — but any valid credit card fraud CSV with the same
schema (columns V1–V28, Time, Amount, Class) will work.
