# Domain: Neuroscience — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **Neuroscience**.

---

## Signal processing standards

**Always filter before epoching, never after:**
```python
import mne

def preprocess_eeg(raw, config):
    raw = raw.copy()
    raw.filter(
        l_freq=config.data.bandpass_low_hz,
        h_freq=config.data.bandpass_high_hz,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"}
    )
    epochs = mne.Epochs(
        raw,
        events=events,
        tmin=config.data.epoch_tmin,
        tmax=config.data.epoch_tmax,
        baseline=(config.data.baseline_tmin, 0),
        preload=True
    )
    return epochs
```

**SNN time constant derivation — always derive from physical constants:**
```python
def compute_beta(tau_mem_ms: float, dt_ms: float) -> float:
    """
    Compute membrane decay factor from time constant.
    beta = exp(-dt / tau_mem) — NEVER hardcode beta directly.
    """
    return float(torch.exp(torch.tensor(-dt_ms / tau_mem_ms)))
```

## Cross-validation — mandatory for all neuroscience models

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Leave-one-session-out or k-fold — always specify in config
def evaluate_decoder(model, X, y, config):
    cv = StratifiedKFold(
        n_splits=config.evaluation.n_folds,
        shuffle=True,
        random_state=config.hardware.seed
    )
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=config.evaluation.metric)
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores.tolist(),
        "chance": 1.0 / len(np.unique(y))
    }
```

## Multiple comparisons correction

```python
from statsmodels.stats.multitest import multipletests

def correct_pvalues(pvalues: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Always correct for multiple comparisons in neuroscience."""
    _, corrected, _, _ = multipletests(pvalues, method=method)
    return corrected
```

## Requirements

```
mne>=1.6.0                  # EEG/MEG processing
snntorch>=0.8.0             # spiking neural networks
nilearn>=0.10.0             # fMRI analysis
nibabel>=5.0.0              # neuroimaging file formats
scikit-learn>=1.3.0         # decoding and CV
statsmodels>=0.14.0         # statistical tests
torch>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
```

## What Must NOT be done

- Do NOT apply any signal processing after epoching that uses data from outside the epoch
- Do NOT use test data to select preprocessing parameters (no nested CV violation)
- Do NOT ignore chance level — always report and compare against it
- Do NOT hardcode SNN time constants as `beta` — always derive from `tau_mem` and `dt`
- Do NOT use subject-level split as trial-level split — subjects must not span train/test
