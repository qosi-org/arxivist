# Domain: Neuroscience — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Neuroscience**.

---

## Framework selection

**SpikingJelly or snnTorch** — for spiking neural network papers.
**MNE-Python** — for EEG/MEG preprocessing pipelines. Never reimplement signal processing
primitives that MNE provides (filtering, epoching, ICA, source localisation).
**Nilearn / Nibabel** — for fMRI analysis.
**PyTorch** — for deep learning decoding models.
**Brian2** — for biophysically detailed neuron simulations (Hodgkin-Huxley level).

## Standard neuroscience module patterns

**LIF spiking neuron layer (snnTorch):**
```python
import snntorch as snn
lif = snn.Leaky(beta=config.model.tau_ratio, spike_grad=surrogate_grad)
# beta = exp(-dt / tau_mem) — always compute from time constants, not hardcode
```

**EEG decoding pipeline:**
```
Raw_EEG [trials, channels, time] →
  Bandpass_filter →
  Epoch [window_size, overlap] →
  Feature_extraction (PSD / CSP / raw) →
  Decoder (LDA / EEGNet / Transformer)
```

**EEGNet (standard compact CNN for EEG):**
```
[B, 1, C, T] → DepthwiseConv2D(C, 1) → BatchNorm → ELU → AvgPool →
SeparableConv2D → BatchNorm → ELU → AvgPool → Flatten → Dense
```

**fMRI GLM pipeline:**
```
BOLD_timeseries [T, voxels] →
  HRF_convolution →
  Design_matrix [T, regressors] →
  OLS_fit → Beta_map [voxels, regressors] →
  Contrast → t/z_map → Threshold
```

## Config schema for neuroscience papers

```yaml
data:
  signal_type: eeg              # eeg, meg, fmri, lfp, spikes
  sampling_rate_hz: 250
  n_channels: 64
  epoch_length_s: 1.0
  epoch_overlap: 0.5
  bandpass_low_hz: 0.5
  bandpass_high_hz: 40.0
  baseline_correction: true

model:
  tau_mem_ms: 20.0              # membrane time constant (SNN)
  tau_syn_ms: 5.0               # synaptic time constant (SNN)
  threshold: 1.0                # spike threshold
  n_latent: 32                  # latent space dim for VAE models

evaluation:
  n_folds: 10                   # always cross-validate in neuroscience
  chance_level: null            # computed from class distribution
  multiple_comparisons: fdr_bh  # Benjamini-Hochberg FDR correction
```
