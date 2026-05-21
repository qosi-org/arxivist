# Domain: Neuroscience — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Neuroscience**.

---

## Architecture extraction — Neuroscience-specific rules

**Neural decoding models:** extract:
- Signal type: EEG, MEG, fMRI (BOLD), LFP, spike trains, calcium imaging
- Temporal resolution (sampling rate in Hz) and spatial resolution
- Preprocessing pipeline: filtering (bandpass range), epoching (window size, overlap),
  artefact rejection method, baseline correction
- Feature extraction: raw signals, frequency bands (delta/theta/alpha/beta/gamma),
  time-frequency (wavelet, STFT), or learned features

**Spiking Neural Networks (SNNs):** extract:
- Neuron model: LIF (Leaky Integrate-and-Fire), Izhikevich, Hodgkin-Huxley
- Synaptic model: conductance-based, current-based
- Learning rule: STDP, surrogate gradient, online vs offline
- Time constants (τ_mem, τ_syn) — critical parameters often in supplementary material
- Coding scheme: rate coding, temporal coding, population coding

**fMRI / neuroimaging papers:** extract:
- Preprocessing pipeline: motion correction, spatial normalisation, smoothing kernel FWHM
- GLM design matrix structure (regressors, HRF model)
- ROI definition method or whole-brain approach
- Multiple comparisons correction (FWE, FDR, cluster threshold)

**Connectome / graph models:** extract:
- Node definition: brain region atlas used (AAL, Desikan, Schaefer)
- Edge definition: functional connectivity (correlation, coherence) or structural (DTI tractography)
- Adjacency matrix construction: threshold, binarise, or weighted

---

## Mathematical spec — Neuroscience-specific rules

**SNN papers:** always extract the membrane potential dynamics equation:
```
τ_mem dV/dt = -(V - V_rest) + R * I(t)   [LIF model]
```
And the spike emission rule and reset condition.

**Decoding papers:** extract the decoder objective — regression (for continuous variables
like position or movement direction) vs classification (for discrete states).

**Dimensionality reduction papers:** extract the latent space objective (VAE ELBO,
contrastive loss, reconstruction loss) and the dimensionality of the latent space.

---

## Evaluation — Neuroscience-specific rules

Extract:
- Decoding accuracy or R² for continuous decoding (position, velocity)
- Bits per spike (for information-theoretic analyses)
- Cross-validated R² (mandatory — overfitting is severe with small neuroscience datasets)
- Comparison against chance level (always reported in neuroscience)
- Statistical test used and correction for multiple comparisons
- Number of subjects/sessions/trials (sample sizes are often very small — N=5 is common)
