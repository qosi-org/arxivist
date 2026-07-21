# Data

## MNIST / Binary MNIST (VAE experiment)
Downloaded automatically by `torchvision` when you run training:
```bash
python train.py --experiment vae
```
Data is saved to `data/mnist/` by default (configurable via `configs/vae.yaml`).

## Synthetic GSL Dataset (GSL experiment)
Generated on-the-fly from the community graph structure (Figure 5).
To pre-generate and cache all five θ* settings:
```bash
python scripts/generate_gsl_data.py --output_dir data/gsl/
```
This saves pickled `(X, Y)` tensors to `data/gsl/gsl_theta{θ*}_seed42.pkl`.

## Atari ROMs (RL experiment)
Atari ROMs must be installed separately via `ale-py`:
```bash
pip install ale-py
python -m ale_py.roms --import-all
```
Or accept the ALE licence and use AutoROM:
```bash
pip install autorom
AutoROM --accept-license
```
Environments used: `BreakoutNoFrameskip-v4`, `SeaquestNoFrameskip-v4`.
