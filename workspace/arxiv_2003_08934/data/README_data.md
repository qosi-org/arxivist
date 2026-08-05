# Data — How to Obtain the NeRF Benchmark Datasets

This paper (arXiv:2003.08934) evaluates on three benchmarks (Sec 6.1). None
of the raw data is redistributed in this repo; here's how to get each one.

## 1. Realistic Synthetic 360 (headline results, Table 1/4)

8 pathtraced Blender scenes: chair, drums, ficus, hotdog, lego, materials, mic, ship.

- **Get it**: Download `nerf_synthetic.zip` from the official project page
  (https://www.matthewtancik.com/nerf, linked Google Drive) and extract to:
  ```
  data/raw/nerf_synthetic/<scene>/
      transforms_train.json
      transforms_val.json
      transforms_test.json
      train/, val/, test/   (PNG images referenced by the JSON files)
  ```
- **Quick smoke test alternative**: `python data/download.py --dest ./data/raw --scene tiny_demo`
  fetches the small single-scene `.npz` used by the official TinyNeRF notebook — good for
  validating the pipeline runs end-to-end, NOT for reproducing Table 1 numbers.
- **Loader**: `nerf.data.dataset.BlenderSyntheticDataset`

## 2. Real Forward-Facing (Table 1/5)

8 real, handheld, forward-facing captures (5 from the LLFF paper, 3 captured by the NeRF authors).

- **Get it (LLFF-provided scenes)**: download from the LLFF project repo
  (https://github.com/Fyusion/LLFF) — `fern`, `flower`, `fortress`, `horns`, `leaves`, `orchids`,
  `room`, `trex` and similar forward-facing sets are commonly distributed there.
- **Get it (your own captures)**: place raw images in `data/raw/<scene>/images/`, then run:
  ```
  python preprocess_llff.py --scenedir data/raw/<scene> --factor 8
  ```
  This runs COLMAP feature extraction + matching + sparse mapping (Appendix B: "use the COLMAP
  structure-from-motion package to estimate these parameters for real data"). See the printed
  note at the end of that script regarding the final `poses_bounds.npy` packaging step, which
  uses the official LLFF conversion utility (not reimplemented in this repo).
- **Expected final layout**:
  ```
  data/raw/<scene>/
      poses_bounds.npy
      images/  (or images_8/, images_4/, ... for pre-downsampled versions)
  ```
- **Loader**: `nerf.data.dataset.LLFFRealDataset`

## 3. Diffuse Synthetic 360 (DeepVoxels, Table 1/3)

4 simple-geometry, purely diffuse objects (cube, cube, vase, pedestal, chair) rendered by the
DeepVoxels authors (Sitzmann et al., cited as [41] in the paper).

- **Get it**: See the DeepVoxels project page/repo
  (https://github.com/vsitzmann/deepvoxels) for the dataset download link and license terms.
- **Loader**: `nerf.data.dataset.DeepVoxelsDataset` is currently a **STUB** — the paper does not
  describe DeepVoxels' own pose/intrinsics file format in enough detail to derive a parser purely
  from the paper text (see the class docstring's SIR-ambiguity note). Implement this loader
  against the official DeepVoxels format before attempting to reproduce Table 3.

## Verifying your setup

Once at least one Blender-synthetic scene is in place:
```
python train.py --config configs/config_debug.yaml --datadir data/raw/nerf_synthetic/lego \
    --expname smoke_test --debug --dry-run
```
This builds the full pipeline (data -> model -> trainer) without training, to confirm paths and
shapes are correct before committing to a multi-hour run.
