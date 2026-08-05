"""
data/dataset.py — Dataset loaders for the three SIR evaluation benchmarks.

Implements loaders for:
  - "Realistic Synthetic 360" / "Diffuse Synthetic 360" (Blender-rendered,
    JSON camera-pose + PNG image format popularized by the official NeRF
    release), covering the paper's synthetic evaluation (Sec 6.1).
  - "Real Forward-Facing" (LLFF-style: `poses_bounds.npy` produced by
    COLMAP), covering Sec 6.1's real-scene evaluation.

DeepVoxels loading is NOT separately implemented: the SIR's
evaluation_protocol lists it as one of the three datasets, but its data
format (paired .txt pose files) is close enough to the Blender loader's
pose-per-image contract that BlenderSyntheticDataset with an alternate pose
parser covers it; a dedicated DeepVoxels parser is left as a documented TODO
below rather than a silent omission (see NOTE in DeepVoxels stub).

SIR reference: evaluation_protocol.datasets (confidence 0.95).
Paths are never hardcoded: all directories come from `NeRFConfig.data.datadir`.
"""

from __future__ import annotations

import json
import os

import imageio.v2 as imageio
import numpy as np


class BlenderSyntheticDataset:
    """Loader for Blender-rendered synthetic scenes (transforms_{split}.json + PNGs).

    Args:
        basedir: directory containing `transforms_train.json`,
            `transforms_val.json`, `transforms_test.json` and the referenced
            image files (standard official-NeRF release layout).
        half_res: if True, downsample images 2x (400x400 instead of 800x800)
            for faster iteration; Table 1's headline numbers use full
            resolution (half_res=False).
        testskip: use every Nth image from the val/test splits (speed only,
            does not affect the train split).
    """

    def __init__(self, basedir: str, half_res: bool = False, testskip: int = 1) -> None:
        if not os.path.isdir(basedir):
            raise FileNotFoundError(
                f"Blender dataset directory not found: {basedir}. "
                f"See data/README_data.md for how to obtain it."
            )
        self.basedir = basedir
        self.half_res = half_res
        self.testskip = testskip

    def load(self) -> dict:
        """Load all splits.

        Returns:
            dict with:
              images: dict[str, np.ndarray[N,H,W,4]] per split ("train","val","test")
              poses:  dict[str, np.ndarray[N,4,4]] per split
              hwf:    (H, W, focal) tuple, shared across splits
              near:   float, the paper's synthetic bounding-cube near default (2.0)
              far:    float, the paper's synthetic bounding-cube far default (6.0)
        """
        images, poses = {}, {}
        H = W = focal = None
        for split in ("train", "val", "test"):
            meta_path = os.path.join(self.basedir, f"transforms_{split}.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            skip = 1 if split == "train" else max(self.testskip, 1)
            frames = meta["frames"][::skip]

            imgs, poses_list = [], []
            for frame in frames:
                fname = os.path.join(self.basedir, frame["file_path"] + ".png")
                img = imageio.imread(fname).astype(np.float32) / 255.0
                imgs.append(img)
                poses_list.append(np.array(frame["transform_matrix"], dtype=np.float32))

            imgs = np.stack(imgs, axis=0)
            poses_arr = np.stack(poses_list, axis=0)

            if H is None:
                H, W = imgs.shape[1], imgs.shape[2]
                camera_angle_x = float(meta["camera_angle_x"])
                focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

            if self.half_res:
                imgs = imgs[:, ::2, ::2, :]

            images[split] = imgs
            poses[split] = poses_arr

        if H is None:
            raise ValueError(f"No transforms_*.json splits found under {self.basedir}")
        if self.half_res:
            H, W, focal = H // 2, W // 2, focal / 2.0

        # Sec 6.1: synthetic scenes are bounded within a cube of side length 2 at the origin;
        # near/far here match the widely-used default for that convention (ASSUMED, see
        # configs/config.yaml `data.near` / `data.far` comments).
        return {
            "images": images,
            "poses": poses,
            "hwf": (H, W, focal),
            "near": 2.0,
            "far": 6.0,
        }

    def __repr__(self) -> str:  # noqa: D105
        return f"BlenderSyntheticDataset(basedir={self.basedir!r}, half_res={self.half_res})"


class LLFFRealDataset:
    """Loader for real, forward-facing scenes in LLFF's `poses_bounds.npy` format.

    Args:
        basedir: directory containing `poses_bounds.npy` (produced by COLMAP
            via `preprocess_llff.py`) and an `images/` (or `images_{factor}/`)
            subdirectory.
        factor: downsampling factor matching an `images_{factor}/` directory
            produced during preprocessing (paper Sec 6.1 real-scene captures).
        bd_factor: scale factor applied to the recovered scene bounds, a
            standard LLFF convention that adds a safety margin around the
            COLMAP-estimated near/far planes.
    """

    def __init__(self, basedir: str, factor: int = 8, bd_factor: float = 0.75) -> None:
        if not os.path.isfile(os.path.join(basedir, "poses_bounds.npy")):
            raise FileNotFoundError(
                f"poses_bounds.npy not found in {basedir}. Run preprocess_llff.py first "
                f"(see data/README_data.md)."
            )
        self.basedir = basedir
        self.factor = factor
        self.bd_factor = bd_factor

    def load(self) -> dict:
        """Load poses, bounds, and images for a real forward-facing scene.

        Returns:
            dict with:
              images: np.ndarray [N, H, W, 3]
              poses:  np.ndarray [N, 4, 4] camera-to-world matrices
              bds:    np.ndarray [N, 2] (near, far) COLMAP-derived bounds per view
              hwf:    (H, W, focal)
              i_test: int, index of the held-out test view (every 8th, Sec 6.1: "hold out 1/8")
        """
        poses_arr = np.load(os.path.join(self.basedir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # [N, 3, 5]: last col is (H, W, focal)
        bds = poses_arr[:, -2:]  # [N, 2]

        img_dir = os.path.join(self.basedir, f"images_{self.factor}" if self.factor > 1 else "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(self.basedir, "images")
        img_files = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        images = np.stack(
            [imageio.imread(os.path.join(img_dir, f)).astype(np.float32) / 255.0 for f in img_files],
            axis=0,
        )

        H, W, focal = poses[0, :, 4]
        H, W = images.shape[1], images.shape[2]  # trust actual loaded image size over metadata
        c2w = poses[:, :3, :4]
        bottom = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (c2w.shape[0], 1, 1))
        poses_4x4 = np.concatenate([c2w, bottom], axis=1)

        bds = bds * (1.0 / (bds.min() * self.bd_factor))  # LLFF scale-normalization convention

        # Sec 6.1: "hold out 1/8 of these for the test set".
        i_test = np.arange(images.shape[0])[:: max(images.shape[0] // 8, 1)][0]

        return {
            "images": images,
            "poses": poses_4x4,
            "bds": bds,
            "hwf": (H, W, focal),
            "i_test": int(i_test),
        }

    def __repr__(self) -> str:  # noqa: D105
        return f"LLFFRealDataset(basedir={self.basedir!r}, factor={self.factor})"


class DeepVoxelsDataset:
    """
    STUB: DeepVoxels' own pose/intrinsics text-file format was not described
    in enough structural detail in the paper body itself (the paper only
    cites the DeepVoxels dataset/paper [41] and reuses its images) to derive
    a parser purely from the SIR without inspecting that external format.

    SIR ambiguity: evaluation_protocol.datasets lists "Diffuse Synthetic
    360 (DeepVoxels)" as a benchmark (confidence 0.95 that it's used, but no
    per-field file-format spec is given in the NeRF paper text). Replace this
    stub with a real loader (mirroring the official DeepVoxels repository's
    pose-file convention) before attempting to reproduce Table 3 / DeepVoxels
    numbers.
    """

    def __init__(self, basedir: str) -> None:
        self.basedir = basedir

    def load(self) -> dict:
        raise NotImplementedError(
            "See class docstring — DeepVoxels loading requires manual implementation "
            "against the external DeepVoxels dataset format."
        )
