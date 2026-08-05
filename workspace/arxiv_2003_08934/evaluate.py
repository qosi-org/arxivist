#!/usr/bin/env python
"""
evaluate.py — Evaluation entrypoint: renders held-out test views and computes
PSNR/SSIM/LPIPS, reproducing Table 1 / per-scene Tables 3-6 of arXiv:2003.08934.

Usage:
    python evaluate.py --config configs/config.yaml \
        --checkpoint checkpoints/lego_full/checkpoints/ckpt-25 \
        --datadir /data/lego --out_dir results/lego_full
"""

from __future__ import annotations

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf

from nerf.data.dataset import BlenderSyntheticDataset, LLFFRealDataset
from nerf.data.rays import RayGenerator
from nerf.evaluation.metrics import ImageMetrics
from nerf.models.radiance_field import NeRFModel
from nerf.utils.config import NeRFConfig, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NeRF checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/")
    return parser.parse_args()


def render_full_image(
    model: NeRFModel, ray_gen: RayGenerator, pose: np.ndarray, hwf: tuple, near: float, far: float, chunk_size: int
) -> np.ndarray:
    """Render one full test image by chunking rays through render_rays (memory-bounded)."""
    H, W, focal = hwf
    rays_o, rays_d = ray_gen.get_rays(H, W, focal, tf.constant(pose, dtype=tf.float32))
    rays_o = tf.reshape(rays_o, [-1, 3])
    rays_d = tf.reshape(rays_d, [-1, 3])

    chunks = []
    for i in range(0, rays_o.shape[0], chunk_size):
        out = model.render_rays(
            rays_o[i : i + chunk_size], rays_d[i : i + chunk_size], near, far, training=False
        )
        chunks.append(out["rgb_fine"])
    rgb = tf.concat(chunks, axis=0)
    return tf.reshape(rgb, [H, W, 3]).numpy()


def main() -> None:
    args = parse_args()
    config = NeRFConfig.from_yaml(args.config).merge_overrides({"data": {"datadir": args.datadir}})
    set_global_seed(config.get("training", "seed", 0))

    dataset_type = config["data"]["dataset_type"]
    if dataset_type == "blender":
        loaded = BlenderSyntheticDataset(
            args.datadir, half_res=config["data"]["half_res"], testskip=config["data"]["testskip"]
        ).load()
        test_images, test_poses = loaded["images"]["test"], loaded["poses"]["test"]
        near, far = loaded["near"], loaded["far"]
    elif dataset_type == "llff":
        loaded = LLFFRealDataset(args.datadir).load()
        i_test = loaded["i_test"]
        test_images = loaded["images"][i_test : i_test + 1]
        test_poses = loaded["poses"][i_test : i_test + 1]
        near, far = float(loaded["bds"].min()) * 0.9, float(loaded["bds"].max()) * 1.1
    else:
        raise ValueError(f"Unsupported dataset_type for evaluation: {dataset_type}")

    model = NeRFModel(config.raw)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.checkpoint).expect_partial()
    print(f"Restored checkpoint: {args.checkpoint}")

    ray_gen = RayGenerator()
    metrics = ImageMetrics()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for idx in range(test_images.shape[0]):
        pred = render_full_image(
            model, ray_gen, test_poses[idx], loaded["hwf"], near, far, config["evaluation"]["chunk_size"]
        )
        target = test_images[idx][..., :3]

        result = metrics.compute_all(
            tf.constant(pred, dtype=tf.float32),
            tf.constant(target, dtype=tf.float32),
            config["evaluation"]["metrics"],
        )
        result["index"] = idx
        all_results.append(result)
        print(f"[test {idx:>4}] " + " ".join(f"{k}={v:.4f}" for k, v in result.items() if k != "index" and v is not None))

        if config["evaluation"]["save_rendered_images"]:
            imageio.imwrite(
                os.path.join(args.out_dir, f"test_{idx:04d}.png"), (np.clip(pred, 0, 1) * 255).astype(np.uint8)
            )

    summary = {
        k: float(np.mean([r[k] for r in all_results if r.get(k) is not None]))
        for k in config["evaluation"]["metrics"]
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_image": all_results, "mean": summary}, f, indent=2)
    print(f"Mean metrics: {summary}")
    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
