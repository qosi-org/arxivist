#!/usr/bin/env python
"""
inference.py — Render a single novel view (or a standard spiral fly-through
video) from a trained NeRF checkpoint, given a camera pose.

Usage:
    python inference.py --checkpoint checkpoints/lego_full/checkpoints/ckpt-25 \
        --pose data/example_pose.npy --out render.png
    python inference.py --checkpoint checkpoints/lego_full/checkpoints/ckpt-25 \
        --out flythrough.mp4     # renders the standard spiral path (no --pose)
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf

from nerf.data.rays import RayGenerator
from nerf.models.radiance_field import NeRFModel
from nerf.utils.config import NeRFConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render novel view(s) from a trained NeRF checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pose", type=str, default=None, help="Path to a 4x4 camera-to-world pose (.npy).")
    parser.add_argument("--out", type=str, default="render.png")
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--focal", type=float, default=555.5555)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument(
        "--n_spiral_frames", type=int, default=40, help="Number of frames if --pose is omitted (spiral fly-through)."
    )
    return parser.parse_args()


def spiral_pose(angle: float, radius: float = 4.0) -> np.ndarray:
    """Generate one camera-to-world pose on a circular orbit at a given angle (radians).

    A minimal stand-in for the paper's supplementary "smooth path of novel
    views" (Sec 6, "we urge readers to view our supplementary video") —
    exact camera-path generation is a visualization convenience, not part of
    the paper's reproducible methodology, so this uses a simple circular orbit.
    """
    cam_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), radius * 0.3], dtype=np.float32)
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    return c2w


def render_image(model: NeRFModel, ray_gen: RayGenerator, pose: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rays_o, rays_d = ray_gen.get_rays(args.height, args.width, args.focal, tf.constant(pose, dtype=tf.float32))
    rays_o = tf.reshape(rays_o, [-1, 3])
    rays_d = tf.reshape(rays_d, [-1, 3])
    chunks = []
    for i in range(0, rays_o.shape[0], args.chunk_size):
        out = model.render_rays(
            rays_o[i : i + args.chunk_size], rays_d[i : i + args.chunk_size], args.near, args.far, training=False
        )
        chunks.append(out["rgb_fine"])
    rgb = tf.concat(chunks, axis=0)
    return (np.clip(tf.reshape(rgb, [args.height, args.width, 3]).numpy(), 0, 1) * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    config = NeRFConfig.from_yaml(args.config)
    model = NeRFModel(config.raw)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.checkpoint).expect_partial()
    print(f"Restored checkpoint: {args.checkpoint}")

    ray_gen = RayGenerator()

    if args.pose is not None:
        pose = np.load(args.pose).astype(np.float32)
        img = render_image(model, ray_gen, pose, args)
        imageio.imwrite(args.out, img)
        print(f"Rendered single view to {args.out}")
    else:
        frames = []
        for i in range(args.n_spiral_frames):
            angle = 2 * np.pi * i / args.n_spiral_frames
            pose = spiral_pose(angle)
            frames.append(render_image(model, ray_gen, pose, args))
            print(f"Rendered spiral frame {i + 1}/{args.n_spiral_frames}")
        imageio.mimwrite(args.out, frames, fps=15)
        print(f"Rendered {args.n_spiral_frames}-frame fly-through to {args.out}")


if __name__ == "__main__":
    main()
