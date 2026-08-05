"""
data/transforms.py — tf.data pipeline helpers for building random ray batches.

This module contains pure data-engineering plumbing (no paper equations):
it flattens a loaded scene's (images, poses) into a shuffled stream of
(ray_o, ray_d, target_rgb) triples for `training.ray_batch_size`-sized
random ray-batch sampling, matching Appendix A: "At each optimization
iteration, we randomly sample a batch of camera rays from the set of all
pixels in the dataset."

SIR reference: training_pipeline.batch_size (confidence 0.95: "4096 rays").
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from nerf.data.rays import RayGenerator


def build_ray_dataset(
    images: np.ndarray,
    poses: np.ndarray,
    hwf: tuple[int, int, float],
    batch_size: int,
    shuffle_buffer: int = 100_000,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Flatten (images, poses) into a shuffled tf.data.Dataset of ray batches.

    Args:
        images: [N, H, W, C] (C=3 or 4; if 4, callers are responsible for
            alpha-compositing onto a background before use as `data.white_background`
            dictates — see models/radiance_field.py `white_background`).
        poses: [N, 4, 4] camera-to-world matrices.
        hwf: (H, W, focal) shared camera intrinsics.
        batch_size: rays per batch (Appendix A default: 4096).
        shuffle_buffer: tf.data shuffle buffer size.
        seed: optional seed for the shuffle op (reproducibility).

    Returns:
        A tf.data.Dataset yielding dicts with keys `rays_o`, `rays_d` (each
        [batch_size, 3]) and `target_rgb` ([batch_size, 3]).
    """
    assert images.ndim == 4, f"Expected images [N,H,W,C], got shape {images.shape}"
    assert poses.shape[1:] == (4, 4), f"Expected poses [N,4,4], got shape {poses.shape}"

    H, W, focal = hwf
    ray_gen = RayGenerator()

    all_rays_o, all_rays_d, all_rgb = [], [], []
    for img, pose in zip(images, poses):
        rays_o, rays_d = ray_gen.get_rays(H, W, focal, tf.constant(pose, dtype=tf.float32))
        rgb = img[..., :3]
        all_rays_o.append(tf.reshape(rays_o, [-1, 3]))
        all_rays_d.append(tf.reshape(rays_d, [-1, 3]))
        all_rgb.append(tf.reshape(tf.constant(rgb, dtype=tf.float32), [-1, 3]))

    rays_o = tf.concat(all_rays_o, axis=0)
    rays_d = tf.concat(all_rays_d, axis=0)
    target_rgb = tf.concat(all_rgb, axis=0)

    ds = tf.data.Dataset.from_tensor_slices(
        {"rays_o": rays_o, "rays_d": rays_d, "target_rgb": target_rgb}
    )
    ds = ds.shuffle(min(shuffle_buffer, rays_o.shape[0]), seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds
