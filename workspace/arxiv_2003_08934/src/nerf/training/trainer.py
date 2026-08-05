"""
training/trainer.py — Main training loop.

Implements Appendix A ("Implementation Details") of arXiv:2003.08934:
  - Adam optimizer, lr 5e-4 -> 5e-5 via exponential decay (ASSUMED exact
    decay formula, see configs/config.yaml comment).
  - Random ray-batch sampling of `training.ray_batch_size` rays per step.
  - Periodic checkpointing and metric logging.

This module contains no equations of its own beyond the LR schedule; the
forward pass and loss live in models/radiance_field.py and training/losses.py
respectively.
"""

from __future__ import annotations

import os
import time

import tensorflow as tf

from nerf.models.radiance_field import NeRFModel
from nerf.training.losses import PhotometricMSELoss


class Trainer:
    """Owns the optimizer, checkpoint manager, and the train_step/fit loop.

    Args:
        model: a constructed NeRFModel.
        config: the validated config dict (`NeRFConfig.raw`).
        dataset: dict as returned by BlenderSyntheticDataset.load() /
            LLFFRealDataset.load() (must contain "hwf", "near"/"far" or
            "bds", and per-split "images"/"poses").
        log_dir: directory for TensorBoard summaries and checkpoints.
    """

    def __init__(self, model: NeRFModel, config: dict, dataset: dict, log_dir: str) -> None:
        self.model = model
        self.config = config
        self.dataset = dataset
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        train_cfg = config["training"]
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=train_cfg["learning_rate_init"],
            decay_steps=train_cfg["num_training_steps"],
            # ASSUMED (SIR ambiguity, confidence 0.6): decay_rate chosen so the schedule reaches
            # exactly learning_rate_final at step == num_training_steps.
            decay_rate=train_cfg["learning_rate_final"] / train_cfg["learning_rate_init"],
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=train_cfg["adam_beta1"],
            beta_2=train_cfg["adam_beta2"],
            epsilon=train_cfg["adam_epsilon"],
        )
        self.loss_fn = PhotometricMSELoss()

        self.near = dataset.get("near")
        self.far = dataset.get("far")
        if self.near is None or self.far is None:
            # LLFF real scenes: derive scalar near/far from per-view COLMAP bounds (Appendix C).
            bds = dataset["bds"]
            self.near = float(bds.min()) * 0.9
            self.far = float(bds.max()) * 1.1

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, os.path.join(log_dir, "checkpoints"), max_to_keep=5
        )
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "tb"))

    def print_training_summary(self, num_train_rays: int) -> None:
        """Print a training summary before the loop starts (required by Stage 4 spec)."""
        # Keras layers build their weights lazily on first call; run one dummy
        # forward pass so trainable_variables (and thus the param count below)
        # is populated even before the first real train_step.
        if not self.model.trainable_variables:
            dummy_o = tf.zeros([1, 3])
            dummy_d = tf.ones([1, 3])
            self.model.render_rays(dummy_o, dummy_d, self.near, self.far, training=False)

        num_params = sum(int(tf.size(v)) for v in self.model.trainable_variables)
        steps = self.config["training"]["num_training_steps"]
        batch = self.config["training"]["ray_batch_size"]
        print("=" * 70)
        print(f"NeRF Trainer — {self.model!r}")
        print(f"  Trainable parameters : {num_params:,}")
        print(f"  Training rays total  : {num_train_rays:,}")
        print(f"  Ray batch size       : {batch:,}")
        print(f"  Planned steps        : {steps:,} (~{steps * batch / max(num_train_rays,1):.1f} epochs)")
        print(f"  Near / Far           : {self.near:.3f} / {self.far:.3f}")
        print(f"  Log dir              : {self.log_dir}")
        print("=" * 70)

    @tf.function
    def train_step(
        self, batch_rays_o: tf.Tensor, batch_rays_d: tf.Tensor, target_rgb: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        """One optimization step: forward render -> Eq. 6 loss -> Adam update.

        Args:
            batch_rays_o: [B, 3] ray origins.
            batch_rays_d: [B, 3] ray directions.
            target_rgb: [B, 3] ground-truth pixel colors.

        Returns:
            dict of scalar metrics for this step (total_loss, coarse_loss,
            fine_loss, psnr).
        """
        with tf.GradientTape() as tape:
            outputs = self.model.render_rays(
                batch_rays_o, batch_rays_d, self.near, self.far, training=True
            )
            loss_dict = self.loss_fn(outputs["rgb_coarse"], outputs["rgb_fine"], target_rgb)
        grads = tape.gradient(loss_dict["total_loss"], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_dict

    def save_checkpoint(self, step: int) -> str:
        """Save a checkpoint and return its path."""
        path = self.ckpt_manager.save(checkpoint_number=step)
        return path

    def fit(self, train_ray_dataset: tf.data.Dataset, num_steps: int | None = None) -> None:
        """Run the full training loop.

        Args:
            train_ray_dataset: an infinite tf.data.Dataset yielding dicts
                with `rays_o`, `rays_d`, `target_rgb` (see
                data/transforms.py::build_ray_dataset), already batched to
                `training.ray_batch_size`.
            num_steps: override for `training.num_training_steps`.
        """
        train_cfg = self.config["training"]
        num_steps = num_steps if num_steps is not None else train_cfg["num_training_steps"]
        log_every = train_cfg["log_every"]
        ckpt_every = train_cfg["checkpoint_every"]

        start_time = time.time()
        step = 0
        for batch in train_ray_dataset.take(num_steps):
            loss_dict = self.train_step(batch["rays_o"], batch["rays_d"], batch["target_rgb"])
            step += 1

            if step % log_every == 0 or step == 1:
                elapsed = time.time() - start_time
                print(
                    f"[step {step:>7}/{num_steps}] "
                    f"loss={float(loss_dict['total_loss']):.5f} "
                    f"psnr={float(loss_dict['psnr']):.3f} "
                    f"({elapsed:.1f}s elapsed)"
                )
                with self.summary_writer.as_default():
                    for key, val in loss_dict.items():
                        tf.summary.scalar(f"train/{key}", val, step=step)

            if step % ckpt_every == 0:
                path = self.save_checkpoint(step)
                print(f"[step {step:>7}] checkpoint saved: {path}")

        # Always save a final checkpoint at the end of training.
        final_path = self.save_checkpoint(step)
        print(f"Training complete. Final checkpoint: {final_path}")

    def __repr__(self) -> str:  # noqa: D105
        return f"Trainer(log_dir={self.log_dir!r}, near={self.near}, far={self.far})"
