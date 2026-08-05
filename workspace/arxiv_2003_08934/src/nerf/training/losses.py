"""
training/losses.py — Combined coarse+fine photometric MSE loss.

Implements Eq. 6 (Sec 5.3) of arXiv:2003.08934:

    L = sum_{r in R} [ ||C_hat_c(r) - C(r)||_2^2 + ||C_hat_f(r) - C(r)||_2^2 ]

where R is the set of rays in the current batch, C(r) is the ground-truth
pixel color, and C_hat_c / C_hat_f are the coarse/fine network's rendered
colors for that ray. Both coarse and fine terms are minimized jointly even
though only C_hat_f is used for the final rendering, "so that the weight
distribution from the coarse network can be used to allocate samples in the
fine network" (Sec 5.3).

SIR reference: mathematical_spec "Combined coarse+fine photometric MSE loss"
(confidence 0.95, Eq. 6).
"""

from __future__ import annotations

import tensorflow as tf


class PhotometricMSELoss:
    """Sum-of-squared-error loss over coarse and fine rendered rays (Eq. 6)."""

    def __call__(
        self, rgb_coarse: tf.Tensor, rgb_fine: tf.Tensor, target_rgb: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        """Compute the combined coarse+fine loss and a PSNR diagnostic.

        Args:
            rgb_coarse: [N_rays, 3], C_hat_c(r) from the coarse network.
            rgb_fine: [N_rays, 3], C_hat_f(r) from the fine network (or the
                same coarse output if hierarchical sampling is disabled;
                see NeRFModel.render_rays).
            target_rgb: [N_rays, 3], ground-truth pixel colors C(r).

        Returns:
            dict with:
              total_loss:  scalar, Eq. 6 (coarse + fine squared-error, summed
                           over the ray batch)
              coarse_loss: scalar, the coarse-only term
              fine_loss:   scalar, the fine-only term
              psnr:        scalar, PSNR computed from fine_loss's mean-squared-error
                           (standard NeRF diagnostic; not itself part of Eq. 6)
        """
        assert rgb_coarse.shape == target_rgb.shape, (
            f"rgb_coarse {rgb_coarse.shape} must match target_rgb {target_rgb.shape}"
        )
        assert rgb_fine.shape == target_rgb.shape, (
            f"rgb_fine {rgb_fine.shape} must match target_rgb {target_rgb.shape}"
        )

        # Eq. 6: ||.||_2^2 per ray, summed over the batch R.
        coarse_loss = tf.reduce_sum(tf.reduce_sum(tf.square(rgb_coarse - target_rgb), axis=-1))
        fine_loss = tf.reduce_sum(tf.reduce_sum(tf.square(rgb_fine - target_rgb), axis=-1))
        total_loss = coarse_loss + fine_loss

        # PSNR diagnostic (mean, not summed, per-pixel MSE -> dB); standard NeRF training-log metric.
        mse_fine = tf.reduce_mean(tf.square(rgb_fine - target_rgb))
        psnr = -10.0 * tf.math.log(mse_fine + 1e-10) / tf.math.log(10.0)

        return {
            "total_loss": total_loss,
            "coarse_loss": coarse_loss,
            "fine_loss": fine_loss,
            "psnr": psnr,
        }

    def __repr__(self) -> str:  # noqa: D105
        return "PhotometricMSELoss()"
