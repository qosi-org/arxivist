"""
evaluation/metrics.py — PSNR, SSIM, LPIPS matching Table 1 / Tables 3-6.

Implements the three image-quality metrics reported throughout Sec 6
(Table 1, per-scene breakdowns in Tables 3-6, ablations in Table 2):

  - PSNR: standard 20*log10(MAX) - 10*log10(MSE), MAX=1.0 for [0,1]-range images.
  - SSIM: structural similarity, computed via `tf.image.ssim`.
  - LPIPS: learned perceptual distance; requires a pretrained network
    (`lpips-tf2` per architecture_plan.json dependencies). If unavailable,
    `lpips` degrades gracefully to `None` with a printed warning rather than
    crashing evaluation, since PSNR/SSIM alone still let a user sanity-check
    reproduction quality (see risk_assessment[5] in architecture_plan.json).

SIR reference: evaluation_protocol.metrics (confidence 0.95).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

try:
    import lpips_tf2  # type: ignore

    _LPIPS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    _LPIPS_AVAILABLE = False


class ImageMetrics:
    """PSNR / SSIM / LPIPS computation for rendered-vs-ground-truth image pairs."""

    def __init__(self) -> None:
        self._lpips_model = None
        if _LPIPS_AVAILABLE:
            self._lpips_model = lpips_tf2.LPIPS(net="alex")
        else:
            print(
                "WARNING: lpips-tf2 not installed; ImageMetrics.lpips() will return None. "
                "See architecture_plan.json risk_assessment for the documented fallback "
                "(run LPIPS in an isolated PyTorch sub-environment for evaluation only)."
            )

    def psnr(self, pred: tf.Tensor, target: tf.Tensor) -> float:
        """Peak signal-to-noise ratio in dB.

        Args:
            pred: [H, W, 3] rendered image, values in [0, 1].
            target: [H, W, 3] ground-truth image, values in [0, 1].

        Returns:
            PSNR in dB (higher is better).
        """
        assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
        mse = tf.reduce_mean(tf.square(pred - target))
        return float(-10.0 * tf.math.log(mse + 1e-10) / tf.math.log(10.0))

    def ssim(self, pred: tf.Tensor, target: tf.Tensor) -> float:
        """Structural similarity index in [0, 1] (higher is better).

        Args:
            pred: [H, W, 3] rendered image, values in [0, 1].
            target: [H, W, 3] ground-truth image, values in [0, 1].
        """
        assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
        return float(tf.image.ssim(pred[None], target[None], max_val=1.0)[0])

    def lpips(self, pred: tf.Tensor, target: tf.Tensor) -> float | None:
        """Learned Perceptual Image Patch Similarity (lower is better).

        Args:
            pred: [H, W, 3] rendered image, values in [0, 1].
            target: [H, W, 3] ground-truth image, values in [0, 1].

        Returns:
            LPIPS distance, or None if lpips-tf2 is not installed (see class
            docstring for the fallback strategy).
        """
        if self._lpips_model is None:
            return None
        assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
        # LPIPS networks conventionally expect inputs in [-1, 1].
        pred_scaled = pred[None] * 2.0 - 1.0
        target_scaled = target[None] * 2.0 - 1.0
        return float(self._lpips_model(pred_scaled, target_scaled)[0])

    def compute_all(self, pred: tf.Tensor, target: tf.Tensor, metrics: list[str]) -> dict[str, float | None]:
        """Compute a subset of {psnr, ssim, lpips} in one call, per config's evaluation.metrics."""
        out: dict[str, float | None] = {}
        if "psnr" in metrics:
            out["psnr"] = self.psnr(pred, target)
        if "ssim" in metrics:
            out["ssim"] = self.ssim(pred, target)
        if "lpips" in metrics:
            out["lpips"] = self.lpips(pred, target)
        return out

    def __repr__(self) -> str:  # noqa: D105
        return f"ImageMetrics(lpips_available={_LPIPS_AVAILABLE})"
