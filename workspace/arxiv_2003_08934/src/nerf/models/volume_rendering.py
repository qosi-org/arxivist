"""
models/volume_rendering.py — Differentiable volume rendering + hierarchical sampling.

Implements Sec 4 and Sec 5.2 of arXiv:2003.08934:

  - VolumeRenderer.composite: the discrete quadrature approximation (Eq. 3)
    of the continuous volume-rendering integral (Eq. 1), i.e. classical alpha
    compositing with alpha_i = 1 - exp(-sigma_i * delta_i).
  - HierarchicalSampler.sample_pdf: inverse-transform sampling (Sec 5.2,
    Eq. 5) that turns the coarse network's per-sample weights into a
    piecewise-constant PDF and draws Nf new samples biased toward
    high-density regions, for the fine network pass.

SIR reference: mathematical_spec "Volume rendering integral" (0.95),
"Discrete quadrature approximation of C(r)" (0.95), "Coarse-network ray-color
weights / hierarchical sampling PDF" (0.9); architecture modules
"VolumeRenderer" (0.95), "HierarchicalSampler" (0.9).
"""

from __future__ import annotations

import tensorflow as tf

_EPS = 1e-10


class VolumeRenderer:
    """Discrete volume-rendering compositor (Eq. 1-3)."""

    def composite(
        self,
        rgb: tf.Tensor,
        sigma: tf.Tensor,
        z_vals: tf.Tensor,
        rays_d: tf.Tensor,
        raw_noise_std: float = 0.0,
        white_background: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Composite per-sample (rgb, sigma) along each ray into a pixel color.

        Implements:
            delta_i = t_{i+1} - t_i                              (Sec 4, below Eq. 3)
            alpha_i = 1 - exp(-sigma_i * delta_i)                 (Eq. 3, reduces to alpha compositing)
            T_i     = exp(-sum_{j<i} sigma_j * delta_j)           (Eq. 3)
            w_i     = T_i * alpha_i                                (Eq. 3 / Eq. 5)
            C_hat(r)= sum_i w_i * c_i                              (Eq. 3)

        Args:
            rgb: [N_rays, N_samples, 3], per-sample color from the MLP.
            sigma: [N_rays, N_samples, 1] or [N_rays, N_samples], per-sample
                density from the MLP (already ReLU-rectified, >= 0).
            z_vals: [N_rays, N_samples], the t_i sample depths along each ray.
            rays_d: [N_rays, 3], unnormalized ray direction (its norm scales
                delta_i into real-world distance, standard NeRF convention).
            raw_noise_std: std of zero-mean Gaussian noise added to raw sigma
                before compositing (Appendix A: real-scene-only regularizer,
                0.0 disables it). Applied here rather than in the MLP so the
                MLP itself stays a pure function.
            white_background: if True, composite remaining ray "acc" onto a
                white background (used for the synthetic white-background
                Blender scenes, Sec 6.1).

        Returns:
            dict with:
              rgb_map:   [N_rays, 3]      composited color C_hat(r), Eq. 3
              weights:   [N_rays, N_samples]  per-sample weights w_i, Eq. 5
              depth_map: [N_rays]          weighted expected depth
              acc_map:   [N_rays]          total accumulated weight (opacity)
        """
        assert rgb.shape[-1] == 3, f"Expected rgb last dim 3, got {rgb.shape[-1]}"
        sigma = tf.squeeze(sigma, axis=-1) if sigma.shape[-1] == 1 else sigma
        assert z_vals.shape.rank == 2, f"Expected z_vals rank 2 [N_rays,N_samples], got {z_vals.shape}"

        # delta_i = t_{i+1} - t_i ; last interval extended to a large constant (standard convention
        # for an unbounded far sample), scaled by ||rays_d|| to convert to metric distance.
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples-1]
        far_dist = tf.fill(tf.shape(z_vals[..., :1]), 1e10)
        dists = tf.concat([dists, far_dist], axis=-1)  # [N_rays, N_samples]
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)  # [N_rays, N_samples]

        if raw_noise_std > 0.0:
            noise = tf.random.normal(tf.shape(sigma), stddev=raw_noise_std)
            sigma = sigma + noise

        alpha = 1.0 - tf.exp(-sigma * dists)  # [N_rays, N_samples], Eq. 3
        # T_i = exp(-sum_{j<i} sigma_j delta_j) == cumulative product of (1-alpha_j) for j<i.
        transmittance = tf.math.cumprod(1.0 - alpha + _EPS, axis=-1, exclusive=True)  # [N_rays, N_samples]
        weights = alpha * transmittance  # [N_rays, N_samples], Eq. 5's w_i

        rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)  # [N_rays, 3]
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)  # [N_rays]
        acc_map = tf.reduce_sum(weights, axis=-1)  # [N_rays]

        if white_background:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return {
            "rgb_map": rgb_map,
            "weights": weights,
            "depth_map": depth_map,
            "acc_map": acc_map,
        }

    def __repr__(self) -> str:  # noqa: D105
        return "VolumeRenderer()"


class HierarchicalSampler:
    """Inverse-transform importance sampler from coarse weights (Sec 5.2, Eq. 5)."""

    def sample_pdf(
        self,
        z_vals_mid: tf.Tensor,
        weights: tf.Tensor,
        n_importance: int,
        deterministic: bool = False,
    ) -> tf.Tensor:
        """Draw `n_importance` new samples from the coarse weight distribution.

        Implements: normalize coarse weights w_i into a piecewise-constant
        PDF (Eq. 5: what_i = w_i / sum_j w_j), then draw new samples via
        inverse-CDF sampling so they concentrate on high-weight (i.e.
        high-expected-visibility) regions of the ray.

        Args:
            z_vals_mid: [N_rays, M], the *midpoints* between consecutive
                coarse z_vals (M = N_coarse - 1), acting as the M edges/bin
                centers ("bins") that new samples are interpolated between.
            weights: [N_rays, M-1], the coarse *interior* per-bin weights
                (i.e. `weights[..., 1:-1]` of VolumeRenderer's output — the
                standard convention of dropping the first/last weight so
                edge bins don't dominate). Its length is exactly one less
                than `z_vals_mid` because the cumulative-sum-with-a-leading-
                zero used to build the CDF (below) restores that missing
                entry, making `cdf` and `z_vals_mid` the same length M.
            n_importance: Nf, number of new samples to draw per ray.
            deterministic: if True, use evenly spaced quantiles instead of
                stochastic sampling (used at evaluation/render time for
                reproducible outputs).

        Returns:
            new_z_vals: [N_rays, n_importance], sorted new sample depths.
        """
        assert weights.shape[-1] == z_vals_mid.shape[-1] - 1, (
            f"Expected weights bins (M-1={z_vals_mid.shape[-1] - 1}) to be one less than "
            f"z_vals_mid (M={z_vals_mid.shape[-1]}), got weights {weights.shape}, "
            f"z_vals_mid {z_vals_mid.shape}"
        )
        weights = weights + _EPS  # avoid nans from all-zero bins (Eq. 5 normalization)
        pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)  # what_i, Eq. 5, length M-1
        cdf = tf.cumsum(pdf, axis=-1)
        cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)  # [N_rays, M] -- now matches z_vals_mid

        n_rays = tf.shape(cdf)[0]
        if deterministic:
            u = tf.linspace(0.0, 1.0, n_importance)
            u = tf.broadcast_to(u, [n_rays, n_importance])
        else:
            u = tf.random.uniform([n_rays, n_importance])

        # Invert the CDF: find bin index for each u, then linearly interpolate within the bin.
        inds = tf.searchsorted(cdf, u, side="right")
        below = tf.maximum(inds - 1, 0)
        above = tf.minimum(inds, tf.shape(cdf)[-1] - 1)
        inds_g = tf.stack([below, above], axis=-1)  # [N_rays, n_importance, 2]

        cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=1)  # [N_rays, n_importance, 2]
        bins_g = tf.gather(z_vals_mid, inds_g, axis=-1, batch_dims=1)  # [N_rays, n_importance, 2]

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = tf.where(denom < _EPS, tf.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        new_z_vals = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # [N_rays, n_importance]

        return tf.stop_gradient(new_z_vals)

    def __repr__(self) -> str:  # noqa: D105
        return "HierarchicalSampler()"
