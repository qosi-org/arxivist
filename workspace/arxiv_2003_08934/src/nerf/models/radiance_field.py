"""
models/radiance_field.py — Top-level NeRFModel: coarse/fine NeRFMLP pair + renderer.

Implements Sec 5.2-5.3 of arXiv:2003.08934: two independently-weighted
instances of the same NeRFMLP architecture ("coarse" and "fine"), composed
with PositionalEncoding, VolumeRenderer, and HierarchicalSampler into the
full two-pass rendering pipeline described in architecture_plan.json's
"Full two-pass render (ray batch -> pixel colors)" tensor flow.

SIR reference: architecture modules "CoarseNetwork (F_theta_coarse)" (0.9),
"FineNetwork (F_theta_fine)" (0.9); mathematical_spec "Combined coarse+fine
photometric MSE loss" context (loss itself lives in training/losses.py).
"""

from __future__ import annotations

import tensorflow as tf

from nerf.models.nerf_mlp import NeRFMLP
from nerf.models.positional_encoding import PositionalEncoding
from nerf.models.volume_rendering import HierarchicalSampler, VolumeRenderer


class NeRFModel(tf.keras.Model):
    """Composes positional encodings, coarse+fine NeRFMLPs, and the renderer.

    Args:
        config: a validated config dict (see `NeRFConfig.raw`) with the
            `model` and `model_variant` sections used to build the network.
    """

    def __init__(self, config: dict, **kwargs: object) -> None:
        super().__init__(**kwargs)
        model_cfg = config["model"]
        variant = config["model_variant"]["name"]

        self.use_view_dirs = model_cfg["use_view_dependence"] and variant != "no_view_dependence"
        self.use_hierarchical = (
            model_cfg["use_hierarchical_sampling"] and variant != "no_hierarchical_sampling"
        )
        use_pos_enc = variant != "no_positional_encoding"

        pos_freqs = model_cfg["pos_enc_freqs_x"] if use_pos_enc else 0
        dir_freqs = model_cfg["pos_enc_freqs_d"] if use_pos_enc else 0
        include_input = model_cfg.get("include_input_in_encoding", False) or not use_pos_enc

        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(pos_freqs, include_input=include_input)
            self.dir_encoder = PositionalEncoding(dir_freqs, include_input=include_input) if self.use_view_dirs else None
            pos_enc_dim = self.pos_encoder.output_dim(3)
            dir_enc_dim = self.dir_encoder.output_dim(3) if self.use_view_dirs else 0
        else:
            # "No Positional Encoding" ablation (Table 2, row 2): raw xyz(theta,phi) fed directly.
            self.pos_encoder = None
            self.dir_encoder = None
            pos_enc_dim = 3
            dir_enc_dim = 3 if self.use_view_dirs else 0

        mlp_kwargs = dict(
            trunk_depth=model_cfg["trunk_depth"],
            trunk_width=model_cfg["trunk_width"],
            color_width=model_cfg["color_hidden_width"],
            skip_layers=tuple(model_cfg["skip_layers"]),
            pos_enc_dim=pos_enc_dim,
            dir_enc_dim=dir_enc_dim,
            use_view_dirs=self.use_view_dirs,
        )
        self.coarse_mlp = NeRFMLP(name="coarse_mlp", **mlp_kwargs)
        self.fine_mlp = NeRFMLP(name="fine_mlp", **mlp_kwargs) if self.use_hierarchical else None

        self.renderer = VolumeRenderer()
        self.sampler = HierarchicalSampler()

        self.num_coarse_samples = model_cfg["num_coarse_samples"]
        self.num_fine_samples = model_cfg["num_fine_samples"] if self.use_hierarchical else 0
        self.raw_noise_std = model_cfg.get("raw_noise_std", 0.0)
        self.white_background = config["data"].get("white_background", False)

    # ------------------------------------------------------------------ #
    def _encode(self, pts: tf.Tensor, dirs: tf.Tensor | None) -> tuple[tf.Tensor, tf.Tensor | None]:
        gamma_x = self.pos_encoder(pts) if self.pos_encoder is not None else pts
        gamma_d = None
        if self.use_view_dirs:
            gamma_d = self.dir_encoder(dirs) if self.dir_encoder is not None else dirs
        return gamma_x, gamma_d

    def query_coarse(self, pts: tf.Tensor, dirs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Query the coarse network at arbitrary 3D points.

        Args:
            pts: [N_rays, N_samples, 3] spatial locations.
            dirs: [N_rays, N_samples, 3] viewing directions (broadcast per-ray).

        Returns:
            (rgb, sigma) each shaped [N_rays, N_samples, {3,1}].
        """
        gamma_x, gamma_d = self._encode(pts, dirs)
        return self.coarse_mlp(gamma_x, gamma_d)

    def query_fine(self, pts: tf.Tensor, dirs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Query the fine network (or the coarse network if hierarchical
        sampling is disabled) at arbitrary 3D points. Same signature/shapes
        as `query_coarse`.
        """
        gamma_x, gamma_d = self._encode(pts, dirs)
        mlp = self.fine_mlp if self.fine_mlp is not None else self.coarse_mlp
        return mlp(gamma_x, gamma_d)

    # ------------------------------------------------------------------ #
    def render_rays(
        self,
        rays_o: tf.Tensor,
        rays_d: tf.Tensor,
        near: float,
        far: float,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Full two-pass (coarse + hierarchical fine) render of a ray batch.

        Implements architecture_plan.json's "Full two-pass render" tensor
        flow end to end.

        Args:
            rays_o: [N_rays, 3] ray origins.
            rays_d: [N_rays, 3] ray directions (unnormalized camera rays).
            near: near bound t_n (Eq. 1-2).
            far: far bound t_f (Eq. 1-2).
            training: if True, stratified sampling uses per-bin random
                perturbation (Eq. 2) and importance sampling is stochastic;
                if False, both are deterministic (evaluation/render mode).

        Returns:
            dict with keys: rgb_coarse, rgb_fine, weights_coarse,
            depth_fine, acc_fine (rgb_fine == rgb_coarse if hierarchical
            sampling is disabled).
        """
        assert rays_o.shape[-1] == 3 and rays_d.shape[-1] == 3, "rays_o/rays_d must be [...,3]"
        n_rays = tf.shape(rays_o)[0]

        # --- Coarse pass: stratified sampling, Eq. 2 ---
        t_vals = tf.linspace(0.0, 1.0, self.num_coarse_samples)
        z_vals_coarse = near * (1.0 - t_vals) + far * t_vals  # [N_coarse]
        z_vals_coarse = tf.broadcast_to(z_vals_coarse, [n_rays, self.num_coarse_samples])
        if training:
            mids = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            upper = tf.concat([mids, z_vals_coarse[..., -1:]], axis=-1)
            lower = tf.concat([z_vals_coarse[..., :1], mids], axis=-1)
            t_rand = tf.random.uniform(tf.shape(z_vals_coarse))
            z_vals_coarse = lower + (upper - lower) * t_rand  # Eq. 2 stratified perturbation

        pts_coarse = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_coarse[..., None]
        dirs_bcast = tf.broadcast_to(rays_d[:, None, :], tf.shape(pts_coarse))
        rgb_c, sigma_c = self.query_coarse(pts_coarse, dirs_bcast)
        out_c = self.renderer.composite(
            rgb_c, sigma_c, z_vals_coarse, rays_d,
            raw_noise_std=self.raw_noise_std if training else 0.0,
            white_background=self.white_background,
        )

        result = {
            "rgb_coarse": out_c["rgb_map"],
            "weights_coarse": out_c["weights"],
        }

        if not self.use_hierarchical:
            result["rgb_fine"] = out_c["rgb_map"]
            result["depth_fine"] = out_c["depth_map"]
            result["acc_fine"] = out_c["acc_map"]
            return result

        # --- Fine pass: hierarchical importance sampling, Eq. 5 ---
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])  # [N_rays, N_coarse-1]
        interior_weights = out_c["weights"][..., 1:-1]  # drop edge bins per standard convention
        z_samples = self.sampler.sample_pdf(
            z_vals_mid, interior_weights, self.num_fine_samples, deterministic=not training
        )
        z_vals_fine = tf.sort(tf.concat([z_vals_coarse, z_samples], axis=-1), axis=-1)

        pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_fine[..., None]
        dirs_bcast_f = tf.broadcast_to(rays_d[:, None, :], tf.shape(pts_fine))
        rgb_f, sigma_f = self.query_fine(pts_fine, dirs_bcast_f)
        out_f = self.renderer.composite(
            rgb_f, sigma_f, z_vals_fine, rays_d,
            raw_noise_std=self.raw_noise_std if training else 0.0,
            white_background=self.white_background,
        )

        result["rgb_fine"] = out_f["rgb_map"]
        result["depth_fine"] = out_f["depth_map"]
        result["acc_fine"] = out_f["acc_map"]
        return result

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"NeRFModel(use_view_dirs={self.use_view_dirs}, "
            f"use_hierarchical={self.use_hierarchical}, "
            f"Nc={self.num_coarse_samples}, Nf={self.num_fine_samples})"
        )
