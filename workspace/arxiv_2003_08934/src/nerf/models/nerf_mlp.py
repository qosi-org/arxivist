"""
models/nerf_mlp.py — Core NeRF MLP: trunk, density head, view-dependent color head.

Implements the architecture described in Sec 3 and shown in Fig. 7 of
arXiv:2003.08934:

  - An 8-layer, 256-channel-wide, ReLU-activated fully-connected trunk that
    consumes gamma(x) and predicts volume density sigma as a function of
    position ONLY (multiview consistency, Sec 3).
  - A skip connection (DeepSDF-style, Fig. 7 caption) that re-concatenates
    gamma(x) into the trunk's 5th-layer activation.
  - A separate, single-ReLU-layer + sigmoid color head that additionally
    consumes gamma(d) (the encoded viewing direction) to predict
    view-dependent RGB radiance (non-Lambertian effects, Fig. 3/4).

This single NeRFMLP class is instantiated twice by radiance_field.py — once
for the "coarse" network and once for the "fine" network (Sec 5.2) — as two
independent weight sets of identical architecture.

SIR reference: architecture modules "MLP_Trunk (F'_Theta, spatial branch)"
(confidence 0.95), "DensityHead" (confidence 0.9), "ColorHead (view-dependent
branch)" (confidence 0.95).
"""

from __future__ import annotations

import tensorflow as tf


class NeRFMLP(tf.keras.Model):
    """The 8-layer trunk + density head + view-dependent color head (Fig. 7).

    Args:
        trunk_depth: number of fully-connected ReLU layers in the trunk.
            Paper default: 8.
        trunk_width: channels per trunk layer. Paper default: 256.
        color_width: channels in the single hidden layer of the color head.
            Paper default: 128.
        skip_layers: 0-indexed trunk layer indices at which gamma(x) is
            re-concatenated into the layer's *input* activation (Fig. 7:
            "include a skip connection that concatenates this input to the
            fifth layer's activation" -> 0-indexed layer 4). Paper default:
            (4,).
        pos_enc_dim: dimensionality of gamma(x). Paper default: 60 (L=10).
        dir_enc_dim: dimensionality of gamma(d). Paper default: 24 (L=4).
            Ignored if use_view_dirs=False.
        use_view_dirs: if False, sigma AND color are both predicted from
            gamma(x) alone (reproduces the "No View Dependence" ablation,
            Table 2 row 3).
    """

    def __init__(
        self,
        trunk_depth: int = 8,
        trunk_width: int = 256,
        color_width: int = 128,
        skip_layers: tuple[int, ...] = (4,),
        pos_enc_dim: int = 60,
        dir_enc_dim: int = 24,
        use_view_dirs: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        assert trunk_depth > 0, f"trunk_depth must be > 0, got {trunk_depth}"
        assert all(0 <= s < trunk_depth for s in skip_layers), (
            f"skip_layers {skip_layers} must be valid indices into a trunk of depth {trunk_depth}"
        )
        self.trunk_depth = trunk_depth
        self.trunk_width = trunk_width
        self.skip_layers = set(skip_layers)
        self.pos_enc_dim = pos_enc_dim
        self.dir_enc_dim = dir_enc_dim
        self.use_view_dirs = use_view_dirs

        # Trunk: 8 Dense(256, ReLU) layers, Fig. 7.
        self.trunk_layers = [
            tf.keras.layers.Dense(trunk_width, activation="relu", name=f"trunk_dense_{i}")
            for i in range(trunk_depth)
        ]

        # Density head: Dense(1, ReLU) ensures sigma >= 0 (Fig. 7 caption).
        self.sigma_head = tf.keras.layers.Dense(1, activation="relu", name="sigma_head")
        # Un-activated 256-d feature that feeds the color head (Fig. 7).
        self.feature_head = tf.keras.layers.Dense(trunk_width, activation=None, name="feature_head")

        if use_view_dirs:
            self.color_hidden = tf.keras.layers.Dense(
                color_width, activation="relu", name="color_hidden"
            )
        self.color_out = tf.keras.layers.Dense(3, activation="sigmoid", name="color_out")

    def call(self, gamma_x: tf.Tensor, gamma_d: tf.Tensor | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass: gamma(x) [, gamma(d)] -> (rgb, sigma).

        Args:
            gamma_x: positionally-encoded location, shape [..., pos_enc_dim].
            gamma_d: positionally-encoded viewing direction, shape
                [..., dir_enc_dim]. Required if use_view_dirs=True.

        Returns:
            (rgb, sigma): rgb has shape [..., 3] in [0,1] (sigmoid); sigma
                has shape [..., 1] and is >= 0 (ReLU-rectified).
        """
        assert gamma_x.shape[-1] == self.pos_enc_dim, (
            f"Expected gamma_x last dim {self.pos_enc_dim}, got {gamma_x.shape[-1]}"
        )
        if self.use_view_dirs:
            assert gamma_d is not None, "gamma_d is required when use_view_dirs=True"
            assert gamma_d.shape[-1] == self.dir_enc_dim, (
                f"Expected gamma_d last dim {self.dir_enc_dim}, got {gamma_d.shape[-1]}"
            )

        h = gamma_x
        for i, layer in enumerate(self.trunk_layers):
            if i in self.skip_layers:
                # Fig. 7 skip connection: re-inject gamma(x) before this layer.
                h = tf.concat([h, gamma_x], axis=-1)
            h = layer(h)

        sigma = self.sigma_head(h)  # [..., 1], ReLU-rectified (Fig. 7 caption)
        feature = self.feature_head(h)  # [..., trunk_width], no activation

        if self.use_view_dirs:
            h2 = tf.concat([feature, gamma_d], axis=-1)
            h2 = self.color_hidden(h2)
            rgb = self.color_out(h2)
        else:
            # Ablation: color predicted from position-only features (Table 2, row 3).
            rgb = self.color_out(feature)

        return rgb, sigma

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"NeRFMLP(trunk_depth={self.trunk_depth}, trunk_width={self.trunk_width}, "
            f"skip_layers={sorted(self.skip_layers)}, use_view_dirs={self.use_view_dirs})"
        )
