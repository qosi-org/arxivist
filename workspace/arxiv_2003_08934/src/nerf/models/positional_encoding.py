"""
models/positional_encoding.py — High-frequency Fourier feature encoding.

Implements Eq. 4 (Sec 5.1) of arXiv:2003.08934:

    gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^(L-1) pi p), cos(2^(L-1) pi p))

Applied independently to each of the 3 components of the spatial location x
(with L=10, paper default) and the 3 components of the unit viewing direction
d (with L=4, paper default). This mapping is the composition target
F_Theta = F'_Theta . gamma described in Sec 5.1: gamma has no learned
parameters, it purely re-expresses low-dimensional coordinates in a higher
dimensional space so the downstream MLP can fit high-frequency detail.

SIR reference: mathematical_spec "Positional (Fourier feature) encoding"
(confidence 0.95); architecture modules "PositionalEncoding_x" /
"PositionalEncoding_d" (confidence 0.95).
"""

from __future__ import annotations

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """Fourier-feature positional encoding, gamma(p) from Eq. 4.

    Args:
        num_freqs: L, the number of frequency bands. Paper uses L=10 for
            gamma(x) and L=4 for gamma(d) (Sec 5.1).
        include_input: if True, concatenates the raw (unencoded) input
            alongside the sin/cos features. The paper's Eq. 4 as written does
            NOT include the raw input; this flag defaults to False to match
            the paper exactly, but is exposed because some downstream
            reproductions optionally add it. See configs/config.yaml
            `model.include_input_in_encoding`.
    """

    def __init__(self, num_freqs: int, include_input: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert num_freqs > 0, f"num_freqs must be > 0, got {num_freqs}"
        self.num_freqs = num_freqs
        self.include_input = include_input
        # 2^0, 2^1, ..., 2^(L-1)  — Eq. 4 frequency bands.
        self.freq_bands = tf.constant(
            [2.0**i for i in range(num_freqs)], dtype=tf.float32
        )

    def call(self, p: tf.Tensor) -> tf.Tensor:
        """Apply gamma(p) to the last dimension of `p`.

        Args:
            p: tensor of shape [..., D] with D=3 for x or d, values expected
                to already be normalized/unit as required by Eq. 4's context
                (x normalized to [-1,1]; d is a unit vector by construction).

        Returns:
            Tensor of shape [..., 2*num_freqs*D] (or
            [..., D + 2*num_freqs*D] if include_input=True).
        """
        assert p.shape[-1] is not None, "Last dimension of positional-encoding input must be static"
        # p: [..., D] -> scaled: [..., D, L] via outer product with freq_bands * pi
        scaled = p[..., None] * self.freq_bands * tf.constant(3.14159265358979, dtype=tf.float32)  # [..., D, L]
        sin_feats = tf.sin(scaled)  # [..., D, L]
        cos_feats = tf.cos(scaled)  # [..., D, L]
        # Interleave sin/cos per Eq. 4 ordering: (sin_0, cos_0, sin_1, cos_1, ...)
        stacked = tf.stack([sin_feats, cos_feats], axis=-1)  # [..., D, L, 2]
        d = p.shape[-1]
        encoded = tf.reshape(stacked, tf.concat([tf.shape(p)[:-1], [d * self.num_freqs * 2]], axis=0))
        if self.include_input:
            encoded = tf.concat([p, encoded], axis=-1)
        return encoded

    def output_dim(self, input_dim: int) -> int:
        """Return the output feature dimension for a given input dimension."""
        base = 2 * self.num_freqs * input_dim
        return base + input_dim if self.include_input else base

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PositionalEncoding(num_freqs={self.num_freqs}, "
            f"include_input={self.include_input})"
        )
