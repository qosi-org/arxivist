"""
data/rays.py — Camera ray generation and the NDC ray reparameterization.

Implements:
  - Pinhole-camera ray generation from a camera-to-world pose.
  - The Normalized Device Coordinate (NDC) ray transform (Appendix C,
    Eq. 25-26) required for "Real Forward-Facing" (LLFF) scenes, which maps
    a perspective ray into a space where z represents disparity (inverse
    depth) so a single [0,1] linear sampling range covers [near, infinity].

Stratified sampling along a ray (Eq. 2) is implemented directly inside
`NeRFModel.render_rays` (models/radiance_field.py) since it is tightly
coupled to the per-call near/far bounds; it is not duplicated here.

SIR reference: mathematical_spec "Normalized Device Coordinate (NDC) ray
reparameterization..." (confidence 0.85, Appendix C).
"""

from __future__ import annotations

import tensorflow as tf


class RayGenerator:
    """Pinhole-camera ray generation and NDC reparameterization."""

    def get_rays(self, H: int, W: int, focal: float, c2w: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate one camera ray per pixel for a pinhole camera.

        Args:
            H: image height in pixels.
            W: image width in pixels.
            focal: focal length in pixels (f_cam).
            c2w: [3, 4] or [4, 4] camera-to-world matrix (rotation + translation).

        Returns:
            (rays_o, rays_d): each [H, W, 3]. rays_o is the (broadcast)
            camera origin in world space; rays_d is the per-pixel,
            unnormalized world-space ray direction.
        """
        assert c2w.shape[-2] >= 3 and c2w.shape[-1] == 4, f"Expected c2w [3-4,4], got {c2w.shape}"
        i, j = tf.meshgrid(
            tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing="xy"
        )
        # Camera-space ray directions (looking down -z, standard NeRF/OpenGL convention).
        dirs = tf.stack(
            [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -tf.ones_like(i)], axis=-1
        )  # [H, W, 3]
        rot = c2w[:3, :3]
        rays_d = tf.reduce_sum(dirs[..., None, :] * rot, axis=-1)  # [H, W, 3]
        rays_o = tf.broadcast_to(c2w[:3, 3], tf.shape(rays_d))  # [H, W, 3]
        return rays_o, rays_d

    def ndc_rays(
        self,
        H: int,
        W: int,
        focal: float,
        near: float,
        rays_o: tf.Tensor,
        rays_d: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Map camera-space rays into Normalized Device Coordinates.

        Implements Appendix C, Eq. 25-26 (far bound assumed to be infinity,
        so the z-constants simplify to a_z=1, b_z=2n as derived in the
        appendix).

        Args:
            H: image height in pixels.
            W: image width in pixels.
            focal: focal length in pixels.
            near: the near clipping plane n.
            rays_o: [..., 3] ray origins in camera/world space.
            rays_d: [..., 3] ray directions in camera/world space.

        Returns:
            (rays_o_ndc, rays_d_ndc): rays reparameterized so that
            linearly sampling t' in [0,1] corresponds to linearly sampling
            disparity from `near` to infinity in the original space.
        """
        # Shift ray origins to the near plane (Appendix C, final paragraph).
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        ox, oy, oz = rays_o[..., 0], rays_o[..., 1], rays_o[..., 2]
        dx, dy, dz = rays_d[..., 0], rays_d[..., 1], rays_d[..., 2]

        o0 = -1.0 * (focal / (W / 2.0)) * ox / oz
        o1 = -1.0 * (focal / (H / 2.0)) * oy / oz
        o2 = 1.0 + 2.0 * near / oz

        d0 = -1.0 * (focal / (W / 2.0)) * (dx / dz - ox / oz)
        d1 = -1.0 * (focal / (H / 2.0)) * (dy / dz - oy / oz)
        d2 = -2.0 * near / oz

        rays_o_ndc = tf.stack([o0, o1, o2], axis=-1)
        rays_d_ndc = tf.stack([d0, d1, d2], axis=-1)
        return rays_o_ndc, rays_d_ndc

    def stratified_sample(
        self, near: float, far: float, n_samples: int, n_rays: int, perturb: bool = True
    ) -> tf.Tensor:
        """Standalone Eq. 2 stratified sampling utility (bin partition + random offset).

        Provided as a reusable, testable unit; `NeRFModel.render_rays` inlines
        an equivalent computation directly so it can share tensors with the
        rest of the forward pass without an extra function-call boundary.

        Args:
            near: t_n.
            far: t_f.
            n_samples: N, number of stratified bins/samples.
            n_rays: number of rays to generate samples for.
            perturb: if True, draw a random offset within each bin (Eq. 2);
                if False, use deterministic bin midpoints.

        Returns:
            z_vals: [n_rays, n_samples].
        """
        t_vals = tf.linspace(0.0, 1.0, n_samples)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = tf.broadcast_to(z_vals, [n_rays, n_samples])
        if not perturb:
            return z_vals
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], axis=-1)
        lower = tf.concat([z_vals[..., :1], mids], axis=-1)
        t_rand = tf.random.uniform(tf.shape(z_vals))
        return lower + (upper - lower) * t_rand

    def __repr__(self) -> str:  # noqa: D105
        return "RayGenerator()"
