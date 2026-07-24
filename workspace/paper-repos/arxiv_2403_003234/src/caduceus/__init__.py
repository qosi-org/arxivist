"""Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling.

Reproduction of arXiv:2403.03234 (Schiff et al., ICML 2024).
"""

__version__ = "1.0.0"


def _patch_transformers_tied_weights() -> None:
    """Compat shim for loading the Hub's Caduceus modeling code on newer transformers.

    transformers >=4.5x renamed/added ``PreTrainedModel.all_tied_weights_keys`` and
    calls it during ``_finalize_model_loading``. The published ``modeling_caduceus.py``
    predates that attribute, so the load crashes with:
        AttributeError: 'Caduceus' object has no attribute 'all_tied_weights_keys'
    even though the weights already loaded fine (mamba_rev.*_proj are intentionally
    tied/missing). We provide the attribute if transformers doesn't define it.
    """
    try:
        import transformers.modeling_utils as mu
    except Exception:  # transformers not importable yet; nothing to patch
        return
    if not hasattr(mu.PreTrainedModel, "all_tied_weights_keys"):
        # transformers >=4.5x treats this as a dict ({tied_param: source_param})
        # and calls .keys() on it. An empty dict means "nothing to reconcile",
        # which is correct here: the mamba_rev projections are intentionally
        # tied / newly-initialized for downstream fine-tuning.
        @property
        def all_tied_weights_keys(self):  # noqa: ANN001, ANN202
            val = getattr(self, "_tied_weights_keys", None)
            return val if isinstance(val, dict) else {}

        mu.PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys


_patch_transformers_tied_weights()
