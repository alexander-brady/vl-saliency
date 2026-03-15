from dataclasses import dataclass

from transformers import PreTrainedModel

from vl_saliency._hooks.infer import (
    infer_attn_scale,
    infer_image_token_id,
    infer_pad_token_id,
    infer_patch_layout_fn,
)
from vl_saliency._types import Backend, HeadOp, LayerOp, PatchLayoutFn, Reduction
from vl_saliency.config.patch_fns import StaticPatchLayout
from vl_saliency.config.select import SelectionSpec


@dataclass(frozen=True)
class SaliencyConfig:
    """
    Configuration for saliency map computation.

    Can be constructed directly or inferred from a model using `SaliencyConfig.from_model`.
    """

    image_token_id: int
    """ID of token used to represent image patches."""

    pad_token_id: int
    """ID of token used for padding."""

    patch_layout_fn: PatchLayoutFn
    """Function that infers the image patch layout from the input sequences."""

    attn_scale: float
    """Scaling factor applied to QK attention. Recommended: 1 / sqrt(head_dim)."""

    head_op: HeadOp | None = None
    """Optional transform applied to head saliency before reduction (e.g., 'abs', 'square')."""

    head_reduce: Reduction = "mean"
    """Reduction used to aggregate saliency across heads (e.g., 'mean', 'max')."""

    layer_op: LayerOp | None = None
    """Optional transform applied to layer saliency before reduction (e.g., 'abs', 'square')."""

    layer_reduce: Reduction = "mean"
    """Reduction used to aggregate saliency across layers (e.g., 'mean', 'max')."""

    selection: SelectionSpec | None = None
    """Optional specification selecting which layers or heads contribute to saliency. If None, all are used."""

    backend: Backend = "auto"
    """Backend used for saliency computation. "auto" selects "triton" if available, otherwise "torch"."""

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        *,
        image_token_id: int | None = None,
        pad_token_id: int | None = None,
        patch_layout_fn: tuple[int, int] | PatchLayoutFn | None = None,
        attn_scale: float | None = None,
        head_op: HeadOp | None = None,
        head_reduce: Reduction = "mean",
        layer_op: LayerOp | None = None,
        layer_reduce: Reduction = "mean",
        selection: SelectionSpec | None = None,
        backend: Backend = "auto",
    ) -> "SaliencyConfig":
        """creates a SaliencyConfig. Any fields not explicitely set will be inferred from the model."""

        if image_token_id is None:
            image_token_id = infer_image_token_id(model.config)

        if pad_token_id is None:
            pad_token_id = infer_pad_token_id(model.config)

        patch_layout_fn = (
            StaticPatchLayout(*patch_layout_fn)
            if isinstance(patch_layout_fn, tuple)
            else patch_layout_fn
            if callable(patch_layout_fn)
            else infer_patch_layout_fn(model.config)
        )

        if attn_scale is None:
            attn_scale = infer_attn_scale(model.config)

        return cls(
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            patch_layout_fn=patch_layout_fn,
            attn_scale=attn_scale,
            head_op=head_op,
            head_reduce=head_reduce,
            layer_op=layer_op,
            layer_reduce=layer_reduce,
            selection=selection,
            backend=backend,
        )
