from dataclasses import dataclass

from transformers import PreTrainedModel

from vl_saliency.types import Backend, HeadOp, ImagePatchFunction, LayerOp, Reduction, SelectionSpec
from vl_saliency.utils.infer import (
    infer_attn_scale,
    infer_image_patch_fn,
    infer_image_token_id,
    infer_pad_token_id,
)
from vl_saliency.utils.patch_fns import FixedPatchLayout


@dataclass(frozen=True)
class SaliencyConfig:
    """Configuration for saliency map computation. Can be constructed directly or inferred from a model using the `from_model` method."""

    image_token_id: int
    """ID of token used to represent image patches."""
    pad_token_id: int
    """ID of token used for padding."""
    image_patch_fn: ImagePatchFunction
    """Function that infers shape from input."""
    attn_scale: float
    """Scaling factor to apply to the saliency map values. Recommended: 1 / sqrt(head_dim)."""
    head_op: HeadOp | None = None
    """Optional operation to apply across heads before reduction (e.g., 'abs', 'square')."""
    head_reduce: Reduction = "mean"
    """Function to reduce saliency across heads (e.g., 'mean', 'max')."""
    layer_op: LayerOp | None = None
    """Optional operation to apply across layers before reduction (e.g., 'abs', 'square')."""
    layer_reduce: Reduction = "mean"
    """Function to reduce saliency across layers (e.g., 'mean', 'max')."""
    backend: Backend = "auto"
    """Backend to use for saliency computation. 'auto' will choose 'triton' if available, otherwise 'torch'."""
    subset_select: SelectionSpec | None = None
    """Optional specification for selecting a subset of layers or heads to accumulate saliency from. If None, saliency will be accumulated from all layers and heads."""

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        *,
        image_token_id: int | None = None,
        pad_token_id: int | None = None,
        image_patch_fn: tuple[int, int] | ImagePatchFunction | None = None,
        attn_scale: float | None = None,
        head_op: HeadOp | None = None,
        head_reduce: Reduction = "mean",
        layer_op: LayerOp | None = None,
        layer_reduce: Reduction = "mean",
        backend: Backend = "auto",
        subset_select: SelectionSpec | None = None,
    ) -> "SaliencyConfig":
        """Infers a SaliencyConfig from a given model and optional parameters."""

        if image_token_id is None:
            image_token_id = infer_image_token_id(model.config)

        if pad_token_id is None:
            pad_token_id = infer_pad_token_id(model.config)

        resolved_patch_fn: ImagePatchFunction = (
            FixedPatchLayout(*image_patch_fn)
            if isinstance(image_patch_fn, tuple)
            else image_patch_fn
            if callable(image_patch_fn)
            else infer_image_patch_fn(model.config)
        )

        if attn_scale is None:
            attn_scale = infer_attn_scale(model.config)

        return cls(
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            image_patch_fn=resolved_patch_fn,
            attn_scale=attn_scale,
            head_op=head_op,
            head_reduce=head_reduce,
            layer_op=layer_op,
            layer_reduce=layer_reduce,
            backend=backend,
            subset_select=subset_select,
        )
