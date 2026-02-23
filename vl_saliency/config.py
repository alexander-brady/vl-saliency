from dataclasses import dataclass

from vl_saliency._types import Backend, HeadOp, ImagePatchFunction, LayerOp, Reduction


@dataclass(frozen=True)
class SaliencyConfig:
    image_token_id: int
    """ID of token used to represent image patches."""
    pad_token_id: int
    """ID of token used for padding."""
    image_patch_fn: ImagePatchFunction
    """Function that infers shape from input."""
    scale: float
    """Scale utilized in saliency computation, typically 1/sqrt(head_dim)."""
    layer_reduce: Reduction = "mean"
    """Function to reduce saliency across layers (e.g., 'mean', 'max')."""
    layer_op: LayerOp | None = None
    """Optional operation to apply across layers before reduction (e.g., 'abs', 'square')."""
    head_reduce: Reduction = "mean"
    """Function to reduce saliency across heads (e.g., 'mean', 'max')."""
    head_op: HeadOp | None = None
    """Optional operation to apply across heads before reduction (e.g., 'abs', 'square')."""
    backend: Backend = "auto"
    """Backend to use for saliency computation. 'auto' will choose 'triton' if available, otherwise 'torch'."""
