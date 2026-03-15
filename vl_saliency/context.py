from typing import Self, overload

from transformers import PreTrainedModel

from vl_saliency._hooks.patch import (
    apply_forward_patch,
    is_patched,
    restore_forward,
)
from vl_saliency._logging import get_logger
from vl_saliency._types import Backend, HeadOp, LayerOp, PatchLayoutFn, Reduction
from vl_saliency.config import SaliencyConfig
from vl_saliency.config.select import SelectionSpec

logger = get_logger(__name__)


class Saliency:
    """
    Context manager for computing saliency maps during the forward pass of a vision-language model.

    Args:
        model (PreTrainedModel): Vision-language model to compute saliency for.
        image_token_id (int | None, default=None): Token ID for image patches. If "None", will be inferred from model config.
        pad_token_id (int | None, default=None): Token ID for padding tokens. If "None", will be inferred from model config.
        patch_layout_fn (tuple[int, int] | Callable | None, default=None):
            If a tuple of (height, width), will use static patch shapes for all images. If a callable is provided,
            it will be called with (image_count, **forward_pass_kwargs) to get the patch shape at runtime.
            If "None", will be inferred from model config.
        attn_scale (float, optional): Optional scaling factor to apply to the saliency map values. If "None", computed from default scaling heuristics.
            Recommended not to set this manually unless you have specific requirements for the saliency map scale.
        head_op (HeadOp, optional): Optional operation to apply to each head's attention map before reduction.
        head_reduce (Reduction, default="mean"): Method to reduce attention across heads ("mean", "max", etc.).
        layer_op (LayerOp, optional): Optional operation to apply to each layer's attention map before reduction.
        layer_reduce (Reduction, default="mean"): Method to reduce attention across layers ("mean", "max", etc.).
        selection (SelectionSpec | None, default=None): Select specific heads or layers to compute saliency from.
        backend (Backend, default="auto"): Backend to use for saliency computation. "auto" will choose "triton" if available, otherwise "torch".

    Usage:
        ```
        with Saliency(model):
            outputs = model(**inputs)

        sal = outputs.saliency  # Access the computed saliency map
        ```
    Note:
        If the model returns a tuple, the saliency map will be the last element.
    """

    @overload
    def __init__(
        self: Self,
        model: PreTrainedModel,
        config: SaliencyConfig,
    ): ...

    @overload
    def __init__(
        self: Self,
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
    ): ...

    def __init__(
        self: Self,
        model: PreTrainedModel,
        config: SaliencyConfig | None = None,
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
    ):
        self.model = model
        self.config = config or SaliencyConfig.from_model(
            model,
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

    def __enter__(self: Self) -> Self:
        self.enable()
        return self

    def __exit__(self: Self, exc_type, exc_val, exc_tb):
        self.disable()
        return False  # Don't suppress exceptions

    def enable(self: Self):
        """
        Enables saliency computation by injecting hooks into the model. This modifies the model's forward pass to compute saliency maps alongside the original outputs.
        Equivalent to using the context manager. After calling this method, the model will compute saliency maps until `disable()` is called.
        """
        if is_patched(self.model):
            logger.warning(
                "Saliency is already enabled for this model. Multiple calls to enable() have no additional effect."
            )
            return  # Already enabled

        apply_forward_patch(self.model, self.config)

    def disable(self: Self):
        """Restores the model's original forward method, disabling saliency computation until `enable()` is called again."""
        if not is_patched(self.model):
            logger.warning(
                "Saliency is already disabled for this model. Multiple calls to disable() have no additional effect."
            )
            return  # Already disabled

        restore_forward(self.model)
