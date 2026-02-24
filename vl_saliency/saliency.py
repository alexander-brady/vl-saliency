import math
from collections.abc import Callable
from types import MethodType
from typing import Any, Literal, overload

from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput

from vl_saliency._types import Backend, HeadOp, ImagePatchFunction, LayerOp, Reduction
from vl_saliency.config import SaliencyConfig
from vl_saliency.outputs import SaliencyOutput
from vl_saliency.trace import SaliencyTrace
from vl_saliency.utils.image_init import infer_image_patch_fn, infer_image_token_id
from vl_saliency.utils.logger import get_logger
from vl_saliency.utils.patch_fns import StaticPatches

logger = get_logger(__name__)


class Saliency:
    """
    Context manager for computing saliency maps during the forward pass of a vision-language model.

    Args:
        model (PreTrainedModel): Vision-language model to compute saliency for.
        image_token_id (int | None, default=None): Token ID for image patches. If "None", will be inferred from model config.
        image_patch_fn (tuple[int, int] | Callable | None, default=None):
            If a tuple of (height, width), will use static patch shapes for all images. If a callable is provided,
            it will be called with (image_count, **forward_pass_kwargs) to get the patch shape at runtime.
            If "None", will be inferred from model config.
        layer_reduce (Reduction, default="mean"): Method to reduce attention across layers ("mean", "max", etc.).
        head_reduce (Reduction, default="mean"): Method to reduce attention across heads ("mean", "max", etc.).
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

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        image_token_id: int | None = None,
        image_patch_fn: tuple[int, int] | ImagePatchFunction | None = None,
        layer_reduce: Reduction = "mean",
        layer_op: LayerOp | None = None,
        head_reduce: Reduction = "mean",
        head_op: HeadOp | None = None,
        backend: Backend = "auto",
    ):
        self.model = model

        pad_token_id = model.config.pad_token_id
        image_token_id = (
            image_token_id if image_token_id is not None else infer_image_token_id(model.config)
        )

        resolved_patch_fn: ImagePatchFunction = (
            StaticPatches(*image_patch_fn)
            if isinstance(image_patch_fn, tuple)
            else image_patch_fn
            if callable(image_patch_fn)
            else infer_image_patch_fn(model.config)
        )

        self.config = SaliencyConfig(
            pad_token_id=pad_token_id,
            image_token_id=image_token_id,
            image_patch_fn=resolved_patch_fn,
            layer_reduce=layer_reduce,
            layer_op=layer_op,
            head_reduce=head_reduce,
            head_op=head_op,
            backend=backend,
        )

        self._prev_forward: Callable | None = None
        self._prev_attn_impl: str | None = None

    @staticmethod
    def from_config(model: PreTrainedModel, config: SaliencyConfig) -> "Saliency":
        """Factory method to create a Saliency instance directly from a SaliencyConfig object."""
        return Saliency(
            model=model,
            image_token_id=config.image_token_id,
            image_patch_fn=config.image_patch_fn,
            layer_reduce=config.layer_reduce,
            layer_op=config.layer_op,
            head_reduce=config.head_reduce,
            head_op=config.head_op,
            backend=config.backend,
        )

    def __enter__(self):
        self.wrap()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unwrap()
        return False  # Don't suppress exceptions

    def wrap(self):
        """Binds the custom attention implementation to the model for saliency extraction."""
        if self._prev_forward is not None and self._prev_attn_impl is not None:
            return  # Already wrapped

        prev_forward = self.model.forward
        prev_attn_impl = self.model.config.text_config._attn_implementation

        new_forward = self._build_saliency_forward(
            config=self.config,
            attn_implementation="saliency",
            forward=prev_forward,
        )

        self._prev_forward = prev_forward
        self._prev_attn_impl = prev_attn_impl

        self.model.set_attn_implementation({"text_config": "saliency"})
        self.model.forward = MethodType(new_forward, self.model)

    def unwrap(self):
        """Restores the model's original attention implementation."""
        if self._prev_forward is None or self._prev_attn_impl is None:
            logger.warning("Model is not wrapped, nothing to unwrap.")
            return

        self.model.forward = self._prev_forward
        self.model.set_attn_implementation({"text_config": self._prev_attn_impl})

        self._prev_forward = None
        self._prev_attn_impl = None

    @staticmethod
    def _build_saliency_forward(
        config: SaliencyConfig,
        attn_implementation: str,
        forward: Callable[..., ModelOutput],
    ) -> Callable[..., SaliencyOutput]:
        """Builds a custom forward method that creates a SaliencyTrace and passes it through the model's forward pass to compute saliency maps."""

        @overload
        def _saliency_forward(
            model_self,
            input_ids: Int[Tensor, "B S"],
            pixel_values: Float[Tensor, "B C H W"] | None = None,
            *,
            return_dict: Literal[True] | None = None,
            **kwargs,
        ) -> SaliencyOutput: ...

        @overload
        def _saliency_forward(
            model_self,
            input_ids: Int[Tensor, "B S"],
            pixel_values: Float[Tensor, "B C H W"] | None = None,
            *,
            return_dict: Literal[False],
            **kwargs,
        ) -> tuple[Any, ...]: ...

        def _saliency_forward(
            model_self,
            input_ids: Int[Tensor, "B S"],
            pixel_values: Float[Tensor, "B C H W"] | None = None,
            *,
            return_dict: bool | None = None,
            **kwargs,
        ) -> SaliencyOutput | tuple[Any, ...]:

            head_dim = model_self.config.hidden_size // model_self.config.num_attention_heads
            scale = 1.0 / math.sqrt(head_dim)

            trace = kwargs.get("saliency") or SaliencyTrace(
                config=config,
                input_ids=input_ids,
                pixel_values=pixel_values,
                scale=scale,
            )

            kwargs["saliency"] = trace
            kwargs.setdefault("attn_implementation", attn_implementation)

            out = forward(model_self, input_ids=input_ids, pixel_values=pixel_values, **kwargs)

            default_return_dict = (
                model_self.config.return_dict if hasattr(model_self, "config") else True
            )
            return_dict = return_dict if return_dict is not None else default_return_dict
            if not return_dict and not isinstance(out, tuple):
                out = out.to_tuple()

            if isinstance(out, tuple):  # return_dict=False
                out = tuple(list(out) + [trace.saliency])
            else:
                out = SaliencyOutput(base_output=out, saliency=trace.saliency)
            return out

        return _saliency_forward
