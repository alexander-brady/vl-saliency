import math

import torch
from transformers import PreTrainedModel

from vl_saliency._types import Backend, Reduction
from vl_saliency.context import SaliencyContext
from vl_saliency.utils.image_init import infer_image_patch_fn, infer_image_token_id
from vl_saliency.utils.logger import get_logger
from vl_saliency.utils.patch_fns import ImagePatchFunction, StaticPatches

logger = get_logger(__name__)


class SaliencyExtractor:
    """Preprocesses model input to allow saliency computation in the forward pass.

    Args:
        model (PreTrainedModel): Vision-language model to compute saliency for.
        bind (bool, default=True): Whether to bind custom attention implementation to the model.
        image_token_id (int | None, default=None): Token ID for image patches. If "None", will be inferred from model config.
        image_patch_fn (tuple[int, int] | Callable | None, default=None):
            If a tuple of (height, width), will use static patch shapes for all images. If a callable is provided,
            it will be called with (image_count, **forward_pass_kwargs) to get the patch shape at runtime.
            If "None", will be inferred from model config.
        layer_reduce (Reduction, default="mean"): Method to reduce attention across layers ("mean", "max", etc.).
        head_reduce (Reduction, default="mean"): Method to reduce attention across heads ("mean", "max", etc.).
        backend (Backend, default="auto"): Backend to use for saliency computation. "auto" will choose "triton" if available, otherwise "torch".
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        bind: bool = True,
        image_token_id: int | None = None,
        image_patch_fn: tuple[int, int] | ImagePatchFunction | None = None,
        layer_reduce: Reduction = "mean",
        head_reduce: Reduction = "mean",
        backend: Backend = "auto",
    ):
        self.model = model
        self.layer_reduce: Reduction = layer_reduce
        self.head_reduce: Reduction = head_reduce
        self.backend: Backend = backend

        # Bind custom attention implementation to the model for saliency extraction if specified.
        if bind:
            self._prev_attn_implementation = model.config.text_config._attn_implementation
            model.set_attn_implementation({"text_config": "saliency"})
        else:
            self._prev_attn_implementation = None

        # Retrieve pad_token_id from model config for later use in context building
        self.pad_token_id = model.config.pad_token_id

        # Retrieve image_token_id to identify image vs text tokens.
        self.image_token_id = (
            image_token_id if image_token_id is not None else infer_image_token_id(model.config)
        )

        # Retrieve image_patch_fn to get patch shapes at runtime.
        if isinstance(image_patch_fn, tuple):
            self.image_patch_fn: ImagePatchFunction = StaticPatches(*image_patch_fn)
        elif callable(image_patch_fn):
            self.image_patch_fn = image_patch_fn
        else:
            self.image_patch_fn = infer_image_patch_fn(model.config)

        # Compute scale factor for attention scores based on model config
        self.scale = 1.0 / math.sqrt(model.config.hidden_size // model.config.num_attention_heads)

    def __call__(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        layer_reduce: Reduction | None = None,
        head_reduce: Reduction | None = None,
        **kwargs,
    ) -> SaliencyContext:
        """Create Saliency Context for given input, to be used to compute saliency maps during the forward pass.
        Called with the same inputs as the model's forward method, plus optional reduction methods for layers and heads.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor | None, optional): Input image pixel values. Defaults to None.
            layer_reduce (Reduction | None, optional): Method to reduce attention across layers. Defaults to None.
            head_reduce (Reduction | None, optional): Method to reduce attention across heads. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the image_patch_fn.
        Returns:
            SaliencyContext: The created SaliencyContext for the given input.
        """
        B, T_gen = input_ids.shape

        # Calculate patch shapes if pixel values are provided and have the expected shape
        if pixel_values is not None and pixel_values.ndim == 4 and pixel_values.shape[0] > 0:
            image_counts = pixel_values.shape[0]
            if image_counts != B:
                raise ValueError(
                    "Current implementation requires one image per input item, but got a different number of images and input items."
                    f"Number of images ({image_counts}) does not match number of input items ({B})."
                )

            patch_shapes = self.image_patch_fn(
                batch_size=B,
                image_count=image_counts,
                input_ids=input_ids,
                pixel_values=pixel_values,
                **kwargs,
            )
            if len(patch_shapes) != B:
                raise ValueError(
                    "image_patch_fn must return empty patch shapes for batch items without images."
                    f"Expected {B} patch shape entries, got {len(patch_shapes)}."
                )
        else:
            patch_shapes: list[list[tuple[int, int]]] = [[] for _ in range(B)]

        # Build saliency context tuned for the current input and model configuration
        context = SaliencyContext(
            input_ids=input_ids,
            pad_token_id=self.pad_token_id,
            image_token_id=self.image_token_id,
            attn_implementation=self._prev_attn_implementation or "eager",
            patch_shapes=patch_shapes,
            scale=self.scale,
            layer_reduce=layer_reduce or self.layer_reduce,
            head_reduce=head_reduce or self.head_reduce,
            backend=self.backend,
        )
        return context

    def unbind(self):
        """Restores bound model's original attention implementation if it was modified for saliency extraction."""
        if self._prev_attn_implementation is None:
            logger.warning("Model not bound to saliency computation. Skipping unbind...")
        else:
            self.model.set_attn_implementation({"text_config": self._prev_attn_implementation})
            self._prev_attn_implementation = None
