from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ..utils.logger import get_logger
from .map import SaliencyMap

if TYPE_CHECKING:
    from transformers.processing_utils import ProcessorMixin

    from ..selectors.base import Selector
    from ..transforms.pipe import TraceTransform

logger = get_logger(__name__)


class Trace:
    """
    Captured attention and gradient data from a model inference.

    Attributes:
        attn (list[torch.Tensor] | None): List of attention tensors per image. Each tensor has shape [layers, heads, gen_tokens, img_tokens, img_tokens] or None if attention was not captured.
        grad (list[torch.Tensor] | None, default=None): List of gradient tensors per image. Each tensor has shape [layers, heads, gen_tokens, img_tokens, img_tokens] or None if gradients were not captured.
        processor (ProcessorMixin | None, default=None): The processor used for tokenization and decoding.
        image_token_id (int | None, default=None): The token ID used to represent image tokens.
        gen_start (int, default=0): The starting index of generated tokens in the sequence.
        generated_ids (torch.Tensor | None, default=None): The tensor of generated token IDs during inference.

    Methods:
        map(token, image_index, mode) -> SaliencyMap: Generate a SaliencyMap for a specific token using stored data.
        visualize_tokens(): Visualize the generated tokens using the processor.
    """

    def __init__(
        self,
        attn: list[torch.Tensor] | None,
        grad: list[torch.Tensor] | None = None,
        *,
        processor: ProcessorMixin | None = None,
        image_token_id: int | None = None,
        gen_start: int = 0,
        generated_ids: torch.Tensor | None = None,
    ):
        # Validate attn and grad shapes
        for attr_name, attr in [("attn", attn), ("grad", grad)]:
            if attr is not None:
                if not all(tensor.shape[:3] == attr[0].shape[:3] for tensor in attr):
                    raise ValueError(
                        f"All tensors in {attr_name} must have the same layers, heads, and generated token dimensions."
                    )

        # Validate attn and grad compatibility
        if attn is not None and grad is not None:
            if len(attn) != len(grad):
                raise ValueError("Attention and gradient lists must have the same length (number of images).")

            if any(a.shape != g.shape for a, g in zip(attn, grad, strict=True)):
                raise ValueError("Attention and gradient tensors must have the same shape for each image.")

        # Determine default mode
        if attn is None and grad is None:
            raise ValueError("At least one of attn or grad must be provided.")
        self._default: Literal["attn", "grad"] = "grad" if attn is None else "attn"

        # Stored data
        self.attn = attn  # list of [layers, heads, gen_tokens, img_tokens, img_tokens] per image or None
        self.grad = grad  # list of [layers, heads, gen_tokens, img_tokens, img_tokens] per image or None

        # Store metadata
        default = getattr(self, self._default)
        self.total_images = len(default)
        self.total_generated_tokens = default[0].shape[2]

        # Processor info
        self.processor = processor
        self.image_token_id = image_token_id

        # Generation info (for visualization only)
        self.gen_start = gen_start
        self.generated_ids = generated_ids

        # Ensure generated_ids is 1D and valid
        if generated_ids is not None:
            if generated_ids.ndim != 2 or generated_ids.size(0) != 1:
                logger.error("generated_ids must be a 2D tensor of token IDs with shape [1, T].")

            if generated_ids.shape[1] != self.gen_start + self.total_generated_tokens:
                logger.error(
                    f"generated_ids length {generated_ids.shape[1]} does not match gen_start + total_generated_tokens"
                    f"({self.gen_start} + {self.total_generated_tokens} = {self.gen_start + self.total_generated_tokens})."
                )

        # Ensure gen_start is valid
        max_len = generated_ids.shape[1] if generated_ids is not None else self.total_generated_tokens
        if not (0 <= self.gen_start < max_len):
            logger.error(f"gen_start ({self.gen_start}) must be between 0 and {max_len - 1} (total generated tokens).")

    def _get_token_index(self, token: int | Selector) -> int:
        """Select desired token (relative to generated tokens)."""
        if not isinstance(token, int):  # If token is a Selector
            token = token(self)

        if not 0 <= token < self.total_generated_tokens:
            raise IndexError("Token index out of range of generated tokens.")
        return token

    def _get_tkn2img_map(self, token: int, image_index: int, mode: Literal["attn", "grad"]) -> SaliencyMap:
        """Get text-to-image saliency map."""

        if getattr(self, mode) is None:
            raise ValueError(f"No {mode} data stored in this trace.")
        tkn2img_map = getattr(self, mode)[image_index]  # [layers, heads, gen_tokens, img_tokens, img_tokens]

        # Extract the token-to-image map
        tkn2img_map = tkn2img_map[:, :, token, :, :]  # [layers, heads, H, W]
        return SaliencyMap(tkn2img_map)

    def map(
        self,
        token: int | Selector,
        mode: Literal["attn", "grad"] | TraceTransform | None = None,
        image_index: int = 0,
    ) -> SaliencyMap:
        """
        Generate a SaliencyMap for a specific token using attention or gradient data.

        Args:
            token (int | Selector): The token index or a Selector instance to choose the token.
            image_index (int, default=0): The index of the image in the trace to use.
            mode (Literal["attn", "grad"] | TraceTransform | None, default=None): Whether to use attention or gradient data. If None, defaults to "attn" if attention data is available, otherwise "grad".

        Returns:
            SaliencyMap: The resulting saliency map for the specified token.

        Raises:
            ValueError: If the required data for the selected mode is not available.
            IndexError: If token or image_index is out of range.
        """
        if not 0 <= image_index < self.total_images:
            raise IndexError("Image index out of range.")

        token = self._get_token_index(token)

        if mode is None:
            mode = self._default

        if isinstance(mode, str):
            return self._get_tkn2img_map(token, image_index, mode)

        else:  # TraceTransform
            if self.attn is None or self.grad is None:
                raise ValueError("TraceTransforms needs both attention and gradient data stored in the trace.")

            attn = self._get_tkn2img_map(token, image_index, "attn")
            grad = self._get_tkn2img_map(token, image_index, "grad")

            return mode(attn, grad)

    def visualize_tokens(self):
        """
        Visualize the generated tokens using the processor.

        Raises:
            ValueError: If the processor, generated_ids, or gen_start is not set.
        """
        if self.processor is None or self.generated_ids is None:
            raise ValueError("Processor and generated_ids must be set to visualize tokens.")

        from ..viz.tokens import render_token_ids

        # Render the token IDs using the processor
        render_token_ids(
            generated_ids=self.generated_ids,
            processor=self.processor,
            gen_start=self.gen_start,
            skip_tokens=self.image_token_id,
            only_number_generated=True,
        )
