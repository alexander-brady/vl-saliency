from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from transformers import ProcessorMixin

from ..selectors import Selector
from ..transforms.pipe import TraceTransform
from .map import SaliencyMap


class Trace:
    """
    Captured attention and gradient data from a model inference.

    Attributes:
        attn (list[torch.Tensor]): List of attention tensors per image. Each tensor has shape [layers * heads, tokens, tokens].
        grad (list[torch.Tensor] | None, default=None): List of gradient tensors per image. Each tensor has shape [layers * heads, tokens, tokens] or None if gradients were not captured.
        processor (ProcessorMixin | None, default=None): The processor used for tokenization and decoding.
        image_token_id (int | None, default=None): The token ID used to represent image tokens.
        gen_start (int | None, default=None): The starting index of generated tokens in the sequence.
        generated_ids (torch.Tensor | None, default=None): The tensor of generated token IDs during inference.

    Methods:
        map(token, image_index, mode) -> SaliencyMap: Generate a SaliencyMap for a specific token using stored data.
        visualize_tokens(): Visualize the generated tokens using the processor.
    """

    def __init__(
        self,
        attn: list[torch.Tensor],
        grad: list[torch.Tensor] | None = None,
        *,
        processor: ProcessorMixin | None = None,
        image_token_id: int | None = None,
        gen_start: int | None = None,
        generated_ids: torch.Tensor | None = None,
    ):
        self.attn = attn  # list of [layers * heads, tokens, tokens] per image
        self.grad = grad  # list of [layers * heads, tokens, tokens] per image or None

        # Processor info
        self.processor = processor
        self.image_token_id = image_token_id

        # Generation info
        self.gen_start = gen_start
        self.generated_ids = generated_ids

    def apply(
        self,
        token: int | Selector,
        transform: TraceTransform,
        image_index: int = 0,
    ) -> SaliencyMap:
        """Apply a TraceTransform, returning a SaliencyMap."""
        if self.grad is None:
            raise ValueError("TraceTransforms needs grad to be stored")

        token = self._get_token_index(token)

        attn_map = self._get_tkn2img_map(token, image_index, "attn")
        grad_map = self._get_tkn2img_map(token, image_index, "grad")

        return transform(attn_map, grad_map)

    def _get_token_index(self, token: int | Selector) -> int:
        """Select desired token."""
        if isinstance(token, Selector):
            token = token.select(self)
        token = token + (self.gen_start or 0)
        return token

    def _get_tkn2img_map(self, token: int, image_index: int, mode: Literal["attn", "grad"]) -> SaliencyMap:
        """Get text-to-image saliency map."""
        if mode == "attn":
            tkn2img_map = self.attn[image_index]
        elif mode == "grad":
            if self.grad is None:
                raise ValueError("No gradient data stored in this trace.")
            tkn2img_map = self.grad[image_index]

        # Extract the token-to-image map
        tkn2img_map = tkn2img_map[:, :, token, :, :]  # [layers, heads, 1, H, W]
        tkn2img_map = tkn2img_map.squeeze(2)  # [layers, heads, H, W]

        return SaliencyMap(tkn2img_map)

    def map(
        self,
        token: int | Selector,
        image_index: int = 0,
        mode: Literal["attn", "grad"] = "attn",
    ) -> SaliencyMap:
        """
        Generate a SaliencyMap for a specific token using attention or gradient data.

        Args:
            token (int | Selector): The token index or a Selector instance to choose the token.
            image_index (int, default=0): The index of the image in the trace to use.
            mode (Literal["attn", "grad"], default="attn"): Whether to use attention or gradient data.

        Returns:
            SaliencyMap: The resulting saliency map for the specified token.

        Raises:
            ValueError: If the mode is "grad" and no gradient data is stored.
        """
        token = self._get_token_index(token)
        return self._get_tkn2img_map(token, image_index, mode)

    def visualize_tokens(self):
        """
        Visualize the generated tokens using the processor.

        Raises:
            ValueError: If the processor, generated_ids, or gen_start is not set.
        """
        if self.processor is None or self.generated_ids is None or self.gen_start is None:
            raise ValueError("Processor, generated_ids, and gen_start must be set to visualize tokens.")

        from ..viz.tokens import render_token_ids

        # Render the token IDs using the processor
        render_token_ids(
            generated_ids=self.generated_ids,
            processor=self.processor,
            gen_start=self.gen_start,
            skip_tokens=self.image_token_id,
            only_number_generated=True,
        )
