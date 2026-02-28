from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin

if TYPE_CHECKING:
    from vl_saliency.core.grid import SaliencyGrid


@runtime_checkable
class Selector(Protocol):
    """Selects token indices for a ScopedSaliencyGrid,
    enabling dynamic access patterns based on token content or position."""

    def __call__(self, scoped: ScopedSaliencyGrid) -> int: ...


class ScopedSaliencyGrid:
    """Saliency grid scoped to a specific image (and batch),
    providing convenient access to its saliency maps.

    Args:
    - saliency_grid: The underlying SaliencyGrid to scope.
    - batch_idx: Optional batch index to scope to. If None, defaults to 0.
    - image_idx: Optional image index to scope to. If None, defaults to 0
    - input_ids: Optional input IDs tensor for token decoding (shape: [B, T] | [T]).
    - processor: Optional processor with a tokenizer for token decoding. If not provided, token decoding
        will not be available.
    """

    def __init__(
        self,
        saliency_grid: SaliencyGrid,
        *,
        batch_idx: int | None = None,
        image_idx: int | None = None,
        input_ids: Int[Tensor, "B T"] | Int[Tensor, " T"] | None = None,
        processor: ProcessorMixin | PreTrainedTokenizerBase | None = None,
    ):
        self.batch_idx, self.image_idx, _ = saliency_grid._normalize_idx_input(
            batch_idx=batch_idx, image_idx=image_idx
        )
        self.saliency_grid = saliency_grid

        self.input_ids = (
            input_ids[self.batch_idx]
            if input_ids is not None and input_ids.ndim == 2
            else input_ids
        )

        self._tok: PreTrainedTokenizerBase | None = (
            processor.tokenizer  # type: ignore[attr-defined]
            if hasattr(processor, "tokenizer")
            else processor
        )

    @cached_property
    def _gen_indices(self) -> Int[Tensor, " T"]:
        """Token indices of generated tokens for the scoped image."""
        indices = self.saliency_grid._layout.gen_token_idx[self.batch_idx]  # [S]
        gen_mask = self.saliency_grid._layout.gen_mask[self.batch_idx].bool()  # [S]
        return indices[gen_mask]  # [T_gen]

    @cached_property
    def gen_start_idx(self) -> int:
        """Starting index of generated tokens for the scoped image."""
        return int(self._gen_indices.min().item())

    @cached_property
    def gen_end_idx(self) -> int:
        """Ending index of generated tokens for the scoped image."""
        return int(self._gen_indices.max().item()) + 1

    @cached_property
    def num_tokens(self) -> int:
        """Text tokens for the scoped image."""
        return self.saliency_grid.num_tokens(batch_idx=self.batch_idx)

    @cached_property
    def gen_tokens(self) -> Int[Tensor, " T"]:
        """Generated token IDs for the scoped image."""
        if self.input_ids is None:
            raise ValueError("Input IDs are required to access generated tokens.")
        return self.input_ids[self._gen_indices]  # [T_gen]

    @cached_property
    def decoded_input_tokens(self) -> list[str]:
        """Decoded input tokens for the scoped image."""
        if self.input_ids is None or self._tok is None:
            raise ValueError("Input IDs and tokenizer are required to decode tokens.")
        return self._tok.convert_ids_to_tokens(self.input_ids.tolist())  # type: ignore[union-attr]

    @cached_property
    def decoded_gen_tokens(self) -> list[str]:
        """Decoded generated tokens for the scoped image."""
        if self.input_ids is None or self._tok is None:
            raise ValueError("Input IDs and tokenizer are required to decode tokens.")

        return self._tok.convert_ids_to_tokens(self.gen_tokens.tolist())  # type: ignore[union-attr]

    @property
    def maps(self) -> Float[Tensor, "T H W"]:
        """ "Saliency maps for all tokens of the scoped image."""
        return self.saliency_grid.maps_for_image(self.batch_idx, self.image_idx)

    def map(self, token_idx: int | Selector) -> Float[Tensor, "H W"]:
        """Map for the specified token index of the scoped image."""
        if isinstance(token_idx, Selector):
            token_idx = token_idx(self)
        return self.saliency_grid.map(self.batch_idx, self.image_idx, token_idx)

    def __getitem__(self, token_idx: int | Selector) -> Float[Tensor, "H W"]:
        """Map for the specified token index of the scoped image."""
        return self.map(token_idx)
