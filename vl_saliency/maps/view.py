from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Protocol, overload, runtime_checkable

from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizerBase, ProcessorMixin

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from PIL.Image import Image

    from vl_saliency.maps.grid import SaliencyGrid


@runtime_checkable
class Selector(Protocol):
    def __call__(self, view: SaliencyView) -> int: ...


class SaliencyView:
    """Saliency grid scoped to a specific image (and batch),
    providing convenient access to its saliency maps.

    Args:
    - saliency_grid: The underlying SaliencyGrid to scope.
    - batch_idx: Optional batch index to scope to. If None, defaults to 0.
    - image_idx: Optional image index to scope to. If None, defaults to 0
    - input_ids: Optional input IDs tensor for token decoding (shape: [B, T] | [T]).
    - image: Optional PIL image corresponding to the scoped image index, for visualization purposes.
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
        image: Image | None = None,
        processor: ProcessorMixin | PreTrainedTokenizerBase | None = None,
    ):
        self.batch_idx, self.image_idx, _ = saliency_grid._normalize_idx_input(
            batch_idx=batch_idx, image_idx=image_idx
        )
        self.image = image
        self.saliency_grid = saliency_grid

        self.input_ids = (
            input_ids[self.batch_idx]
            if input_ids is not None and input_ids.ndim == 2
            else input_ids
        )

        self._tok: PreTrainedTokenizerBase | None = (
            processor.tokenizer  # type: ignore[attr-defined]
            if processor and hasattr(processor, "tokenizer")
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
    def maps(self) -> Float[Tensor, "... T H W"]:
        """ "Saliency maps for all tokens of the scoped image."""
        return self.saliency_grid.maps_for_image(self.batch_idx, self.image_idx)

    def map(self, token_idx: int | str | Selector) -> Float[Tensor, "... H W"]:
        """Map for the specified token index of the scoped image."""
        if isinstance(token_idx, str):
            from vl_saliency.select.factories import regex

            token_idx = regex(token_idx)
        if callable(token_idx):
            token_idx = token_idx(self)
        return self.saliency_grid.map(self.batch_idx, self.image_idx, token_idx)

    def plot(self, token_idx: int | Selector, image: Image | None = None, **plot_kwargs) -> Figure:
        """
        Plot the saliency map for the specified token index of the scoped image, optionally overlaying it on a provided/scoped image.

        Args:
        - token_idx: The token index or a Selector to specify which token's saliency map to plot.
        - image: Optional PIL image to overlay the saliency map on. If None, uses the image associated with the scoped grid if available.
        - plot_kwargs: Additional keyword arguments to pass to matplotlib's plotting function for customizing the visualization.

        Returns:
        - Figure: A matplotlib Figure object containing the plotted saliency map.
        """
        from vl_saliency.viz.overlay import plot

        map = self.map(token_idx)
        image = image if image is not None else self.image
        return plot(map, image=image, **plot_kwargs)

    @overload
    def visualize_tokens(self, return_html: Literal[True]) -> str: ...

    @overload
    def visualize_tokens(self, return_html: Literal[False] = False) -> None: ...

    def visualize_tokens(self, return_html: bool = False) -> str | None:
        """Renders the input tokens for the scoped image as HTML with tooltips showing token IDs and decoded text. Requires input IDs and a tokenizer to be available.

        Args:
            return_html (bool, optional): Return the rendered HTML string instead of displaying it. Defaults to False.
        """
        if self.input_ids is None or self._tok is None:
            raise ValueError("Input IDs and tokenizer are required to visualize tokens.")

        skip_tokens = (
            self.saliency_grid._layout.pad_token_id,
            self.saliency_grid._layout.image_token_id,
        )
        from vl_saliency.viz.tokens import render_token_ids

        return render_token_ids(
            token_ids=self.input_ids.tolist(),
            processor=self._tok,
            return_html=return_html,  # type: ignore
            gen_start=self.gen_start_idx,
            skip_tokens=skip_tokens,
            only_number_generated=True,
        )

    def __getitem__(self, token_idx: int | Selector | str) -> Float[Tensor, "... H W"]:
        """Map for the specified token index of the scoped image."""
        return self.map(token_idx)

    def __len__(self) -> int:
        return self.num_tokens

    def __iter__(self) -> Iterator[Float[Tensor, "... H W"]]:
        for i in range(self.num_tokens):
            yield self.map(i)
