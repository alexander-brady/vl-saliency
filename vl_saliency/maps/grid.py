from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from jaxtyping import Float, Int
from torch import Tensor
from transformers import ProcessorMixin

from vl_saliency._core.seq_layout import SequenceLayout
from vl_saliency.maps.index import Index, IndexLike
from vl_saliency.maps.view import SaliencyView

if TYPE_CHECKING:
    from PIL.Image import Image


class SaliencyGrid:
    """Structured grid of saliency maps.

    Access patterns:
        grid[token_idx] → (H, W) for single-batch, single-image inputs
        grid[image_idx, token_idx] → (H, W) for single-batch inputs
        grid[batch_idx, image_idx, token_idx] → (H, W) for multi-batch inputs (general case)
    """

    def __init__(self, tensor: Float[Tensor, "B ... T_gen T_img"], layout: SequenceLayout):
        self._tensor = tensor
        self._layout = layout

        self._num_tokens: list[int] = [
            int(self._layout.gen_mask[b].sum().item()) for b in range(self._layout.B)
        ]

        # Simplified access patterns for single-batch or single-image scenarios
        self._single_batch = self._layout.B == 1
        self._single_images = all(len(p) == 1 for p in self._layout.patch_shapes)

    @property
    def batch_size(self) -> int:
        """Batch size of the saliency grid."""
        return self._layout.B

    def num_images(self, batch_idx: int | None = None) -> int:
        """Amount of images for the given batch index (or 0 if no images)."""
        batch_idx = self._normalize_idx("batch", batch_idx)
        return len(self._layout.patch_shapes[batch_idx])

    def num_tokens(self, batch_idx: int | None = None) -> int:
        """Amount of text tokens for the given batch index (or 0 if no generated tokens)."""
        batch_idx = self._normalize_idx("batch", batch_idx)
        return self._num_tokens[batch_idx]

    @overload
    def map(self, idx: Index, /) -> Float[Tensor, "... H W"]: ...
    @overload
    def map(
        self, c1: int, c2: int | None = None, c3: int | None = None, /
    ) -> Float[Tensor, "... H W"]: ...
    @overload
    def map(
        self,
        /,
        *,
        batch_idx: int | None = None,
        image_idx: int | None = None,
        token_idx: int | None = None,
    ) -> Float[Tensor, "... H W"]: ...

    def map(self, *args, **kwargs) -> Float[Tensor, "... H W"]:
        """
        Retrieves the saliency map for a specific image token index and batch item.

        Supports the following indexing formats:
            [token_idx]: Retrieves the saliency map for the specified token index, if batch size is 1 and there is only one image.
            [image_idx, token_idx]: Retrieves the saliency map for the specified image and token index, if there is only one batch item.
            [batch_idx, image_idx, token_idx]: Retrieves the saliency map for the specified batch, image, and token index.

        Returns:
            torch.Tensor: The saliency map for the specified token and image, of shape (H, W) where H and W are the height and width of the image patch.
        """

        batch_idx, image_idx, token_idx = self._normalize_idx_input(*args, **kwargs)
        if token_idx is None:
            raise IndexError("Token index must be specified to retrieve a saliency map.")

        H, W = self._layout.patch_shapes[batch_idx][image_idx]
        start = self._layout.image_token_offsets[batch_idx][image_idx]

        flat = self._tensor[batch_idx, ..., token_idx, start : start + H * W]  # [H * W]
        return flat.view(H, W)

    @overload
    def maps_for_image(self, idx: Index, /) -> Float[Tensor, "... T H W"]: ...
    @overload
    def maps_for_image(self, c1: int, c2: int | None = None, /) -> Float[Tensor, "... T H W"]: ...
    @overload
    def maps_for_image(
        self, /, *, batch_idx: int | None = None, image_idx: int | None = None
    ) -> Float[Tensor, "... T H W"]: ...

    def maps_for_image(self, *args, **kwargs) -> Float[Tensor, "... T H W"]:
        """
        Retrieves the saliency maps for all generated tokens for a specific image and batch item.

        Supports the following indexing formats:
            []: Retrieves the saliency maps for the only image and batch item, if there is only one of each.
            [image_idx]: Retrieves the saliency maps for the specified image index, if batch size is 1 and there is only one image.
            [batch_idx, image_idx]: Retrieves the saliency maps for the specified batch and image index.

        Returns:
            torch.Tensor: The saliency maps for all generated tokens for the specified image, of shape (T, H, W) where T is the number of generated tokens and H and W are the height and width of the image patch.
        """
        batch_idx, image_idx, _ = self._normalize_idx_input(*args, expect_token=False, **kwargs)
        n_tokens = self.num_tokens(batch_idx)

        H, W = self._layout.patch_shapes[batch_idx][image_idx]
        start = self._layout.image_token_offsets[batch_idx][image_idx]
        flat = self._tensor[batch_idx, ..., :n_tokens, start : start + H * W]  # [n_tokens, H * W]
        return flat.view(n_tokens, H, W)

    @overload
    def view(
        self,
        idx: Index,
        /,
        *,
        input_ids: Int[Tensor, "B T"] | Int[Tensor, " T"] | None = None,
        processor: ProcessorMixin | None = None,
        image: Image | None = None,
    ) -> SaliencyView: ...
    @overload
    def view(
        self,
        c1: int,
        c2: int | None = None,
        /,
        *,
        input_ids: Int[Tensor, "B T"] | Int[Tensor, " T"] | None = None,
        processor: ProcessorMixin | None = None,
        image: Image | None = None,
    ) -> SaliencyView: ...
    @overload
    def view(
        self,
        /,
        *,
        batch_idx: int | None = None,
        image_idx: int | None = None,
        input_ids: Int[Tensor, "B T"] | Int[Tensor, " T"] | None = None,
        image: Image | None = None,
        processor: ProcessorMixin | None = None,
    ) -> SaliencyView: ...

    def view(
        self,
        *args,
        input_ids: Int[Tensor, "B T"] | Int[Tensor, " T"] | None = None,
        image: Image | None = None,
        processor: ProcessorMixin | None = None,
        **kwargs,
    ) -> SaliencyView:
        """
        Creates a SaliencyView for the specified batch and image index,
        which provides convenient access to saliency maps within that scope.
        """
        batch_idx, image_idx, _ = self._normalize_idx_input(*args, expect_token=False, **kwargs)
        return SaliencyView(
            self,
            batch_idx=batch_idx,
            image_idx=image_idx,
            input_ids=input_ids,
            image=image,
            processor=processor,
        )

    def __getitem__(self, idx: IndexLike) -> Float[Tensor, "... H W"]:
        """
        Direct indexing to retrieve the saliency map.

        Supports the following indexing formats:
            [token_idx]: Retrieves the saliency map for the specified token index, if batch size is 1 and there is only one image.
            [image_idx, token_idx]: Retrieves the saliency map for the specified image and token index, if there is only one batch item.
            [batch_idx, image_idx, token_idx]: Retrieves the saliency map for the specified batch, image, and token index.
        """
        return self.map(Index.from_indices(idx))

    def __repr__(self) -> str:
        return f"SaliencyGrid(batch_size={self.batch_size})"

    def _normalize_idx(self, kind: Literal["batch", "image"], idx: int | None) -> int:
        """Normalizes the index, handling the case where idx is None for single-{kind} scenarios."""
        if idx is None:
            if (kind == "batch" and self._single_batch) or (
                kind == "image" and self._single_images
            ):
                return 0
            else:
                raise IndexError(
                    f"{kind.capitalize()} index must be specified for multi-{kind} saliency grids."
                )
        return idx

    def _normalize_idx_input(
        self, *args, expect_token: bool = True, **kwargs
    ) -> tuple[int, int, int | None]:
        """
        Builds and validates the input index.

        Args:
            args: Positional index components (batch_idx, image_idx, token_idx) from left to right, or an Index object.
            expect_token: Whether to expect a token index in the arguments
            kwargs: Keyword index components (batch_idx, image_idx, token_idx).

        Returns:
            A tuple of (batch_idx, image_idx, token_idx)
        """

        if args:
            if kwargs:
                raise IndexError("Cannot mix positional and keyword arguments for indexing.")
            if isinstance(args[0], Index):
                args = args[0]  # Unpack Index object if provided as the first argument
            elif not expect_token:
                args = args + (-1,)  # Pad unused token_idx with -1 to align indices

        batch_idx, image_idx, token_idx = Index.from_indices(args) if args else Index(**kwargs)

        batch_idx = self._normalize_idx("batch", batch_idx)
        image_idx = self._normalize_idx("image", image_idx)

        return batch_idx, image_idx, token_idx if expect_token else None
