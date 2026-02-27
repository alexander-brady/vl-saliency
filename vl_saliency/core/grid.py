from __future__ import annotations

from typing import Literal, cast, overload

from jaxtyping import Float
from torch import Tensor

from vl_saliency.core.index import Index, IndexLike
from vl_saliency.core.layout import SequenceLayout


class SaliencyGrid:
    """Structured grid of saliency maps.

    Access patterns:
        grid[token_idx] → (H, W) for single-batch, single-image scenario
        grid[image_idx, token_idx] → (H, W) for single-batch scenario
        grid[batch_idx, image_idx, token_idx] → (H, W) for multi-batch scenario (general case)

    Methods:
        map(token_idx, batch_idx=None, image_idx=None) → (H, W) saliency map for the specified token and image
        image_maps(image_idx, batch_idx=None) → (T, H, W) saliency maps for all tokens for the specified image
    """

    def __init__(self, tensor: Float[Tensor, "B T_gen T_img"], layout: SequenceLayout):
        self._tensor = tensor
        self._layout = layout

        # Simplified access patterns for single-batch or single-image scenarios
        self._single_batch = self._layout.B == 1
        self._single_images = all(len(p) == 1 for p in self._layout.patch_shapes)

    @property
    def batch_size(self) -> int:
        """Batch size of the saliency grid."""
        return self._layout.B

    def num_images(self, batch_idx: int | None = None) -> int:
        """Amount of images for the given batch index (or 0 if no images)."""
        batch_idx = self._validate_batch_idx(batch_idx)
        return len(self._layout.patch_shapes[batch_idx])

    def num_tokens(self, batch_idx: int | None = None) -> int:
        """Amount of text tokens for the given batch index (or 0 if no generated tokens)."""
        batch_idx = self._validate_batch_idx(batch_idx)
        return int(self._layout.gen_mask[batch_idx].sum().item())

    @overload
    def map(self, idx: Index) -> Float[Tensor, "H W"]: ...

    @overload
    def map(self, idx: int, *components: int) -> Float[Tensor, "H W"]: ...

    def map(self, idx: int | Index, *components: int) -> Float[Tensor, "H W"]:
        """
        Retrieves the saliency map for a specific image token index and batch item.

        Supports the following indexing formats:
            [token_idx]: Retrieves the saliency map for the specified token index, if batch size is 1 and there is only one image.
            [image_idx, token_idx]: Retrieves the saliency map for the specified image and token index, if there is only one batch item.
            [batch_idx, image_idx, token_idx]: Retrieves the saliency map for the specified batch, image, and token index.

        Returns:
            torch.Tensor: The saliency map for the specified token and image, of shape (H, W) where H and W are the height and width of the image patch.
        """
        if isinstance(idx, int):
            if len(components) > 2:
                raise IndexError(
                    "Too many indices provided. Expected at most 3 (batch_idx, image_idx, token_idx)."
                )
            idx = Index.from_indices(cast(IndexLike, (idx,) + components))

        batch_idx, image_idx, token_idx = self._validate_index(idx)

        H, W = self._layout.patch_shapes[batch_idx][image_idx]
        start = self._layout.image_token_offsets[batch_idx][image_idx]
        flat = self._tensor[batch_idx, token_idx, start : start + H * W]  # [H * W]
        return flat.view(H, W)

    @overload
    def maps_for_image(
        self, idx: Index, component: Literal[None] = None
    ) -> Float[Tensor, "T H W"]: ...

    @overload
    def maps_for_image(self, idx: int, component: int | None = None) -> Float[Tensor, "T H W"]: ...

    @overload
    def maps_for_image(
        self, idx: Literal[None], component: Literal[None]
    ) -> Float[Tensor, "T H W"]: ...

    def maps_for_image(
        self, idx: int | Index | None = None, component: int | None = None
    ) -> Float[Tensor, "T H W"]:
        """
        Retrieves the saliency maps for all generated tokens for a specific image and batch item.

        Supports the following indexing formats:
            []: Retrieves the saliency maps for the only image and batch item, if there is only one of each.
            [image_idx]: Retrieves the saliency maps for the specified image index, if batch size is 1 and there is only one image.
            [batch_idx, image_idx]: Retrieves the saliency maps for the specified batch and image index.

        Returns:
            torch.Tensor: The saliency maps for all generated tokens for the specified image, of shape (T, H, W) where T is the number of generated tokens and H and W are the height and width of the image patch.
        """

        if isinstance(idx, int):
            batch_idx, image_idx = (idx, component) if component is not None else (None, idx)
            idx = Index(batch_idx=batch_idx, image_idx=image_idx)
        elif idx is None:
            idx = Index(batch_idx=None, image_idx=None)

        batch_idx = self._validate_batch_idx(idx.batch_idx)
        image_idx = self._validate_image_idx(batch_idx, idx.image_idx)
        n_tokens = self.num_tokens(batch_idx)

        H, W = self._layout.patch_shapes[batch_idx][image_idx]
        start = self._layout.image_token_offsets[batch_idx][image_idx]
        flat = self._tensor[batch_idx, :n_tokens, start : start + H * W]  # [n_tokens, H * W]
        return flat.view(n_tokens, H, W)

    def __getitem__(self, idx: IndexLike) -> Float[Tensor, "H W"]:
        """
        Direct indexing to retrieve the saliency map.

        Supports the following indexing formats:
            [token_idx]: Retrieves the saliency map for the specified token index, if batch size is 1 and there is only one image.
            [image_idx, token_idx]: Retrieves the saliency map for the specified image and token index, if there is only one batch item.
            [batch_idx, image_idx, token_idx]: Retrieves the saliency map for the specified batch, image, and token index.
        """
        return self.map(Index.from_indices(idx))

    def _validate_batch_idx(self, batch_idx: int | None) -> int:
        """Validates and returns the batch index, handling the case where batch_idx is None for single-batch scenarios."""
        if batch_idx is None:
            if self._single_batch:
                return 0
            else:
                raise IndexError("Batch index must be specified for multi-batch saliency grids.")
        elif batch_idx < 0 or batch_idx >= self._layout.B:
            raise IndexError(
                f"Batch index {batch_idx} is out of bounds for batch size {self._layout.B}."
            )
        return batch_idx

    def _validate_image_idx(self, batch_idx: int, image_idx: int | None) -> int:
        """Validates the image index for the given batch index."""
        num_imgs = self.num_images(batch_idx)
        if image_idx is None:
            if self._single_images:
                image_idx = 0
            else:
                raise IndexError("Image index must be specified for multi-image saliency grids.")
        elif image_idx < 0 or image_idx >= num_imgs:
            raise IndexError(
                f"Image index {image_idx} is out of bounds for batch index {batch_idx} with {num_imgs} images."
            )
        return image_idx

    def _validate_token_idx(self, batch_idx: int, token_idx: int) -> int:
        """Validates the token index for the given batch index."""
        num_tokens = self.num_tokens(batch_idx)
        if token_idx < 0 or token_idx >= num_tokens:
            raise IndexError(
                f"Token index {token_idx} is out of bounds for batch index {batch_idx} with {num_tokens} tokens."
            )
        return token_idx

    def _validate_index(self, index: IndexLike) -> tuple[int, int, int]:
        """Validates and returns the batch index, image index, and token index."""
        index = Index.from_indices(index)
        batch_idx = self._validate_batch_idx(index.batch_idx)
        image_idx = self._validate_image_idx(batch_idx, index.image_idx)
        if index.token_idx is None:
            raise IndexError("Token index must be specified.")
        token_idx = self._validate_token_idx(batch_idx, index.token_idx)
        return batch_idx, image_idx, token_idx
