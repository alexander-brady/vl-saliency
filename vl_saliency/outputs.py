from __future__ import annotations

from dataclasses import dataclass, is_dataclass

from jaxtyping import Float
from torch import Tensor
from transformers.utils.generic import ModelOutput

from vl_saliency.tokens import TokenLayout


@dataclass
class SaliencyOutput(ModelOutput):
    """Output wrapper for model outputs that includes the saliency map alongside the original model output.
    Fields from the original model output can be accessed directly on this object,
    and the saliency map is available as the `saliency` attribute."""

    base_output: ModelOutput
    """The original output from the model's forward pass."""
    saliency: SaliencyGrid
    """Saliency map computed during the forward pass."""

    def __getattr__(self, name):
        if is_dataclass(self.base_output) and name in self.base_output.__dataclass_fields__:
            return getattr(self.base_output, name)
        return super().__getattribute__(name)


class SaliencyGrid:
    """Structured grid of saliency maps.

    Access patterns:
        grid[token_idx] → (H, W) for single-batch, single-image scenario
        grid[img_idx, token_idx] → (H, W) for single-batch scenario
        grid[batch_idx, img_idx, token_idx] → (H, W) for multi-batch scenario (general case)

    Methods:
        map(token_idx, batch_idx=None, img_idx=None) → (H, W) saliency map for the specified token and image
        image_maps(img_idx, batch_idx=None) → (T, H, W) saliency maps for all tokens for the specified image
    """

    def __init__(self, tensor: Float[Tensor, "B T_gen T_img"], layout: TokenLayout):
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

    def map(
        self, token_idx: int, *, batch_idx: int | None = None, img_idx: int | None = None
    ) -> Float[Tensor, "H W"]:
        """
        Retrieves the saliency map for a specific image token index and batch item.

        Args:
            token_idx (int): The index of the generated token for which to retrieve the saliency map.
            batch_idx (int, optional): The index of the batch item. Defaults to 0.
            img_idx (int, optional): The index of the image token within the batch item. Defaults to 0.

        Returns:
            torch.Tensor: The saliency map for the specified token and image, of shape (H, W) where H and W are the height and width of the image patch.
        """
        batch_idx, img_idx, token_idx = self._validate_indices(batch_idx, img_idx, token_idx)

        H, W = self._layout.patch_shapes[batch_idx][img_idx]
        start = self._layout.image_token_offsets[batch_idx][img_idx]
        flat = self._tensor[batch_idx, token_idx, start : start + H * W]  # [H * W]
        return flat.view(H, W)

    def image_maps(
        self, img_idx: int | None = None, batch_idx: int | None = None
    ) -> Float[Tensor, "T H W"]:
        """
        Retrieves the saliency maps for all generated tokens for a specific image token index and batch item.

        Args:
            img_idx (int, optional): The index of the image token within the batch item. Defaults to 0. If batch has multiple images, this must be specified.
            batch_idx (int, optional): The index of the batch item. Defaults to 0. If batch has multiple items, this must be specified.

        Returns:
            torch.Tensor: The saliency maps for all generated tokens for the specified image, of shape (T, H, W) where T is the number of generated tokens and H and W are the height and width of the image patch.
        """
        batch_idx = self._validate_batch_idx(batch_idx)
        img_idx = self._validate_img_idx(batch_idx, img_idx)
        n_tokens = self.num_tokens(batch_idx)

        H, W = self._layout.patch_shapes[batch_idx][img_idx]
        start = self._layout.image_token_offsets[batch_idx][img_idx]
        flat = self._tensor[batch_idx, :n_tokens, start : start + H * W]  # [n_tokens, H * W]
        return flat.view(n_tokens, H, W)

    def __getitem__(
        self, idx: int | tuple[int, int] | tuple[int, int, int]
    ) -> Float[Tensor, "H W"]:
        """
        Direct indexing to retrieve the saliency map.

        Supports the following indexing formats:
            [token_idx]: Retrieves the saliency map for the specified token index, if batch size is 1 and there is only one image.
            [img_idx, token_idx]: Retrieves the saliency map for the specified image and token index, if there is only one batch item.
            [batch_idx, img_idx, token_idx]: Retrieves the saliency map for the specified batch, image, and token index.
        """

        batch_idx, img_idx = None, None
        if isinstance(idx, int):
            token_idx = idx
        elif isinstance(idx, tuple) and len(idx) == 2:
            img_idx, token_idx = idx
        elif isinstance(idx, tuple) and len(idx) == 3:
            batch_idx, img_idx, token_idx = idx
        else:
            raise ValueError(
                "Invalid index format. Use [token_idx], [img_idx, token_idx], or [batch_idx, img_idx, token_idx]."
            )

        batch_idx, img_idx, token_idx = self._validate_indices(batch_idx, img_idx, token_idx)
        return self.map(token_idx, batch_idx=batch_idx, img_idx=img_idx)

    def _validate_batch_idx(self, batch_idx: int | None) -> int:
        """Validates and returns the batch index, handling the case where batch_idx is None for single-batch scenarios."""
        if batch_idx is None:
            if self._single_batch:
                return 0
            else:
                raise ValueError("Batch index must be specified for multi-batch saliency grids.")
        elif batch_idx < 0 or batch_idx >= self._layout.B:
            raise ValueError(
                f"Batch index {batch_idx} is out of bounds for batch size {self._layout.B}."
            )
        return batch_idx

    def _validate_img_idx(self, batch_idx: int, img_idx: int | None) -> int:
        """Validates the image index for the given batch index."""
        num_imgs = self.num_images(batch_idx)
        if img_idx is None:
            if self._single_images:
                img_idx = 0
            else:
                raise ValueError("Image index must be specified for multi-image saliency grids.")
        elif img_idx < 0 or img_idx >= num_imgs:
            raise ValueError(
                f"Image index {img_idx} is out of bounds for batch index {batch_idx} with {num_imgs} images."
            )
        return img_idx

    def _validate_token_idx(self, batch_idx: int, img_idx: int, token_idx: int) -> int:
        """Validates the token index for the given batch index."""
        num_tokens = self.num_tokens(batch_idx)
        if token_idx < 0 or token_idx >= num_tokens:
            raise ValueError(
                f"Token index {token_idx} is out of bounds for batch index {batch_idx} with {num_tokens} tokens."
            )
        H, W = self._layout.patch_shapes[batch_idx][img_idx]
        start = self._layout.image_token_offsets[batch_idx][img_idx]
        if token_idx < start + H * W:
            raise ValueError(
                f"Token index {token_idx} does not attend to previous image tokens for batch index {batch_idx} and image index {img_idx}."
            )
        return token_idx

    def _validate_indices(
        self, batch_idx: int | None, img_idx: int | None, token_idx: int
    ) -> tuple[int, int, int]:
        """Validates and returns the batch index, image index, and token index."""
        batch_idx = self._validate_batch_idx(batch_idx)
        img_idx = self._validate_img_idx(batch_idx, img_idx)
        token_idx = self._validate_token_idx(batch_idx, img_idx, token_idx)
        return batch_idx, img_idx, token_idx
