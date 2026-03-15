import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency._types import PatchLayoutFn
from vl_saliency.config import SaliencyConfig


class SequenceLayout:
    """Computes token layout information useful for saliency extraction.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        **kwargs: Additional keyword arguments from the forward pass.

    """

    def __init__(
        self,
        config: SaliencyConfig,
        input_ids: Int[Tensor, "B S"],
        **kwargs,
    ):
        self.device = input_ids.device
        self.B, self.S = input_ids.shape

        self.pad_token_id = config.pad_token_id
        self.image_token_id = config.image_token_id

        is_img, is_gen = self._build_masks(input_ids, config.pad_token_id, config.image_token_id)
        self.img_token_idx, self.img_mask, self.T_img = self._compact_mask_indices(is_img)
        self.gen_token_idx, self.gen_mask, self.T_gen = self._compact_mask_indices(is_gen)

        self.patch_shapes = self._patch_shapes(config.patch_layout_fn, input_ids, **kwargs)
        self.image_token_offsets = self._image_offsets(self.patch_shapes)

    @staticmethod
    def _build_masks(input_ids: Int[Tensor, "B S"], pad_token_id: int, image_token_id: int):
        """Masks to identify image and generated tokens in the input sequences, while minimizing padding."""
        device = input_ids.device
        _, S = input_ids.shape

        # Masks
        is_pad = input_ids == pad_token_id  # [B, S]
        is_img = input_ids == image_token_id  # [B, S]

        # Last image position per sequence (or S if no image tokens)
        rev_idx = is_img.flip(1).int().argmax(1)  # [B]
        has_img = is_img.any(dim=1)  # [B]

        last_img = S - 1 - rev_idx  # [B]
        last_img = torch.where(has_img, last_img, torch.full_like(last_img, S))

        # Generated token mask
        pos = torch.arange(S, device=device)  # [S]
        is_gen = (~is_pad) & (~is_img) & (pos.unsqueeze(0) > last_img.unsqueeze(1))  # [B, S]

        return is_img, is_gen

    @staticmethod
    def _compact_mask_indices(
        mask: Bool[Tensor, "B S"],
    ) -> tuple[Int[Tensor, "B T"], Bool[Tensor, "B T"], int]:
        """
        Compacts the mask to minimize padding, returning new lengths and a compacted mask.

        Returns:
        - indices: Tensor of shape [B, T] containing the original indices of the masked tokens, padded with -1.
        - out_mask: Boolean tensor of shape [B, T] indicating valid token positions (True for valid tokens, False for padding).
        - T: The maximum number of valid tokens across the batch after compaction.
        """
        device = mask.device
        counts = mask.sum(dim=1)  # [B]

        B = mask.shape[0]
        T = int(counts.max().item())

        if T == 0:
            return (
                torch.empty((B, 0), dtype=torch.int32, device=device),
                torch.empty((B, 0), dtype=torch.bool, device=device),
                0,
            )

        # Column indices after compaction
        col = mask.cumsum(dim=1) - 1  # [B, S]
        out = torch.full((B, T), -1, dtype=torch.int32, device=device)  # [B, T]

        rows, cols = mask.nonzero(as_tuple=True)  # [total_count]

        out[rows, col[rows, cols]] = cols.to(torch.int32)  # [B, T]
        out_mask = torch.arange(T, device=device).unsqueeze(0) < counts.unsqueeze(1)  # [B, T]
        return out, out_mask, T

    @staticmethod
    def _patch_shapes(
        patch_layout_fn: PatchLayoutFn,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        **kwargs,
    ) -> list[list[tuple[int, int]]]:
        """Applies the patch_layout_fn to get patch shapes for each image in the batch."""
        batch_size = input_ids.shape[0]
        patch_shapes: list[list[tuple[int, int]]]

        if pixel_values is not None and pixel_values.ndim == 4 and pixel_values.shape[0] > 0:
            image_counts = pixel_values.shape[0]
            if image_counts != batch_size:
                raise ValueError(
                    "Current implementation requires one image per input item, but got a different number of images and input items."
                    f"Number of images ({image_counts}) does not match number of input items ({batch_size})."
                )

            patch_shapes = patch_layout_fn(
                batch_size=batch_size,
                image_count=image_counts,
                input_ids=input_ids,
                pixel_values=pixel_values,
                **kwargs,
            )
            if len(patch_shapes) != batch_size:
                raise ValueError(
                    "patch_layout_fn must return empty patch shapes for batch items without images."
                    f"Expected {batch_size} patch shape entries, got {len(patch_shapes)}."
                )
        else:
            patch_shapes = [[] for _ in range(batch_size)]

        return patch_shapes

    @staticmethod
    def _image_offsets(patch_shapes: list[list[tuple[int, int]]]) -> list[list[int]]:
        """Builds a list of image token offsets for each sequence in the batch based on the provided patch shapes."""
        offsets = []
        for b in range(len(patch_shapes)):
            seq_offsets = [0]
            for h, w in patch_shapes[b]:
                seq_offsets.append(seq_offsets[-1] + h * w)
            offsets.append(seq_offsets)
        return offsets
