import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency._types import Backend, HeadOp, LayerOp, Reduction
from vl_saliency.backends.dispatcher import assign_auto, get_saliency_qk
from vl_saliency.utils.logger import get_logger

logger = get_logger(__name__)


class SaliencyContext:
    """
    Context object that holds necessary information for saliency extraction during the forward pass.

    Args:
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        pad_token_id (int): The token ID used for padding, to identify and ignore padding tokens.
        image_token_id (int): The token ID used for image tokens, to identify which tokens correspond to images.
        patch_shapes (list[list[tuple[int, int]]]): A list of lists containing the patch shapes (height, width) for each image in each batch item. Outer list is over batch items, inner list is over images within that item.
        attn_implementation (str): The attention implementation to use for saliency extraction (e.g., "sdpa", "eager"). This is used to ensure model's standard output remains unchanged while computing saliency.

    Methods:
        reset(): Resets the saliency map to zeros. Should be called at the start of each forward pass if reusing the same context object.
        qk_step(q, k): Computes the saliency map for the given query and key tensors using the configured backend and reduction methods.
        get_map(token: int, batch_idx: int = 0, img_idx: int = 0): Retrieves the saliency map for a specific generated token and image token, returning it as a 2D tensor of shape (height, width) corresponding to the patch layout of the image.
    """

    def __init__(
        self,
        input_ids: Float[Tensor, "B S"],
        pad_token_id: int,
        image_token_id: int,
        patch_shapes: list[list[tuple[int, int]]],
        scale: float,
        head_reduce: Reduction = "mean",
        head_op: HeadOp | None = None,
        layer_reduce: Reduction = "mean",
        layer_op: LayerOp | None = None,
        attn_implementation: str = "sdpa",
        backend: Backend = "auto",
    ) -> None:
        self.patch_shapes = patch_shapes

        self.scale = scale
        self.attn_implementation = attn_implementation
        self.layer_reduce = layer_reduce

        # Build indices and masks to identify image and generated tokens
        self._build_indices(input_ids, pad_token_id, image_token_id)
        self.reset()

        # Store saliency computation function based on the specified backend and reduction methods
        self._set_qk_fn(
            backend=backend,
            head_reduce=head_reduce,
            layer_reduce=layer_reduce,
            head_op=head_op,
            layer_op=layer_op,
        )

        # Precompute image token offsets for efficient indexing
        self.image_token_offsets: list[list[int]] = []
        for b in range(self.B):
            offsets = [0]
            for h, w in self.patch_shapes[b]:
                offsets.append(offsets[-1] + h * w)
            self.image_token_offsets.append(offsets)

    def reset(self):
        """Resets the saliency map to zeros."""
        self.saliency = torch.zeros(
            (self.B, self.T_gen, self.T_img),
            device=self.device,
        )
        self.updates = 0  # Track updates for layer reduction (e.g., averaging across layers)

    def get_map(self, token: int, batch_idx: int = 0, img_idx: int = 0) -> torch.Tensor:
        """
        Returns the saliency map for the specified text token and image token.

        Args:
            token (int): The index of the text token to retrieve saliency for.
            batch_idx (int, default=0): The index of the batch item to retrieve saliency for.
            img_idx (int, default=0): The index of the image token within the batch item to retrieve saliency for.

        Returns:
            torch.Tensor: A 2D tensor of shape (height, width) representing the saliency map for the specified text token and image token.
        """

        # Validate indices
        if batch_idx >= self.B:
            raise ValueError(f"Batch index {batch_idx} is out of bounds (max {self.B - 1}).")
        if not (0 <= img_idx < len(self.patch_shapes[batch_idx])):
            raise ValueError(
                f"Image index {img_idx} is out of bounds for batch index {batch_idx} (max {len(self.patch_shapes[batch_idx]) - 1})."
            )
        if not (0 <= token < self.T_gen) or not self.gen_mask[batch_idx, token]:
            raise ValueError(
                f"Generated token index {token} is out of bounds for batch index {batch_idx}."
            )

        # Extract relevant slice of saliency map
        H, W = self.patch_shapes[batch_idx][img_idx]
        image_token_start = self.image_token_offsets[batch_idx][img_idx]
        image_tokens = self.saliency[
            batch_idx, token, image_token_start : image_token_start + H * W
        ]

        # Average saliency across all layers
        if self.layer_reduce == "mean" and self.updates > 0:
            image_tokens = image_tokens / self.updates

        return image_tokens.view(H, W)

    def qk_step(self, q: Float[Tensor, "B Hq T D"], k: Float[Tensor, "B Hkv T D"]):
        """Compute the saliency map for the given query and key tensors using the configured backend and reduction methods."""
        self.updates += 1
        self._saliency_qk_fn(
            q,
            k,
            gen_idx=self.gen_token_idx,
            gen_mask=self.gen_mask,
            img_idx=self.img_token_idx,
            img_mask=self.img_mask,
            scale=self.scale,
            saliency=self.saliency,
        )

    def _set_qk_fn(
        self,
        backend: Backend,
        head_reduce: Reduction,
        layer_reduce: Reduction,
        head_op: HeadOp | None,
        layer_op: LayerOp | None,
    ):
        """Set the backend for saliency computation, allowing dynamic switching between implementations."""

        if backend == "auto":
            backend = assign_auto(self.device)
            logger.info_once(f"Auto-assigned backend '{backend}'.")

        self._saliency_qk_fn = get_saliency_qk(
            backend,
            head_reduce=head_reduce,
            layer_reduce=layer_reduce,
            head_op=head_op,  # For now, we don't support custom head/layer ops with compiled backends
            layer_op=layer_op,
        )

    def _build_indices(
        self, input_ids: Float[Tensor, "B S"], pad_token_id: int, image_token_id: int
    ):
        """Builds index tensors and masks to identify image and generated tokens in the input sequences, while minimizing padding."""

        device = input_ids.device
        B, S = input_ids.shape

        # Masks
        is_pad = input_ids == pad_token_id  # [B, S]
        is_img = input_ids == image_token_id  # [B, S]

        # Last image position per sequence (or S if no image tokens)
        rev_idx = is_img.flip(dims=[1]).float().argmax(dim=1)  # [B]
        has_img = is_img.any(dim=1)  # [B]

        last_img = S - 1 - rev_idx  # [B]
        last_img = torch.where(has_img, last_img, torch.full_like(last_img, S))

        # Generated token mask
        pos = torch.arange(S, device=device)  # [S]
        is_gen = (~is_pad) & (~is_img) & (pos.unsqueeze(0) > last_img.unsqueeze(1))  # [B, S]

        def compact(mask: Bool[Tensor, "B S"]) -> tuple[Int[Tensor, "B"], Bool[Tensor, "B S"], int]:
            """Compacts the mask to minimize padding, returning new lengths and a compacted mask."""
            counts = mask.sum(dim=1)  # [B]
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

        self.gen_token_idx, self.gen_mask, self.T_gen = compact(is_gen)
        self.img_token_idx, self.img_mask, self.T_img = compact(is_img)

        self.B = B
        self.device = device
