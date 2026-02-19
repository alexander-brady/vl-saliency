import torch

from vl_saliency._types import Backend, Reduction


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
        update(saliency: torch.Tensor): Updates the saliency map with the provided tensor. This is called internally by the attention function after computing the saliency map for the current layer.
        map(token: int, batch_idx: int = 0, img_idx: int = 0): Retrieves the saliency map for a specific generated token and image token, returning it as a 2D tensor of shape (height, width) corresponding to the patch layout of the image.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
        image_token_id: int,
        patch_shapes: list[list[tuple[int, int]]],
        scale: float,
        layer_reduce: Reduction = "mean",
        head_reduce: Reduction = "mean",
        attn_implementation: str = "sdpa",
        backend: Backend = "auto",
    ) -> None:
        self.patch_shapes = patch_shapes
        self.scale = scale

        self.layer_reduce: Reduction = layer_reduce
        self.head_reduce: Reduction = head_reduce

        self.attn_implementation = attn_implementation
        self.backend: Backend = backend

        # Build indices and masks to identify image and generated tokens
        self._build_indices(input_ids, pad_token_id, image_token_id)
        self.reset()

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

    def update(self, saliency: torch.Tensor):
        """Replace the current saliency map with the provided one, and increment the update counter."""
        self.saliency = saliency
        self.updates += 1

    def map(self, token: int, batch_idx: int = 0, img_idx: int = 0) -> torch.Tensor:
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

    def _build_indices(self, input_ids: torch.Tensor, pad_token_id: int, image_token_id: int):
        """Builds index tensors and masks to identify image and generated tokens in the input sequences, while minimizing padding."""
        self.device = input_ids.device
        self.B, S = input_ids.shape

        is_pad = input_ids == pad_token_id
        is_img = input_ids == image_token_id

        pos = torch.arange(
            S, device=self.device
        )  # [S], position indices for each token in the sequence

        # Find the position of the last image token in each sequence (or S if no image token is present)
        has_img = is_img.any(dim=1)  # [B]
        last_img = torch.where(
            has_img,
            (is_img * pos).max(dim=1).values,
            torch.full(
                (self.B,), S, device=self.device, dtype=pos.dtype
            ),  # If no image token, set to S (out of bounds)
        )  # [B]

        # Non-image tokens after the last image token are considered generated
        pos = pos[None, :]
        is_gen = ~is_pad & ~is_img & (pos > last_img[:, None])  # [B, S]

        pos = pos.expand(self.B, S)  # [B, S]
        img_pos = pos.masked_fill(~is_img, -1)  # [B, S]
        gen_pos = pos.masked_fill(~is_gen, -1)  # [B, S]

        # Minimize padding by compacting indices to the left and keeping track of valid lengths with masks
        img_lists = [row[row != -1] for row in img_pos]
        gen_lists = [row[row != -1] for row in gen_pos]

        self.T_img = max((len(lst) for lst in img_lists), default=0)
        self.T_gen = max((len(lst) for lst in gen_lists), default=0)

        # Create padded index tensors and masks for image and generated tokens
        self.img_token_idx = torch.full(
            (self.B, self.T_img), -1, dtype=torch.int32, device=self.device
        )
        self.gen_token_idx = torch.full(
            (self.B, self.T_gen), -1, dtype=torch.int32, device=self.device
        )
        self.img_mask = torch.zeros((self.B, self.T_img), dtype=torch.bool, device=self.device)
        self.gen_mask = torch.zeros((self.B, self.T_gen), dtype=torch.bool, device=self.device)

        # Fill the index tensors and masks based on the compacted lists of image and generated token positions
        for i, (img_row, gen_row) in enumerate(zip(img_lists, gen_lists, strict=False)):
            if (n := img_row.numel()) > 0:
                self.img_token_idx[i, :n] = img_row
                self.img_mask[i, :n] = True
            if (n := gen_row.numel()) > 0:
                self.gen_token_idx[i, :n] = gen_row
                self.gen_mask[i, :n] = True
