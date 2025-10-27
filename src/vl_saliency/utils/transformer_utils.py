"""Helper functions for dealing with HF Transformers models."""

import torch


def _get_image_token_id(config) -> int:
    """
    Get the image token id from a multimodal config. If not found, return -1.
    """
    return getattr(config, "image_token_id", getattr(config, "image_token_index", -1))


def _get_vision_patch_shape(config) -> tuple[int, int] | None:
    """
    Get the number of height and width tokens from a multimodal config.
    """
    # If explicit count is given, prefer that
    if "mm_tokens_per_image" in config:
        side = int(config.mm_tokens_per_image**0.5)
        return side, side  # Assume Square Tokens

    # Otherwise, check vision_config
    if "vision_config" in config:
        vision_cfg = config.vision_config
        if "image_size" in vision_cfg and "patch_size" in vision_cfg:
            image_size = vision_cfg.image_size
            patch_size = vision_cfg.patch_size
            side = image_size // patch_size
            return side, side  # Assume Square Tokens

    return None


def _image_patch_shapes(
    image_count: int, patch_shape: tuple[int, int] | None = None, image_grid_thw: torch.Tensor | None = None
) -> list[tuple[int, int]]:
    """
    Get the number of height and width patches for the input images.
    If `self.patch_shape` is set, return that directly.
    Otherwise, infer from the input images or provided image grid sizes.
    """
    if image_grid_thw is not None:
        patches = (image_grid_thw[:, 1:] // 2).tolist()  # Each row -> [H, W]
        if len(patches) != image_count:
            raise ValueError(
                f"Number of image grid sizes ({len(patches)}) does not match number of images ({image_count})."
            )
        patches = [tuple(patch) for patch in patches]

    elif patch_shape is not None:
        patches = [patch_shape] * image_count

    else:
        raise ValueError(
            "Cannot infer image patch shapes. Please provide `image_grid_thw` or set a static `patch_shape` in the model config."
        )

    return patches
