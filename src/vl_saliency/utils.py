from typing import Optional, Tuple, Sequence
import torch


ALL_LAYERS: object = object() 

def _get_image_token_id(config) -> int:
    """
    Get the image token id from a multimodal config. If not found, return -1.
    """
    return getattr(config, "image_token_id", getattr(config, "image_token_index", -1))


def _get_vision_patch_shape(config) -> Optional[Tuple[int, int]]:
    """
    Get the number of height and width tokens from a multimodal config.
    """
    # If explicit count is given, prefer that
    if "mm_tokens_per_image" in config: 
        side = int(config.mm_tokens_per_image ** 0.5)
        return side, side # Assume Square Tokens

    # Otherwise, check vision_config
    if "vision_config" in config:
        vision_cfg = config.vision_config
        if "image_size" in vision_cfg and "patch_size" in vision_cfg:
            image_size = vision_cfg.image_size
            patch_size = vision_cfg.patch_size
            side = image_size // patch_size
            return side, side  # Assume Square Tokens

    return None


def _select_layers(tensors: torch.Tensor, indices: int | object | Sequence[int]) -> torch.Tensor:
    """
    Extract specific layers from a tensor.
        - None: use all layers
        - int > 0: use the first `extracted_layers` layers
        - int < 0: use the last `abs(extracted_layers)` layers
        - Sequence[int]: use specific layer indices
    """
    if indices is ALL_LAYERS:
        return tensors
    if isinstance(indices, int):
        return tensors[:indices] if indices > 0 else tensors[indices:]
    return tensors[indices]