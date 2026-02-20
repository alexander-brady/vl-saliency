from transformers import PretrainedConfig

from vl_saliency._types import ImagePatchFunction

from .patch_fns import StaticPatches, image_thw_to_patches


def infer_image_token_id(config: PretrainedConfig) -> int:
    """Infers image token id from model config if possible. Raises ValueError if inference fails."""
    # Check for common config attributes that specify image token id
    if hasattr(config, "image_token_id"):
        return config.image_token_id
    elif hasattr(config, "image_token_index"):
        return config.image_token_index
    else:
        raise ValueError(
            "Could not infer image token id from model config. Please specify it explicitly."
        )


def infer_image_patch_fn(config: PretrainedConfig) -> ImagePatchFunction:
    """Get the image patch function from a multimodal config. Raises ValueError if inference fails."""
    if "mm_tokens_per_image" in config:
        side = int(config.mm_tokens_per_image**0.5)
        return StaticPatches(side, side)  # Assume Square Tokens

    # Otherwise, check vision_config
    if "vision_config" in config:
        vision_cfg = config.vision_config
        if "image_size" in vision_cfg and "patch_size" in vision_cfg:
            side = vision_cfg.image_size // vision_cfg.patch_size
            return StaticPatches(side, side)  # Assume Square Tokens

    match config.model_type:
        case s if s.startswith("qwen"):
            # For Qwen models, we can infer patch shape from the input
            # images at runtime, since they use dynamic patching.
            return image_thw_to_patches

    raise ValueError(
        "Could not infer image patch shape from model config. "
        "Please provide a value for `image_patch_shape` or ensure your model config contains either `mm_tokens_per_image` or `vision_config` with `image_size` and `patch_size`."
    )
