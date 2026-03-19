import math

from transformers import PreTrainedConfig, PretrainedConfig

from vl_saliency._types import PatchLayoutFn
from vl_saliency.config.patch_fns import StaticPatchLayout, thw_patch_layout


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


def infer_pad_token_id(config: PretrainedConfig) -> int:
    """Infers pad token id from model config if possible. Raises ValueError if inference fails."""
    if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        return config.pad_token_id
    else:
        raise ValueError(
            "Could not infer pad token id from model config. Please specify it explicitly or ensure your model config has a `pad_token_id` attribute."
        )


def infer_patch_layout_fn(config: PretrainedConfig) -> PatchLayoutFn:
    """Get the image patch function from a multimodal config. Raises ValueError if inference fails."""
    if "mm_tokens_per_image" in config:
        side = int(config.mm_tokens_per_image**0.5)
        return StaticPatchLayout(side, side)  # Assume Square Tokens

    # Otherwise, check vision_config
    if "vision_config" in config:
        vision_cfg = config.vision_config
        if "image_size" in vision_cfg and "patch_size" in vision_cfg:
            side = vision_cfg.image_size // vision_cfg.patch_size
            return StaticPatchLayout(side, side)  # Assume Square Tokens

    match config.model_type:
        case s if s.startswith("qwen"):
            # For Qwen models, we can infer patch shape from the input
            # images at runtime, since they use dynamic patching.
            return thw_patch_layout

    raise ValueError(
        "Could not infer image patch shape from model config. "
        "Please provide a value for `image_patch_shape` or ensure your model config contains either `mm_tokens_per_image` or `vision_config` with `image_size` and `patch_size`."
    )


def infer_attn_scale(config: PreTrainedConfig) -> float:
    """Infers attention scaling factor from model config. Raises ValueError if inference fails."""
    if hasattr(config, "text_config"):
        config = config.text_config
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim
    elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        head_dim = config.hidden_size // config.num_attention_heads
    else:
        raise ValueError(
            "Could not infer attention scaling factor from model config. Please specify it explicitly or ensure your model config has `hidden_size` and `num_attention_heads` attributes."
        )

    return 1.0 / math.sqrt(head_dim)
