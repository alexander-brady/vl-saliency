from vl_saliency.utils.transformer_utils import (
    _get_image_token_id,
    _get_vision_patch_shape,
)

# Small helper to mimic HF config behavior: supports `"key" in config` and attribute access.
class ConfigLike:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return hasattr(self, key)

# ------------------------- _get_image_token_id -----------------------------


def test_get_image_token_id_prefers_image_token_id():
    cfg = ConfigLike(image_token_id=123, image_token_index=7)
    assert _get_image_token_id(cfg) == 123


def test_get_image_token_id_falls_back_to_image_token_index():
    cfg = ConfigLike(image_token_index=7)
    assert _get_image_token_id(cfg) == 7


def test_get_image_token_id_default_minus_one():
    cfg = ConfigLike()  # neither attribute present
    assert _get_image_token_id(cfg) == -1


# ------------------------- _get_vision_patch_shape -------------------------


def test_get_vision_patch_shape_from_mm_tokens_per_image():
    # sqrt(49) = 7 -> (7, 7)
    cfg = ConfigLike(mm_tokens_per_image=49)
    assert _get_vision_patch_shape(cfg) == (7, 7)


def test_get_vision_patch_shape_from_vision_config():
    vision_cfg = ConfigLike(image_size=224, patch_size=16)  # 224/16 = 14
    cfg = ConfigLike(vision_config=vision_cfg)
    assert _get_vision_patch_shape(cfg) == (14, 14)


def test_get_vision_patch_shape_none_when_missing():
    cfg = ConfigLike()  # no mm_tokens_per_image, no vision_config
    assert _get_vision_patch_shape(cfg) is None
