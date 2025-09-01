import torch

from vl_saliency.core.utils import (
    ALL_LAYERS,
    _get_image_token_id,
    _get_vision_patch_shape,
    _select_layers,
)


# Small helper to mimic HF config behavior: supports `"key" in config` and attribute access.
class ConfigLike:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return hasattr(self, key)


# ---- _get_image_token_id ----


def test_get_image_token_id_prefers_image_token_id():
    cfg = ConfigLike(image_token_id=123, image_token_index=7)
    assert _get_image_token_id(cfg) == 123


def test_get_image_token_id_falls_back_to_image_token_index():
    cfg = ConfigLike(image_token_index=7)
    assert _get_image_token_id(cfg) == 7


def test_get_image_token_id_default_minus_one():
    cfg = ConfigLike()  # neither attribute present
    assert _get_image_token_id(cfg) == -1


# ---- _get_vision_patch_shape ----


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


# ---- _select_layers ----


def test_select_layers_all_layers_identity():
    t = torch.arange(24).view(6, 4)  # 6 "layers"
    out = _select_layers(t, ALL_LAYERS)
    # same object or at least same content/shape
    assert out.data_ptr() == t.data_ptr() or torch.equal(out, t)


def test_select_layers_positive_int_first_n():
    t = torch.arange(20).view(5, 4)  # 5 layers
    out = _select_layers(t, 2)
    assert out.shape[0] == 2
    assert torch.equal(out, t[:2])


def test_select_layers_negative_int_last_n():
    t = torch.arange(20).view(5, 4)
    out = _select_layers(t, -2)
    assert out.shape[0] == 2
    assert torch.equal(out, t[-2:])


def test_select_layers_with_explicit_indices_sequence():
    t = torch.arange(30).view(6, 5)
    out = _select_layers(t, [0, 2, 5])
    assert out.shape[0] == 3
    assert (
        torch.equal(out[0], t[0])
        and torch.equal(out[1], t[2])
        and torch.equal(out[2], t[5])
    )
