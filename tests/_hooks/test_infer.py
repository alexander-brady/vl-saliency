import pytest

import vl_saliency._hooks.infer as m
from vl_saliency._hooks.infer import infer_image_token_id, infer_patch_layout_fn
from vl_saliency.config.patch_fns import thw_patch_layout

# ------ Dummy implementations for testing ------


class DummyFixedPatchLayout:
    def __init__(self, width, height):
        self.patch_shape = (width, height)


@pytest.fixture
def patch_fixed_patch_layout(monkeypatch):
    monkeypatch.setattr(m, "StaticPatchLayout", DummyFixedPatchLayout)


# ------ Infer image token id tests ---------


def test_infer_image_token_id(build_model_config):
    config = build_model_config(image_token_id=999)
    assert infer_image_token_id(config) == 999

    config = build_model_config(image_token_index=888)
    assert infer_image_token_id(config) == 888

    config = build_model_config()
    with pytest.raises(ValueError, match="Could not infer"):
        infer_image_token_id(config)


# ------ Infer pad token id tests -------


def test_infer_pad_token_id(build_model_config):
    config = build_model_config(pad_token_id=123)
    assert m.infer_pad_token_id(config) == 123

    config = build_model_config()
    with pytest.raises(ValueError, match="Could not infer"):
        m.infer_pad_token_id(config)


# ------ Infer image patch function tests -------


def test_infer_patch_layout_fn_mm_tokens_per_image(patch_fixed_patch_layout, build_model_config):
    config = build_model_config(mm_tokens_per_image=16)
    patch_fn = infer_patch_layout_fn(config)
    assert isinstance(patch_fn, DummyFixedPatchLayout)
    assert patch_fn.patch_shape == (4, 4)


def test_infer_patch_layout_fn_vision_config(patch_fixed_patch_layout, build_model_config):
    vision_config = build_model_config(image_size=224, patch_size=16)
    config = build_model_config(vision_config=vision_config)
    patch_fn = infer_patch_layout_fn(config)
    assert isinstance(patch_fn, DummyFixedPatchLayout)
    assert patch_fn.patch_shape == (14, 14)


def test_infer_patch_layout_fn_qwen(build_model_config):
    config = build_model_config(model_type="qwen-2b")
    patch_fn = infer_patch_layout_fn(config)
    assert patch_fn == thw_patch_layout


def test_infer_patch_layout_fn_failure(build_model_config):
    config = build_model_config(model_type="unknown")
    with pytest.raises(ValueError, match="Could not infer image patch shape"):
        infer_patch_layout_fn(config)


# ------ Infer attention scale tests -------


def test_infer_attn_scale_head_dim(build_model_config):
    config = build_model_config(head_dim=64)
    assert m.infer_attn_scale(config) == 1 / (64**0.5)


def test_infer_attn_scale_hidden_size(build_model_config):
    config = build_model_config(hidden_size=512, num_attention_heads=8)
    assert m.infer_attn_scale(config) == 1 / ((512 // 8) ** 0.5)


def test_infer_attn_scale_failure(build_model_config):
    config = build_model_config()
    with pytest.raises(ValueError, match="Could not infer attention scaling factor"):
        m.infer_attn_scale(config)
