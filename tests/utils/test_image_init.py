import pytest
from transformers import PreTrainedConfig

from vl_saliency.utils.image_init import infer_image_patch_fn, infer_image_token_id
from vl_saliency.utils.patch_fns import image_thw_to_patches

# ------ Dummy implementations for testing ------


class DummyConfig(PreTrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyStaticPatches:
    def __init__(self, width, height):
        self.patch_shape = (width, height)


@pytest.fixture
def patch_static_patches(monkeypatch):
    monkeypatch.setattr("vl_saliency.utils.image_init.StaticPatches", DummyStaticPatches)


# ------ Test cases for image initialization utilities ------


def test_infer_image_token_id():
    config = DummyConfig(image_token_id=999)
    assert infer_image_token_id(config) == 999

    config = DummyConfig(image_token_index=888)
    assert infer_image_token_id(config) == 888

    config = DummyConfig()
    with pytest.raises(ValueError, match="Could not infer"):
        infer_image_token_id(config)


def test_infer_image_patch_fn_mm_tokens_per_image(patch_static_patches):
    config = DummyConfig(mm_tokens_per_image=16)
    patch_fn = infer_image_patch_fn(config)
    assert isinstance(patch_fn, DummyStaticPatches)
    assert patch_fn.patch_shape == (4, 4)


def test_infer_image_patch_fn_vision_config(patch_static_patches):
    vision_config = DummyConfig(image_size=224, patch_size=16)
    config = DummyConfig(vision_config=vision_config)
    patch_fn = infer_image_patch_fn(config)
    assert isinstance(patch_fn, DummyStaticPatches)
    assert patch_fn.patch_shape == (14, 14)


def test_infer_image_patch_fn_qwen():
    config = DummyConfig(model_type="qwen-2b")
    patch_fn = infer_image_patch_fn(config)
    assert patch_fn == image_thw_to_patches


def test_infer_image_patch_fn_failure():
    config = DummyConfig(model_type="unknown")
    with pytest.raises(ValueError, match="Could not infer image patch shape"):
        infer_image_patch_fn(config)
