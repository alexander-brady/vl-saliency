from typing import Any

import pytest

import vl_saliency.api.config as m
from vl_saliency.api.config import SaliencyConfig
from vl_saliency.utils.patch_fns import FixedPatchLayout


@pytest.fixture(autouse=True)
def infer_mocks(monkeypatch):
    monkeypatch.setattr(m, "infer_image_token_id", lambda config: 42)
    monkeypatch.setattr(m, "infer_pad_token_id", lambda config: 99)
    monkeypatch.setattr(m, "infer_image_patch_fn", lambda config: FixedPatchLayout(8, 8))
    monkeypatch.setattr(m, "infer_attn_scale", lambda config: 0.5)


def test_config_from_model_infer(build_config, build_model, monkeypatch):
    base = dict[str, Any](
        layer_reduce="mean",
        layer_op=None,
        head_reduce="mean",
        head_op=None,
        backend="auto",
    )

    model = build_model()
    config = SaliencyConfig.from_model(model, **base)
    expected = build_config(
        image_token_id=42,
        pad_token_id=99,
        image_patch_fn=FixedPatchLayout(8, 8),
        attn_scale=0.5,
        **base,
    )

    for field in SaliencyConfig.__dataclass_fields__:
        if field != "image_patch_fn":  # Skip complex field for direct equality check
            assert getattr(config, field) == getattr(expected, field)

    assert type(config.image_patch_fn) is type(expected.image_patch_fn)
    assert config.image_patch_fn.patch_shape == expected.image_patch_fn.patch_shape


def test_config_resolve_patch_fn(build_config, build_model):
    model = build_model()

    # Test that a callable image_patch_fn is used directly
    patch_fn = FixedPatchLayout(16, 16)
    config = SaliencyConfig.from_model(model, image_patch_fn=patch_fn)
    assert config.image_patch_fn == patch_fn, (
        "Expected image_patch_fn to be used directly when callable."
    )

    # Test that a tuple image_patch_fn is converted to FixedPatchLayout
    config = SaliencyConfig.from_model(model, image_patch_fn=(32, 32))
    assert isinstance(config.image_patch_fn, FixedPatchLayout), (
        "Expected image_patch_fn to be converted to FixedPatchLayout when given as a tuple."
    )
