from typing import Any

import pytest

import vl_saliency.config.config as m
from vl_saliency.config import SaliencyConfig, StaticPatchLayout


@pytest.fixture(autouse=True)
def infer_mocks(monkeypatch):
    monkeypatch.setattr(m, "infer_attn_scale", lambda config: 0.5)
    monkeypatch.setattr(m, "infer_image_token_id", lambda config: 42)
    monkeypatch.setattr(m, "infer_pad_token_id", lambda config: 99)
    monkeypatch.setattr(m, "infer_patch_layout_fn", lambda config: StaticPatchLayout(8, 8))


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
        patch_layout_fn=StaticPatchLayout(8, 8),
        attn_scale=0.5,
        **base,
    )

    for field in SaliencyConfig.__dataclass_fields__:
        if field != "patch_layout_fn":  # Skip complex field for direct equality check
            assert getattr(config, field) == getattr(expected, field)

    assert type(config.patch_layout_fn) is type(expected.patch_layout_fn)
    assert config.patch_layout_fn.patch_shape == expected.patch_layout_fn.patch_shape  # type: ignore


def test_config_resolve_patch_fn(build_model):
    model = build_model()

    # Test that a callable patch_layout_fn is used directly
    patch_fn = StaticPatchLayout(16, 16)
    config = SaliencyConfig.from_model(model, patch_layout_fn=patch_fn)
    assert config.patch_layout_fn == patch_fn

    # Test that a tuple patch_layout_fn is converted to StaticPatchLayout
    config = SaliencyConfig.from_model(model, patch_layout_fn=(32, 32))
    assert isinstance(config.patch_layout_fn, StaticPatchLayout)
