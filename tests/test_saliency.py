from typing import Any

import pytest

import vl_saliency.context as m
from vl_saliency.context import Saliency


@pytest.fixture
def dummy_saliency(build_model, build_config):
    model = build_model()
    config = build_config()
    return Saliency(model, config)


def test_saliency_config_init(build_model, build_config):
    model = build_model()
    config = build_config()
    sal = Saliency(model, config)

    assert sal.model is model
    assert sal.config == config


def test_saliency_kwarg_init(build_model, monkeypatch):
    monkeypatch.setattr(m.SaliencyConfig, "from_model", lambda *args, **kwargs: kwargs)

    kwargs = dict[str, Any](
        image_token_id=42,
        pad_token_id=0,
        patch_layout_fn=(16, 16),
        attn_scale=0.5,
        head_op=None,
        head_reduce="mean",
        layer_op=None,
        layer_reduce="sum",
        selection=None,
        backend="auto",
    )

    model = build_model()
    sal = Saliency(model, **kwargs)

    assert sal.config == kwargs
    assert sal.config == m.SaliencyConfig.from_model(model, **kwargs)


def test_saliency_context_manager(dummy_saliency, monkeypatch):
    enabled = set()
    monkeypatch.setattr(m.Saliency, "enable", lambda self: enabled.update({True}))
    monkeypatch.setattr(m.Saliency, "disable", lambda self: enabled.clear())

    sal = dummy_saliency
    with sal as s:
        assert s is sal
        assert enabled == {True}
    assert enabled == set()


def test_saliency_enable_patches_forward(dummy_saliency, monkeypatch, caplog):
    forward_kwargs = {"times_called": 0}

    def fake_apply_forward_patch(model, config):
        forward_kwargs["model"] = model
        forward_kwargs["config"] = config
        forward_kwargs["times_called"] += 1

    monkeypatch.setattr(m, "apply_forward_patch", fake_apply_forward_patch)
    monkeypatch.setattr(m, "is_patched", lambda model: forward_kwargs["times_called"] > 0)

    sal = dummy_saliency

    sal.enable()
    assert forward_kwargs["model"] is dummy_saliency.model
    assert forward_kwargs["config"] == dummy_saliency.config

    with caplog.at_level("WARNING"):
        sal.enable()
        assert "already enabled" in caplog.text
        assert forward_kwargs["times_called"] == 1  # Still only called once, second should be no-op


def test_saliency_disable_restores_forward(dummy_saliency, monkeypatch, caplog):
    restore_called = {"times_called": 1}

    def fake_restore_forward(model):
        restore_called["model"] = model
        restore_called["times_called"] -= 1

    monkeypatch.setattr(m, "restore_forward", fake_restore_forward)
    monkeypatch.setattr(m, "is_patched", lambda model: restore_called["times_called"] > 0)

    sal = dummy_saliency
    sal.disable()

    assert restore_called["model"] is dummy_saliency.model
    assert restore_called["times_called"] == 0

    with caplog.at_level("WARNING"):
        sal.disable()
        assert "already disabled" in caplog.text
        assert restore_called["times_called"] == 0  # Still only called once, second should be no-op
