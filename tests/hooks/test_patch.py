from types import MethodType

import pytest

import vl_saliency.hooks.patch as patch_module
from vl_saliency.hooks.patch import (
    _PATCH_ATTR_NAME,
    apply_forward_patch,
    is_patched,
    restore_forward,
)

# ------- Fixtures and helper classes for testing -------


@pytest.fixture
def patchable_model(build_model, build_model_config):
    text_config = build_model_config(_attn_implementation="original")
    model = build_model(text_config=text_config)

    # Minimal attention setter stub
    model.set_attn_implementation = MethodType(
        lambda self, config: setattr(
            self.config.text_config, "_attn_implementation", config["text_config"]
        ),
        model,
    )

    # Deterministic original forward
    model.forward = MethodType(
        lambda self, *args, **kwargs: "original_forward",
        model,
    )

    return model


@pytest.fixture
def mock_saliency_forward(monkeypatch):
    """
    Installs a fake build_saliency_forward that returns
    a forward method yielding the provided return_value.
    """

    def _install(return_value: str):
        def fake_builder(config, attn_implementation, forward):
            def patched(self, *args, **kwargs):
                return return_value

            return patched

        monkeypatch.setattr(
            patch_module,
            "build_saliency_forward",
            fake_builder,
        )

    return _install


# ------- is_patched tests -------


def test_is_patched_flag(patchable_model):
    model = patchable_model

    assert not is_patched(model)

    setattr(model, _PATCH_ATTR_NAME, object())
    assert is_patched(model)

    delattr(model, _PATCH_ATTR_NAME)
    assert not is_patched(model)


# ------- apply_forward_patch tests -------


def test_apply_forward_patch_replaces_forward(
    patchable_model,
    build_config,
    mock_saliency_forward,
):
    mock_saliency_forward("patched_forward")

    model = patchable_model
    config = build_config()

    # Sanity check
    assert model.forward() == "original_forward"

    apply_forward_patch(model, config)

    assert is_patched(model)
    assert model.config.text_config._attn_implementation == "saliency"
    assert model.forward() == "patched_forward"


def test_apply_forward_patch_is_idempotent(
    patchable_model,
    build_config,
    mock_saliency_forward,
):
    mock_saliency_forward("patched_forward")

    model = patchable_model
    config = build_config()

    apply_forward_patch(model, config)

    patched_forward = model.forward
    patched_attn_impl = model.config.text_config._attn_implementation

    assert model.forward() == "patched_forward"

    # Second application should not change anything
    apply_forward_patch(model, build_config())

    assert is_patched(model)
    assert model.forward is patched_forward
    assert model.config.text_config._attn_implementation == patched_attn_impl
    assert model.forward() == "patched_forward"


# ------- restore_forward tests -------


def test_restore_forward_restores_original_state(
    patchable_model,
    build_config,
    mock_saliency_forward,
):
    mock_saliency_forward("patched_forward")

    model = patchable_model
    config = build_config()

    original_forward = model.forward

    apply_forward_patch(model, config)
    assert is_patched(model)
    assert model.forward() == "patched_forward"

    restore_forward(model)

    assert not is_patched(model)
    assert model.forward == original_forward
    assert model.config.text_config._attn_implementation == "original"
    assert model.forward() == "original_forward"


def test_restore_forward_is_noop_when_not_patched(patchable_model):
    model = patchable_model

    restore_forward(model)

    assert not is_patched(model)
    assert model.forward() == "original_forward"
