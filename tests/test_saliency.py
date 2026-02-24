import math
from dataclasses import dataclass

import pytest
import torch
from transformers import PreTrainedConfig
from transformers.utils.generic import ModelOutput

from vl_saliency.saliency import Saliency
from vl_saliency.utils.image_init import StaticPatches

# ----- Dummmy Model Doubles -----


class DummyTextConfig(PreTrainedConfig):
    def __init__(self):
        self.model_name = "dummy_text_model"
        self._attn_implementation = "default"


class DummyModelConfig:
    def __init__(self):
        self.model_name = "dummy_model"
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.pad_token_id = 0
        self.text_config = DummyTextConfig()
        self.return_dict = True


@dataclass
class DummyOutput(ModelOutput):
    out: str


class DummyModel:
    def __init__(self):
        self.config = DummyModelConfig()

    def forward(self, *args, **kwargs):
        return DummyOutput(out="out")

    def set_attn_implementation(self, impl):
        self.config.text_config._attn_implementation = impl["text_config"]

    @property
    def attn_implementation(self):
        return self.config.text_config._attn_implementation


class StubTrace:
    def __init__(self, config, input_ids, pixel_values, scale):
        self.config = config
        self.input_ids = input_ids
        self.pixel_values = pixel_values
        self.scale = scale
        self.saliency = "generated"


@pytest.fixture
def model() -> DummyModel:
    return DummyModel()


# ----- Builders -----


def make_saliency(model: DummyModel, **overrides) -> Saliency:
    base = dict(
        model=model,
        image_token_id=1,
        image_patch_fn=(16, 16),
        layer_reduce="mean",
        layer_op=None,
        head_reduce="mean",
        head_op=None,
        backend="auto",
    )
    base.update(overrides)
    return Saliency(**base)  # type: ignore


# ----- Config/init tests -----


def test_init_sets_defaults_and_wraps_patch_fn(model):
    sal = make_saliency(model, image_patch_fn=(16, 16))

    assert sal.model is model
    assert sal.config.pad_token_id == model.config.pad_token_id
    assert sal.config.image_token_id == 1

    assert isinstance(sal.config.image_patch_fn, StaticPatches)
    assert sal.config.image_patch_fn.patch_shape == (16, 16)

    assert sal.config.layer_reduce == "mean"
    assert sal.config.layer_op is None
    assert sal.config.head_reduce == "mean"
    assert sal.config.head_op is None


def test_init_reuses_patch_fn_instance(model):
    sal = make_saliency(model, image_patch_fn=(16, 16))
    patch_fn = sal.config.image_patch_fn

    sal2 = make_saliency(model, image_patch_fn=patch_fn)
    assert sal2.config.image_patch_fn is patch_fn


def test_init_infers_image_params_when_missing(monkeypatch, model):
    monkeypatch.setattr("vl_saliency.saliency.infer_image_token_id", lambda config: 42)
    monkeypatch.setattr(
        "vl_saliency.saliency.infer_image_patch_fn",
        lambda config: StaticPatches(32, 32),
    )

    sal = make_saliency(model, image_token_id=None, image_patch_fn=None)

    assert sal.config.image_token_id == 42
    assert isinstance(sal.config.image_patch_fn, StaticPatches)
    assert sal.config.image_patch_fn.patch_shape == (32, 32)


def test_from_config_copies_all_fields(model, build_config):
    cfg = build_config()
    sal = Saliency.from_config(model, cfg)

    for name in cfg.__dataclass_fields__:
        assert getattr(sal.config, name) == getattr(cfg, name)


# ----- Wrap/unwrap tests -----


def test_wrap_replaces_forward_and_sets_attn_impl(model):
    sal = make_saliency(model)

    def dummy_build_forward(config, attn_implementation, forward):
        def wrapped_forward(*args, **kwargs):
            return DummyOutput(out="wrapped")

        return wrapped_forward

    sal._build_saliency_forward = dummy_build_forward  # type: ignore[attr-defined]

    sal.wrap()
    assert sal._prev_forward is not None
    assert sal._prev_attn_impl == "default"
    assert model.attn_implementation == "saliency"

    out = model.forward()
    assert out.out == "wrapped"


def test_unwrap_restores_forward_and_attn_impl(model):
    sal = make_saliency(model)

    def dummy_build_forward(config, attn_implementation, forward):
        def wrapped_forward(*args, **kwargs):
            return DummyOutput(out="wrapped")

        return wrapped_forward

    sal._build_saliency_forward = dummy_build_forward  # type: ignore[attr-defined]

    sal.wrap()
    sal.unwrap()

    assert sal._prev_forward is None
    assert sal._prev_attn_impl is None
    assert model.attn_implementation == "default"

    out = model.forward()
    assert out.out == "out"


def test_context_manager_wraps_and_unwraps(model):
    sal = make_saliency(model)
    sal.wrap = lambda: setattr(sal, "_prev_forward", "wrapped")  # type: ignore[method-assign]
    sal.unwrap = lambda: setattr(sal, "_prev_forward", None)  # type: ignore[method-assign]

    with sal:
        assert sal._prev_forward == "wrapped"
    assert sal._prev_forward is None


def test_double_wrap_is_noop_and_double_unwrap_warns(caplog, model):
    sal = make_saliency(model)

    sal.wrap()
    prev = sal._prev_attn_impl
    sal.wrap()
    assert sal._prev_attn_impl == prev

    sal.unwrap()
    with caplog.at_level("WARNING"):
        sal.unwrap()
        assert "not wrapped" in caplog.text


# ----- Build forward tests -----


def test_build_forward_creates_trace_sets_scale_and_wraps_return(monkeypatch, model, build_config):
    monkeypatch.setattr("vl_saliency.saliency.SaliencyTrace", StubTrace)

    config = build_config()
    input_ids = torch.ones((1, 2), dtype=torch.long)
    expected_scale = 1.0 / math.sqrt(64)  # head_dim = 768 / 12

    def base_forward(model_self, **kwargs):
        assert kwargs["attn_implementation"] == "flash"
        assert isinstance(kwargs["saliency"], StubTrace)
        assert kwargs["saliency"].scale == expected_scale
        return DummyOutput(out="base")

    wrapped = Saliency._build_saliency_forward(
        config=config,
        attn_implementation="flash",
        forward=base_forward,
    )

    result = wrapped(model, input_ids=input_ids)
    assert result.base_output.out == "base"  # type: ignore[attr-defined]
    assert result.saliency == "generated"


def test_build_forward_tuple_return_path(monkeypatch, model, build_config):
    monkeypatch.setattr("vl_saliency.saliency.SaliencyTrace", StubTrace)

    config = build_config()
    input_ids = torch.ones((1, 2), dtype=torch.long)

    def base_forward(model_self, **kwargs):
        return DummyOutput(out="base")

    wrapped = Saliency._build_saliency_forward(
        config=config,
        attn_implementation="flash",
        forward=base_forward,
    )

    result = wrapped(model, input_ids=input_ids, return_dict=False)
    assert result == ("base", "generated")


def test_build_forward_reuses_existing_trace_and_does_not_override_attn_impl(
    monkeypatch, model, build_config
):
    monkeypatch.setattr("vl_saliency.saliency.SaliencyTrace", StubTrace)

    config = build_config()
    input_ids = torch.ones((1, 2), dtype=torch.long)

    class ExistingTrace:
        saliency = "existing"

    def forward_existing(model_self, **kwargs):
        assert kwargs["saliency"].saliency == "existing"
        assert kwargs["attn_implementation"] == "custom"
        return DummyOutput(out="forward_called")

    wrapped = Saliency._build_saliency_forward(
        config=config,
        attn_implementation="flash",
        forward=forward_existing,
    )

    result = wrapped(
        model,
        input_ids=input_ids,
        saliency=ExistingTrace(),
        attn_implementation="custom",
    )
    assert result.saliency == "existing"
