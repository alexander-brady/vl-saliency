import math

import pytest
import torch

from vl_saliency.extractor import SaliencyExtractor

# ------ Dummy implementations for testing ------


class DummyTextConfig:
    def __init__(self):
        self._attn_implementation = "eager"


class DummyConfig:
    def __init__(self):
        self.text_config = DummyTextConfig()
        self.pad_token_id = 0
        self.hidden_size = 64
        self.num_attention_heads = 8


class DummyModel:
    def __init__(self):
        self.config = DummyConfig()
        self.set_calls = []

    def set_attn_implementation(self, impl_dict):
        self.set_calls.append(impl_dict)
        self.config.text_config._attn_implementation = impl_dict["text_config"]


# ------ Fixtures ------
@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def input_ids():
    return torch.randint(0, 100, (2, 10))


@pytest.fixture
def pixel_values():
    return torch.randn(2, 3, 32, 32)


@pytest.fixture(autouse=True)
def monkeypath_infer_helpers(monkeypatch):
    monkeypatch.setattr(
        "vl_saliency.extractor.infer_image_token_id",
        lambda config: 42,
    )
    monkeypatch.setattr(
        "vl_saliency.extractor.infer_image_patch_fn",
        lambda config: lambda **_: [[]],
    )


@pytest.fixture
def context_as_dict(monkeypatch):
    captured = {}

    class DummyContext:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "vl_saliency.extractor.SaliencyContext",
        DummyContext,
    )

    return captured


# ------ Test case for extractor ------


def test_bind_sets_saliency_attn(model):
    extractor = SaliencyExtractor(model, bind=True)

    assert model.set_calls[0] == {"text_config": "saliency"}
    assert extractor._prev_attn_implementation == "eager"


def test_unbind_restores_previous_attn(model):
    extractor = SaliencyExtractor(model, bind=True)
    extractor.unbind()

    assert model.set_calls[-1] == {"text_config": "eager"}
    assert extractor._prev_attn_implementation is None


def test_unbind_without_bind_logs_warning(model, caplog):
    extractor = SaliencyExtractor(model, bind=False)
    extractor.unbind()

    assert "Skipping unbind" in caplog.text


def test_scale_computation(model):
    extractor = SaliencyExtractor(model)

    expected = 1.0 / math.sqrt(64 // 8)
    assert extractor.scale == expected


def test_static_patch_tuple_is_wrapped(model, monkeypatch):
    called = {}

    def fake_static(h, w):
        called["args"] = (h, w)
        return lambda **_: [[(h, w)]]

    monkeypatch.setattr(
        "vl_saliency.extractor.StaticPatches",
        fake_static,
    )

    _ = SaliencyExtractor(model, image_patch_fn=(4, 4))

    assert called["args"] == (4, 4)


def test_callable_patch_fn_used(model, input_ids, pixel_values, context_as_dict):
    def patch_fn(batch_size, image_count, **kwargs):
        return [[(1, 1)] for _ in range(image_count)]

    extractor = SaliencyExtractor(model, image_patch_fn=patch_fn)

    _ = extractor(input_ids=input_ids, pixel_values=pixel_values)

    assert context_as_dict["patch_shapes"] == [[(1, 1)], [(1, 1)]]
    assert context_as_dict["scale"] == extractor.scale


def test_image_counts_neq_batch_size_raises(model):
    def patch_fn(batch_size, image_count, **kwargs):
        return [[(1, 1)] for _ in range(image_count)]

    input_ids = torch.randint(0, 100, (2, 10))
    pixel_values = torch.randn(3, 3, 32, 32)  # 3 images but batch size is 2

    extractor = SaliencyExtractor(model, image_patch_fn=patch_fn)

    with pytest.raises(ValueError, match="one image per input"):
        extractor(input_ids=input_ids, pixel_values=pixel_values)


def test_no_pixel_values_gives_empty_patch_shapes(model, input_ids, context_as_dict):
    extractor = SaliencyExtractor(model)

    _ = extractor(input_ids=input_ids, pixel_values=None)

    assert context_as_dict["patch_shapes"] == [[] for _ in range(input_ids.shape[0])]


def test_reduction_overrides(model, input_ids, context_as_dict):
    extractor = SaliencyExtractor(model, layer_reduce="mean", head_reduce="mean")

    _ = extractor(
        input_ids=input_ids,
        layer_reduce="max",
        head_reduce="sum",
    )

    assert context_as_dict["layer_reduce"] == "max"
    assert context_as_dict["head_reduce"] == "sum"


def test_bad_patch_fn_raises(model):
    def bad_patch_fn(batch_size, image_count, **kwargs):
        return [[(1, 1)] for _ in range(batch_size + 1)]  # Return one more than needed

    extractor = SaliencyExtractor(model, image_patch_fn=bad_patch_fn)

    with pytest.raises(ValueError, match="patch shapes"):
        extractor(input_ids=torch.randint(0, 100, (1, 10)), pixel_values=torch.randn(1, 3, 32, 32))


def test_ops_overrides(model, input_ids, context_as_dict):
    extractor = SaliencyExtractor(model, layer_reduce="mean", head_reduce="mean")

    def head_op(scores: torch.Tensor, mask: torch.Tensor):
        return scores * 2

    def layer_op(scores: torch.Tensor, mask: torch.Tensor):
        return scores + 1

    _ = extractor(
        input_ids=input_ids,
        layer_op=layer_op,
        head_op=head_op,
    )

    assert context_as_dict["layer_op"] == layer_op
    assert context_as_dict["head_op"] == head_op
