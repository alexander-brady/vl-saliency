import pytest
import torch

import vl_saliency.core.accum.base as m
from vl_saliency.core.accum.base import SaliencyAccumulator

# ------- Helper -------


def simple_input():
    # One image token (id=1) at pos 0, one generated token at pos 2
    # B=1, S=3
    input_ids = torch.tensor([[1, 2, 3]])
    pixel_values = torch.randn((1, 3, 4, 4))
    return input_ids, pixel_values


# ------- Behavior -------


@pytest.mark.parametrize(
    ["reduction", "expected"],
    [
        ("mean", 0.0),
        ("sum", 0.0),
        ("max", float("-inf")),
        ("min", float("inf")),
        ("prod", 1.0),
    ],
)
def test_accum_init_saliency_reductions(monkeypatch, reduction, expected, build_config):
    input_ids, pixel_values = simple_input()

    # stub backend resolution
    monkeypatch.setattr(m, "assign_auto", lambda *args, **kwargs: "dummy")
    monkeypatch.setattr(m, "get_qk_accumulator", lambda **kwargs: lambda **kw: kw["saliency"])

    config = build_config(layer_reduce=reduction)

    trace = SaliencyAccumulator(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    tensor = trace._saliency
    assert tensor.shape == (1, trace.layout.T_gen, trace.layout.T_img)

    if tensor.numel() > 0:
        assert torch.all(tensor == expected)


@pytest.mark.parametrize(
    ["layer_reduce", "head_reduce", "expect_zero", "expected_shape"],
    [
        ("stack", "mean", False, (1, 0)),
        ("stack", "stack", False, (1, 0, 1)),
        ("mean", "stack", True, (1, 1)),
    ],
)
def test_accum_init_saliency_stack(
    monkeypatch, build_config, layer_reduce, head_reduce, expect_zero, expected_shape
):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr(m, "assign_auto", lambda *_, **__: "dummy")
    monkeypatch.setattr(
        m,
        "get_qk_accumulator",
        lambda **_: lambda **kw: kw["saliency"],
    )

    config = build_config(layer_reduce=layer_reduce, head_reduce=head_reduce)

    trace = SaliencyAccumulator(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    tensor = trace._saliency
    expected_shape = (*expected_shape, trace.layout.T_gen, trace.layout.T_img)

    assert tensor.shape == expected_shape

    if expect_zero:
        assert torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        assert tensor.numel() == 0


@pytest.mark.parametrize(["reduction", "expected"], [("mean", 1.0), ("sum", 2.0)])
def test_accum_accumulate(monkeypatch, build_config, reduction, expected):

    monkeypatch.setattr(m, "assign_auto", lambda *args, **kwargs: "dummy")
    monkeypatch.setattr(
        m, "get_qk_accumulator", lambda **kwargs: lambda *a, **kw: kw["saliency"] + 1
    )

    config = build_config(layer_reduce=reduction)
    input_ids, pixel_values = simple_input()
    trace = SaliencyAccumulator(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert trace.layers_accumulated == 0
    assert torch.all(trace._saliency == 0)

    q = torch.randn((1, 2, 3, 4))
    k = torch.randn((1, 2, 3, 4))

    trace.accumulate_qk(q, k)
    assert trace.layers_accumulated == 1
    assert torch.all(trace._saliency == 1)

    trace.accumulate_qk(q, k)
    assert trace.layers_accumulated == 2
    assert torch.all(trace._saliency == 2)

    assert trace.saliency._tensor.equal(torch.full_like(trace._saliency, expected))


# -------Backend Resolution -----


def test_accum_resolve_auto(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    called_with = {}

    def mock_assign_auto(device, head_reduce):
        called_with["device"] = device
        called_with["head_reduce"] = head_reduce
        return "dummy"

    monkeypatch.setattr(m, "assign_auto", mock_assign_auto)
    monkeypatch.setattr(m, "get_qk_accumulator", lambda **kwargs: called_with.update(kwargs))

    config = build_config(layer_reduce="mean")

    SaliencyAccumulator(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert called_with["device"] == input_ids.device
    assert called_with["head_reduce"] == "mean"
    assert called_with["backend"] == "dummy"


def test_accum_resolve_backend(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    called_with = {}

    monkeypatch.setattr(m, "get_qk_accumulator", lambda **kwargs: called_with.update(kwargs))

    config = build_config(layer_reduce="mean", backend="triton")

    SaliencyAccumulator(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert called_with["backend"] == "triton"
