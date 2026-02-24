import pytest
import torch

from vl_saliency.trace import SaliencyTrace  # adjust if module path differs

# ----- Helpers -----


def simple_input():
    # One image token (id=1) at pos 0, one generated token at pos 2
    # B=1, S=3
    input_ids = torch.tensor([[1, 2, 3]])
    pixel_values = torch.randn((1, 3, 4, 4))
    return input_ids, pixel_values


# ----- Initialization tests -----


@pytest.mark.parametrize(
    "reduction,expected",
    [
        ("mean", 0.0),
        ("sum", 0.0),
        ("max", float("-inf")),
        ("min", float("inf")),
        ("prod", 1.0),
    ],
)
def test_init_saliency_reductions(monkeypatch, reduction, expected, build_config):
    input_ids, pixel_values = simple_input()

    # stub backend resolution
    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "dummy")
    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: lambda **kw: kw["saliency"],
    )

    config = build_config(layer_reduce=reduction)

    trace = SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    tensor = trace._saliency
    assert tensor.shape == (1, trace.layout.T_gen, trace.layout.T_img)

    if tensor.numel() > 0:
        assert torch.all(tensor == expected)


def test_init_saliency_invalid_reduction(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "dummy")
    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: lambda **kw: kw["saliency"],
    )

    config = build_config(layer_reduce="invalid")

    with pytest.raises(ValueError, match="Unsupported layer_reduce"):
        SaliencyTrace(
            config=config,
            input_ids=input_ids,
            pixel_values=pixel_values,
        )


# ----- _resolve_qk_fn tests -----


def test_resolve_qk_fn_auto_backend(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "auto_backend")

    captured = {}

    def fake_get_saliency_qk(**kwargs):
        captured.update(kwargs)
        return lambda **kw: kw["saliency"]

    monkeypatch.setattr("vl_saliency.trace.get_saliency_qk", fake_get_saliency_qk)

    config = build_config(backend="auto", head_reduce="sum")

    SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert captured["backend"] == "auto_backend"
    assert captured["head_reduce"] == "sum"


def test_resolve_qk_fn_explicit_backend(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: lambda **kw: kw["saliency"],
    )

    config = build_config(backend="custom")

    trace = SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert trace._saliency_qk_fn is not None


# ------ _patch_shapes tests -----


def test_accumulate_qk_increments_and_updates(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "dummy")

    def fake_qk_fn(q, k, **kwargs):
        # verify scale and masks exist
        assert "gen_idx" in kwargs
        assert "img_idx" in kwargs
        assert "scale" in kwargs
        return kwargs["saliency"] + 1.0

    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: fake_qk_fn,
    )

    config = build_config()

    trace = SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
        scale=2.0,
    )

    q = torch.randn((1, 1, 3, 4))
    k = torch.randn((1, 1, 3, 4))

    trace.accumulate_qk(q, k)
    assert trace.layers_accumulated == 1
    assert torch.all(trace._saliency == 1.0)

    trace.accumulate_qk(q, k)
    assert trace.layers_accumulated == 2
    assert torch.all(trace._saliency == 2.0)


# -------- saliency property tests -----


def test_saliency_property_mean_divides(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "dummy")

    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: lambda q, k, **kw: kw["saliency"] + 2.0,
    )

    config = build_config(layer_reduce="mean")

    trace = SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    q = torch.randn((1, 1, 3, 4))
    k = torch.randn((1, 1, 3, 4))

    trace.accumulate_qk(q, k)  # +2
    trace.accumulate_qk(q, k)  # +2 → total 4

    grid = trace.saliency
    # mean reduction divides by layers_accumulated (2)
    assert torch.all(grid._tensor == 2.0)


def test_saliency_property_no_division_for_sum(monkeypatch, build_config):
    input_ids, pixel_values = simple_input()

    monkeypatch.setattr("vl_saliency.trace.assign_auto", lambda device: "dummy")

    monkeypatch.setattr(
        "vl_saliency.trace.get_saliency_qk",
        lambda **kwargs: lambda q, k, **kw: kw["saliency"] + 3.0,
    )

    config = build_config(layer_reduce="sum")

    trace = SaliencyTrace(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    q = torch.randn((1, 1, 3, 4))
    k = torch.randn((1, 1, 3, 4))

    trace.accumulate_qk(q, k)
    trace.accumulate_qk(q, k)

    grid = trace.saliency
    # sum reduction → no division
    assert torch.all(grid._tensor == 6.0)
