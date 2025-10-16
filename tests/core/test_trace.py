from contextlib import nullcontext

import pytest
import torch

from vl_saliency import SaliencyTrace


def test_capture_raises_on_batch_size_not_one(dummy_model, dummy_processor):
    trace = SaliencyTrace(dummy_model, dummy_processor, method="dummy")
    trace.image_token_id = 99
    # generated_ids has batch=2 -> should raise
    with pytest.raises(ValueError):
        trace.capture(
            generated_ids=torch.ones(2, 5, dtype=torch.long),
            input_ids=torch.ones(2, 5, dtype=torch.long),
            pixel_values=torch.randn(1, 3, 8, 8),
        )


@pytest.mark.parametrize(
    "have_grid,patch_shape,expect_ok",
    [
        (True, None, True),  # grid provided
        (False, None, False),  # neither -> error
        (False, (5, 1), True),  # only patch_shape -> ok
        (False, (2, 2), False),  # wrong patch_shape -> error (when true size is 4x4)
    ],
)
def test_capture_patch_shape(dummy_model, dummy_processor, have_grid, patch_shape, expect_ok):
    trace = SaliencyTrace(dummy_model, dummy_processor, method="dummy")
    trace.image_token_id = 1

    generated_ids = torch.ones(1, 5, dtype=torch.long)
    input_ids = torch.ones(1, 5, dtype=torch.long)
    pixel_values = torch.randn(1, 3, 8, 8)

    kwargs = {}
    if have_grid:
        kwargs["image_grid_thw"] = torch.tensor([[1, 10, 2]], dtype=torch.int32)
    if patch_shape is not None:
        trace.patch_shape = patch_shape

    ctx = pytest.raises(ValueError) if not expect_ok else nullcontext()
    with ctx:
        trace.capture(
            generated_ids=generated_ids,
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )


def test_map_raises_without_capture(dummy_model, dummy_processor):
    trace = SaliencyTrace(dummy_model, dummy_processor, method="dummy")
    trace.image_token_id = 99
    with pytest.raises(ValueError):
        trace.map(token=0)


@pytest.mark.parametrize("H,W", [(2, 3)])
def test_capture_then_map_produces_mask(tiny_model, io_factory, make_trace, H, W):
    trace = make_trace(tiny_model, method=lambda a, g: a * g)
    io = io_factory(H=H, W=W)
    trace.capture(**io.__dict__)
    sal = trace.map(token=io.generated_ids.shape[1] - 1)
    assert sal.shape == (1, 1, H, W)
    assert torch.isfinite(sal).all()


@pytest.mark.parametrize("H,W", [(2, 3)])
def test_calls_render_token_ids(monkeypatch, tiny_model, io_factory, make_trace, H, W, dummy_processor):
    called = {}

    def fake_render_token_ids(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("vl_saliency.viz.tokens.render_token_ids", fake_render_token_ids)

    trace = make_trace(tiny_model, method=lambda a, g: a * g)
    io = io_factory(H=H, W=W)
    trace.capture(**io.__dict__, visualize_tokens=True)

    assert torch.equal(called["generated_ids"], io.generated_ids)
    assert called["processor"] == trace.processor


@pytest.mark.parametrize("H,W", [(2, 3)])
def test_capture_maps_are_equiv(tiny_model, io_factory, make_trace, H, W):
    trace = make_trace(tiny_model, method="dummy")
    io = io_factory(H=H, W=W)
    trace.capture(**io.__dict__)

    token = io.generated_ids.shape[1] - 1
    outs = [
        trace.map(token=token, method=None),
        trace.map(token=token, method="dummy"),
        trace.map(token=token, method=lambda a, g: a * g),
    ]
    base = outs[0]
    for o in outs[1:]:
        assert torch.allclose(base, o)
