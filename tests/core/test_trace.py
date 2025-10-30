import importlib

import pytest
import torch

from vl_saliency.core.map import SaliencyMap
from vl_saliency.core.trace import Trace
from vl_saliency.selectors.base import Selector

trace_module = importlib.import_module("vl_saliency.core.trace")


def create_trace(proc, attn=True, grad=True) -> Trace:
    return Trace(
        attn=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)] if attn else None,
        grad=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)] if grad else None,
        processor=proc,
        image_token_id=1,
        gen_start=1,
        token_ids=[5, 6, 7, 8],
    )


# ------------------------- Constructor -------------------------


def test_constructor(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    assert trace.attn is not None
    assert trace.grad is not None
    assert trace.processor is dummy_processor
    assert trace.image_token_id == 1
    assert len(trace.attn) == 6
    assert len(trace.grad) == 6
    assert trace.gen_start == 1
    assert trace.token_ids == [5, 6, 7, 8]


def test_set_default_mode(dummy_processor, monkeypatch, caplog):
    monkeypatch.setattr(trace_module.logger, "propagate", True)
    with caplog.at_level("ERROR"):
        trace = create_trace(dummy_processor, attn=True, grad=True)

    assert trace._default == "attn"
    assert len(caplog.messages) == 0

    trace_no_attn = create_trace(dummy_processor, attn=False, grad=True)
    assert trace_no_attn._default == "grad"

    with pytest.raises(ValueError):
        create_trace(dummy_processor, attn=False, grad=False)


def test_invalid_shapes_constructor(dummy_processor):
    # Mismatched layers/heads/gen_tokens
    attn = [torch.randn(2, 2, 3, 6, 6), torch.randn(2, 2, 4, 6, 6)]
    grad = [torch.randn(2, 2, 3, 6, 6), torch.randn(2, 2, 3, 6, 6)]
    with pytest.raises(ValueError):
        Trace(attn=attn, grad=grad, processor=dummy_processor)

    # Mismatched number of images
    attn = [torch.randn(2, 2, 3, 6, 6)]
    grad = [torch.randn(2, 2, 3, 6, 6), torch.randn(2, 2, 3, 6, 6)]
    with pytest.raises(ValueError):
        Trace(attn=attn, grad=grad, processor=dummy_processor)

    # Mismatched tensor shapes
    attn = [torch.randn(2, 2, 3, 6, 6), torch.randn(2, 2, 3, 6, 6)]
    grad = [torch.randn(2, 2, 3, 6, 5), torch.randn(2, 2, 3, 6, 6)]
    with pytest.raises(ValueError):
        Trace(attn=attn, grad=grad, processor=dummy_processor)


def test_invalid_gen_start(monkeypatch, caplog):
    monkeypatch.setattr(trace_module.logger, "propagate", True)
    with caplog.at_level("ERROR"):
        Trace(
            attn=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)],
            token_ids=[5, 6, 7, 8],
            gen_start=10,
        )
    assert any("between 0 and " in message for message in caplog.messages)
    caplog.clear()

    with caplog.at_level("ERROR"):
        Trace(
            attn=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)],
            gen_start=-1,
        )
    assert any("between 0 and " in message for message in caplog.messages)


# ------------------------- Helper Methods -------------------------


def test_get_token_index(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    assert trace.gen_start == 1

    # Direct index
    token_index = trace._get_token_index(1)
    assert token_index == 1

    # Using Selector
    class DummySelector(Selector):
        def __call__(self, trace):
            return 2

    selector = DummySelector()
    token_index = trace._get_token_index(selector)
    assert token_index == 2


def test_get_token_index_out_of_range(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)

    with pytest.raises(IndexError):
        trace._get_token_index(-1)

    with pytest.raises(IndexError):
        trace._get_token_index(10)


def test_get_tkn2img_map(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    token_index = 2
    image_index = 1

    attn_map = trace._get_tkn2img_map(token_index, image_index, "attn")
    grad_map = trace._get_tkn2img_map(token_index, image_index, "grad")

    assert isinstance(attn_map, SaliencyMap)
    assert attn_map.map.shape == (2, 2, 6, 6)  # [layers, heads, H, W]

    assert isinstance(grad_map, SaliencyMap)
    assert grad_map.map.shape == (2, 2, 6, 6)  # [layers, heads, H, W]


def test_get_tkn2img_map_no_data(dummy_processor):
    trace_no_attn = create_trace(dummy_processor, attn=False, grad=True)
    with pytest.raises(ValueError):
        trace_no_attn._get_tkn2img_map(2, 0, "attn")

    trace_no_grad = create_trace(dummy_processor, attn=True, grad=False)
    with pytest.raises(ValueError):
        trace_no_grad._get_tkn2img_map(2, 0, "grad")


# ------------------------- Map --------------------------------


def test_invalid_image_index_map(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    with pytest.raises(IndexError):
        trace.map(2, mode="attn", image_index=10)


def test_map_trace_transform(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)

    def dummy_transform(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
        combined_tensor = attn.tensor() + grad.tensor()
        return SaliencyMap(combined_tensor)

    token_index = 2
    image_index = 0

    result = trace.map(token_index, dummy_transform, image_index=image_index)

    attn_map = trace._get_tkn2img_map(token_index, image_index, "attn")
    grad_map = trace._get_tkn2img_map(token_index, image_index, "grad")
    expected_tensor = attn_map.tensor() + grad_map.tensor()

    assert isinstance(result, SaliencyMap)
    torch.testing.assert_close(result.tensor(), expected_tensor)


def test_map_trace_transform_no_tensor(dummy_processor):
    def dummy_transform(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
        combined_tensor = attn.tensor() + grad.tensor()
        return SaliencyMap(combined_tensor)

    token_index = 2
    trace = create_trace(dummy_processor, attn=True, grad=False)
    with pytest.raises(ValueError):
        trace.map(token_index, dummy_transform, image_index=0)

    trace = create_trace(dummy_processor, attn=False, grad=True)
    with pytest.raises(ValueError):
        trace.map(token_index, dummy_transform, image_index=0)


def test_map_generation(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    token_index = 1
    image_index = 1

    attn_map = trace.map(token_index, image_index=image_index)  # default is attn
    grad_map = trace.map(token_index, image_index=image_index, mode="grad")

    expected_attn_map = trace._get_tkn2img_map(token_index, image_index, "attn")
    expected_grad_map = trace._get_tkn2img_map(token_index, image_index, "grad")

    assert isinstance(attn_map, SaliencyMap)
    assert attn_map == expected_attn_map

    assert isinstance(grad_map, SaliencyMap)
    assert grad_map == expected_grad_map


# ------------------------- Visualization -------------------------


def test_visualize_tokens(dummy_processor, monkeypatch):
    trace = create_trace(dummy_processor, attn=True, grad=True)

    def dummy_render_token_ids(token_ids, processor, gen_start, skip_tokens, only_number_generated):
        assert trace.token_ids == token_ids
        assert processor is trace.processor
        assert gen_start == trace.gen_start
        assert skip_tokens == trace.image_token_id
        assert only_number_generated is True

    import vl_saliency.viz.tokens as tokens_viz

    monkeypatch.setattr(tokens_viz, "render_token_ids", dummy_render_token_ids)

    trace.visualize_tokens()


def test_visualize_tokens_missing_processor():
    trace = create_trace(None, attn=True, grad=True)
    assert trace.processor is None

    with pytest.raises(ValueError):
        trace.visualize_tokens()
