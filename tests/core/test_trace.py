import pytest
import torch

from vl_saliency.core.map import SaliencyMap
from vl_saliency.core.trace import Trace
from vl_saliency.selectors.base import Selector


def create_trace(proc, attn=True, grad=True) -> Trace:
    return Trace(
        attn=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)] if attn else None,
        grad=[torch.randn(2, 2, 3, 6, 6) for _ in range(6)] if grad else None,
        processor=proc,
        image_token_id=1,
        gen_start=5,
        generated_ids=torch.tensor([[5, 6, 7]]),
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
    assert trace.gen_start == 5
    assert trace.generated_ids is not None
    assert torch.equal(trace.generated_ids, torch.tensor([[5, 6, 7]]))


def test_set_default_mode(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    assert trace._default == "attn"

    trace_no_attn = create_trace(dummy_processor, attn=False, grad=True)
    assert trace_no_attn._default == "grad"

    with pytest.raises(ValueError):
        create_trace(dummy_processor, attn=False, grad=False)


# ------------------------- Helper Methods -------------------------


def test_get_token_index(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    assert trace.gen_start == 5

    # Direct index
    token_index = trace._get_token_index(3)
    assert token_index == 3

    # Using Selector
    class DummySelector(Selector):
        def select(self, trace):
            return 2

    selector = DummySelector()
    token_index = trace._get_token_index(selector)
    assert token_index == 2


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


# ------------------------- API --------------------------------


def test_apply_trace_transform(dummy_processor):
    trace = create_trace(dummy_processor, attn=True, grad=True)

    def dummy_transform(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
        combined_tensor = attn.tensor() + grad.tensor()
        return SaliencyMap(combined_tensor)

    token_index = 2
    image_index = 0

    result = trace.apply(token_index, dummy_transform, image_index=image_index)

    attn_map = trace._get_tkn2img_map(token_index, image_index, "attn")
    grad_map = trace._get_tkn2img_map(token_index, image_index, "grad")
    expected_tensor = attn_map.tensor() + grad_map.tensor()

    assert isinstance(result, SaliencyMap)
    torch.testing.assert_close(result.tensor(), expected_tensor)


def test_apply_trace_transform_no_tensor(dummy_processor):
    def dummy_transform(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
        combined_tensor = attn.tensor() + grad.tensor()
        return SaliencyMap(combined_tensor)

    token_index = 2
    trace = create_trace(dummy_processor, attn=True, grad=False)
    with pytest.raises(ValueError):
        trace.apply(token_index, dummy_transform, image_index=0)

    trace = create_trace(dummy_processor, attn=False, grad=True)
    with pytest.raises(ValueError):
        trace.apply(token_index, dummy_transform, image_index=0)


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


def test_visualize_tokens(dummy_processor, monkeypatch):
    trace = create_trace(dummy_processor, attn=True, grad=True)

    def dummy_render_token_ids(generated_ids, processor, gen_start, skip_tokens, only_number_generated):
        assert trace.generated_ids is not None
        assert torch.equal(generated_ids, trace.generated_ids)
        assert processor is trace.processor
        assert gen_start == trace.gen_start
        assert skip_tokens == trace.image_token_id
        assert only_number_generated is True

    import vl_saliency.viz.tokens as tokens_viz

    monkeypatch.setattr(tokens_viz, "render_token_ids", dummy_render_token_ids)

    trace.visualize_tokens()


@pytest.mark.parametrize("missing_attr", ["processor", "generated_ids", "gen_start"])
def test_visualize_tokens_missing_attributes(dummy_processor, missing_attr):
    trace = create_trace(dummy_processor, attn=True, grad=True)
    setattr(trace, missing_attr, None)

    with pytest.raises(ValueError):
        trace.visualize_tokens()
