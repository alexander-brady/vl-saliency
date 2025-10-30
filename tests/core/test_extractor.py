import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vl_saliency.core.extractor import SaliencyExtractor
from vl_saliency.core.trace import Trace

saliency = importlib.import_module("vl_saliency.core.extractor")


class DummyModel(nn.Module):
    def __init__(self, n_layers=1, n_heads=1):
        super().__init__()
        self.config = SimpleNamespace()

        self.n_layers = n_layers
        self.n_heads = n_heads

        # Register a real parameter to provide device & .parameters()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
    ):
        B = input_ids.size(0)
        T = input_ids.size(1)
        # create attentions that require grad so .grad is populated on backward
        attentions = [torch.ones(B, self.n_heads, T, T, requires_grad=True) for _ in range(self.n_layers)]

        # a simple scalar loss depending on attentions so grads flow to them
        loss = sum(a.sum() for a in attentions)
        return SimpleNamespace(
            loss=loss,
            attentions=tuple(attentions),  # hf models return tuple
        )


@pytest.fixture
def dummy_model():
    return DummyModel(n_layers=2, n_heads=2)


@pytest.fixture
def monkeypatched_utils(monkeypatch):
    """
    Default: 1 image -> patch shape (1,1) => expected 1 image token.
    image_token_id fixed to 99.
    """
    monkeypatch.setattr(saliency, "_get_image_token_id", lambda cfg: 99)
    monkeypatch.setattr(saliency, "_get_vision_patch_shape", lambda cfg: (1, 1))
    monkeypatch.setattr(
        saliency, "_image_patch_shapes", lambda image_count, patch_shape, image_grid_thw: [(1, 1)] * image_count
    )


def make_minimal_inputs(gen_len=5, prompt_len=2, image_tokens=1, image_count=1, image_token_id=99):
    """
    generated_ids: [1, input_ids + [5] * gen_len] (non-pad tokens).
    input_ids: [1, prompt_len] containing `image_tokens` copies of image_token_id.
    pixel_values: [image_count=1, 3, 2, 2] (content unused).
    """
    # put `image_tokens` image markers at the end of the prompt
    prompt = [7] * (prompt_len - image_tokens) + [image_token_id] * image_tokens
    input_ids = torch.tensor([prompt], dtype=torch.long)
    pixel_values = torch.zeros(image_count, 3, 2, 2)  # content unused

    generated = [prompt + [5] * gen_len]  # non-pad tokens (pad=0)
    generated_ids = torch.tensor(generated, dtype=torch.long)
    return generated_ids, input_ids, pixel_values


# ------------------------- Warning cases -------------------------


def test_warns_if_no_image_token_id(dummy_model, dummy_processor, monkeypatched_utils, monkeypatch, caplog):
    # Return -1 to trigger warning
    monkeypatch.setattr(saliency, "_get_image_token_id", lambda cfg: -1)
    monkeypatch.setattr(saliency.logger, "propagate", True)

    with caplog.at_level("WARNING"):
        SaliencyExtractor(dummy_model, dummy_processor)

    assert any("image token id" in msg.lower() for msg in caplog.messages)


def test_info_if_no_patch_shape(dummy_model, dummy_processor, monkeypatched_utils, monkeypatch, caplog):
    monkeypatch.setattr(saliency, "_get_vision_patch_shape", lambda cfg: None)
    monkeypatch.setattr(saliency.logger, "propagate", True)

    with caplog.at_level("INFO"):
        SaliencyExtractor(dummy_model, dummy_processor)

    assert any("image patch shape" in msg.lower() for msg in caplog.messages)


# ------------------------- Error cases -------------------------


def test_raises_when_no_attn_and_no_grad(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor, store_attns=False, store_grads=False)
    gen, inp, px = make_minimal_inputs()

    with pytest.raises(ValueError, match="At least one of store_attns or store_grads"):
        eng.capture(gen, input_ids=inp, pixel_values=px, store_attns=False, store_grads=False)


def test_raises_on_batch_size_not_one(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor)
    gen = torch.ones(2, 4, dtype=torch.long)  # batch=2
    inp = torch.ones(1, 2, dtype=torch.long)
    px = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError, match="Batch size must be 1"):
        eng.capture(gen, input_ids=inp, pixel_values=px)


def test_raises_on_generated_not_extending_input(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor)
    gen = torch.tensor([[1, 2, 3]], dtype=torch.long)  # does not extend inp
    inp = torch.tensor([[1, 2, 4]], dtype=torch.long)
    px = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError, match="generated_ids must extend input_ids"):
        eng.capture(gen, input_ids=inp, pixel_values=px)


def test_raises_on_no_image_tokens(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor)
    gen, inp, px = make_minimal_inputs(image_count=0)  # no image tokens
    print(gen.shape)
    print(inp.shape)
    print(px.shape)
    with pytest.raises(ValueError, match="image must be provided"):
        eng.capture(gen, input_ids=inp, pixel_values=px)


def test_raises_on_mismatched_image_token_count(dummy_model, dummy_processor, monkeypatched_utils, monkeypatch):
    # Setup to expect 2 image tokens
    monkeypatch.setattr(
        saliency, "_image_patch_shapes", lambda image_count, patch_shape, image_grid_thw: [(0, 0)] * image_count
    )
    eng = SaliencyExtractor(dummy_model, dummy_processor)

    gen, inp, px = make_minimal_inputs()
    with pytest.raises(ValueError, match="does not match expected"):
        eng.capture(gen, input_ids=inp, pixel_values=px)


# ------------------------- Happy paths -------------------------


def test_attn_only_returns_expected_shapes(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor, store_attns=True, store_grads=False)
    gen, inp, px = make_minimal_inputs(gen_len=6, prompt_len=2, image_tokens=1, image_token_id=99)

    trace = eng.capture(gen, input_ids=inp, pixel_values=px, store_attns=True, store_grads=False)
    # Trace object actually is DummyTrace; but ensure it's created
    assert isinstance(trace, Trace)
    assert trace.grad is None
    assert isinstance(trace.attn, list) and len(trace.attn) == 1  # one image
    attn0 = trace.attn[0]
    # Expected: [layers, heads, gen_tokens, H, W]
    layers, heads = dummy_model.n_layers, dummy_model.n_heads
    gen_tokens = gen.shape[1] - inp.shape[1]  # sliced by gen_start

    assert list(attn0.shape) == [layers, heads, gen_tokens, 1, 1]
    assert trace.image_token_id == 99
    assert trace.gen_start == inp.shape[1]
    assert trace.processor == dummy_processor
    assert trace.token_ids == gen[0].tolist()


def test_grad_only_returns_expected_shapes(dummy_model, dummy_processor, monkeypatched_utils):
    eng = SaliencyExtractor(dummy_model, dummy_processor, store_attns=False, store_grads=True)
    gen, inp, px = make_minimal_inputs(gen_len=7, prompt_len=3, image_tokens=1, image_token_id=99)

    trace = eng.capture(gen, input_ids=inp, pixel_values=px, store_attns=False, store_grads=True)
    assert trace.attn is None
    assert isinstance(trace.grad, list) and len(trace.grad) == 1
    grad0 = trace.grad[0]
    layers, heads = dummy_model.n_layers, dummy_model.n_heads
    gen_tokens = gen.shape[1] - inp.shape[1]
    assert list(grad0.shape) == [layers, heads, gen_tokens, 1, 1]
