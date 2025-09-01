import pytest
import torch

from vl_saliency import SaliencyTrace


def test_capture_raises_on_batch_size_not_one(dummy_model, dummy_processor):
    trace = SaliencyTrace(dummy_model, dummy_processor, method=lambda a, g: a * g)
    trace.image_token_id = 99

    generated_ids = torch.ones(2, 5, dtype=torch.long)  # batch=2
    input_ids = torch.ones(2, 5, dtype=torch.long)
    pixel_values = torch.randn(1, 3, 8, 8)

    with pytest.raises(ValueError):
        trace.capture(
            generated_ids=generated_ids,
            input_ids=input_ids,
            pixel_values=pixel_values,
        )


def test_map_raises_without_capture(dummy_model, dummy_processor):
    trace = SaliencyTrace(dummy_model, dummy_processor, method=lambda a, g: a * g)
    trace.image_token_id = 99
    with pytest.raises(ValueError):
        _ = trace.map(token=0)


@pytest.mark.parametrize("H,W", [(2, 3)])
def test_capture_then_map_produces_mask(tiny_model, dummy_processor, H, W):
    trace = SaliencyTrace(tiny_model, dummy_processor, method=lambda a, g: a * g)
    trace.image_token_id = 42

    image_grid_thw = torch.tensor([[1, H * 2, W * 2]], dtype=torch.int32)

    patch_tokens = H * W
    prompt_len = 7
    input_ids = torch.full((1, prompt_len), 1, dtype=torch.long)
    input_ids[0, -patch_tokens:] = trace.image_token_id  # exactly H*W image tokens

    T_gen = 10
    generated_ids = torch.arange(T_gen, dtype=torch.long).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 8, 8)

    trace.capture(
        generated_ids=generated_ids,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    sal = trace.map(token=T_gen - 1)
    assert sal.shape == (1, 1, H, W)
    assert torch.isfinite(sal).all()
