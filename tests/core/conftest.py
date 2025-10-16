from dataclasses import dataclass

import pytest
import torch

from vl_saliency import SaliencyTrace
from vl_saliency.methods.registry import register


# One-time registration for the tests
@register("dummy")
def _dummy(a, g):
    return a * g


@pytest.fixture(scope="session")
def image_token_id() -> int:
    return 42


@dataclass
class IO:
    generated_ids: torch.Tensor
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


def _mk_io(*, H: int, W: int, prompt_len: int, T_gen: int, image_token_id: int) -> IO:
    # exactly H*W image tokens at the end
    input_ids = torch.full((1, prompt_len), 1, dtype=torch.long)
    patch_tokens = H * W
    input_ids[0, -patch_tokens:] = image_token_id
    return IO(
        generated_ids=torch.arange(T_gen, dtype=torch.long).unsqueeze(0),
        input_ids=input_ids,
        pixel_values=torch.randn(1, 3, 8, 8),
        image_grid_thw=torch.tensor([[1, H * 2, W * 2]], dtype=torch.int32),
    )


@pytest.fixture
def io_factory(image_token_id: int):
    def _factory(H=2, W=3, prompt_len=7, T_gen=10):
        return _mk_io(H=H, W=W, prompt_len=prompt_len, T_gen=T_gen, image_token_id=image_token_id)

    return _factory


@pytest.fixture
def make_trace(dummy_processor, image_token_id: int):
    """Factory so tests can create a configured SaliencyTrace quickly."""

    def _factory(model, method="dummy"):
        tr = SaliencyTrace(model, dummy_processor, method=method)
        tr.image_token_id = image_token_id
        return tr

    return _factory
