import pytest
import torch

from vl_saliency.core.map import SaliencyMap
from vl_saliency.core.trace import Trace


class DummyTokenizer:
    pad_token_id = 0


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


@pytest.fixture
def dummy_processor():
    return DummyProcessor()


@pytest.fixture
def smap() -> SaliencyMap:
    t = torch.randn(3, 4, 6, 6)  # [layers, heads, H, W]
    return SaliencyMap(t)


@pytest.fixture
def dummy_trace():
    return Trace(
        attn=[torch.randn(4, 4, 3, 6, 6)],
        grad=[torch.randn(4, 4, 3, 6, 6)],
        gen_start=2,
        generated_ids=torch.tensor([[10, 11, 12, 13, 14]]),
    )
