import pytest
import torch

from vl_saliency.core.map import SaliencyMap


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
    t = torch.arange(0, 16.0).view(1, 1, 4, 4)
    return SaliencyMap(t)
