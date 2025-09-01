import types

import matplotlib
import pytest
import torch
import torch.nn as nn

matplotlib.use("Agg")  # Use a non-interactive backend for testing

# ---- Dummy test doubles (shared across all tests) ----


class DummyProcessor:
    class _Tok:
        pad_token_id = 0

    tokenizer = _Tok()


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        return hasattr(self, key)


class DummyModel(nn.Module):
    def __init__(self, num_layers=2, num_heads=3):
        super().__init__()
        self.config = DummyConfig()
        self._p = nn.Parameter(torch.zeros(1))
        self.num_layers = num_layers
        self.num_heads = num_heads

    @property
    def device(self):
        return self._p.device

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    ):
        bsz, T = input_ids.shape
        L, H = self.num_layers, self.num_heads
        attentions, total = [], 0.0
        for _ in range(L):
            a = torch.randn(bsz, H, T, T, device=self.device, requires_grad=True)
            attentions.append(a)
            total = total + a.sum()
        loss = (total / (L * H * T * T)).pow(2)
        return types.SimpleNamespace(loss=loss, attentions=tuple(attentions))


# ---- Fixtures ----
@pytest.fixture
def dummy_processor():
    return DummyProcessor()


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def tiny_model():
    # handy when you want deterministic-ish tiny shapes
    return DummyModel(num_layers=2, num_heads=3)


@pytest.fixture
def image_grid_hw():
    # default H,W for image grid in tests
    return (2, 3)  # H=2, W=3
