import pytest
import torch

from vl_saliency.core.trace import Trace


@pytest.fixture
def get_attn_grad_maps():
    def _get_maps(trace, token):
        attn_map = trace.attn[0][:, :, token, :, :]
        grad_map = trace.grad[0][:, :, token, :, :]
        return attn_map, grad_map

    return _get_maps


@pytest.fixture
def dummy_trace():
    attn = [torch.randn(4, 4, 5, 6, 6)]
    grad = [torch.randn(4, 4, 5, 6, 6)]

    return Trace(attn=attn, grad=grad)
