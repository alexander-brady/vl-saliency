import pytest

from vl_saliency.core.trace import Trace


@pytest.fixture
def get_attn_grad_maps():
    def _get_maps(trace: Trace, token: int):
        assert trace.attn is not None
        assert trace.grad is not None
        attn_map = trace.attn[0][:, :, token, :, :]
        grad_map = trace.grad[0][:, :, token, :, :]
        return attn_map, grad_map

    return _get_maps
