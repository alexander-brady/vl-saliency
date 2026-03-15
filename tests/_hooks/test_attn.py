import pytest
import torch

import vl_saliency._hooks.attn as m
from vl_saliency._hooks.attn import attention_with_saliency

# ------ Dummy implementations ------


class DummyAttentionFunctions:
    def __init__(self, interface):
        self._interface = interface

    def get_interface(self, implementation, default):
        return self._interface


def dummy_attention_forward(module, query, key, value, attention_mask, **kwargs):
    return query, torch.ones(1, 2, 3, 3)


@pytest.fixture
def dummy_context():
    class DummyContext:
        def accumulate_qk(self, q, k):
            self.qk_step_called = True

    return DummyContext()


# ------ Test case ------


def test_saliency_attention(monkeypatch, build_model, dummy_context):
    monkeypatch.setattr(
        m,
        "ALL_ATTENTION_FUNCTIONS",
        DummyAttentionFunctions(dummy_attention_forward),
    )

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    mask = torch.ones(1, 1, 3, 3)

    ctx = dummy_context

    output, weights = attention_with_saliency(
        module=build_model(),
        query=q,
        key=k,
        value=v,
        attention_mask=mask,
        attn_implementation="dummy_attention",
        saliency=ctx,
        extra="kwarg",
    )
    assert torch.equal(output, q)
    assert weights is not None
    assert torch.equal(weights, torch.ones(1, 2, 3, 3))
    assert ctx.qk_step_called
