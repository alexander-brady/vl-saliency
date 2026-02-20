import torch

from vl_saliency.attn import saliency_attention
from vl_saliency.context import SaliencyContext

# ------ Dummy implementations for testing ------


class DummyModule(torch.nn.Module):
    pass


class DummyAttentionFunctions:
    def __init__(self, interface):
        self._interface = interface

    def get_interface(self, implementation, default):
        return self._interface


def dummy_attention_forward(module, query, key, value, attention_mask, **kwargs):
    return query, torch.ones(1, 2, 3, 3)


class DummyContext(SaliencyContext):
    def __init__(self, attn_implementation="dummy_attention"):
        self.attn_implementation = attn_implementation

    def qk_step(self, q, k):
        self.qk_step_called = True


# ------ Test case for saliency_attention ------


def test_saliency_attention(monkeypatch):
    monkeypatch.setattr(
        "vl_saliency.attn.ALL_ATTENTION_FUNCTIONS",
        DummyAttentionFunctions(dummy_attention_forward),
    )

    ctx = DummyContext()

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    mask = torch.ones(1, 1, 3, 3)

    output, weights = saliency_attention(
        module=DummyModule(),
        query=q,
        key=k,
        value=v,
        attention_mask=mask,
        saliency=ctx,
        extra="kwarg",
    )
    assert torch.equal(output, q)
    assert weights is not None
    assert torch.equal(weights, torch.ones(1, 2, 3, 3))
    assert ctx.qk_step_called
