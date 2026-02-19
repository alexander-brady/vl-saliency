import torch

from vl_saliency.attn import saliency_attention

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


def mock_get_saliency_qk(backend, device):
    return lambda **kwargs: torch.tensor([99.0])


# ------ Test case for saliency_attention ------


def test_saliency_attention(monkeypatch, dummy_context):
    monkeypatch.setattr(
        "vl_saliency.attn.ALL_ATTENTION_FUNCTIONS",
        DummyAttentionFunctions(dummy_attention_forward),
    )
    monkeypatch.setattr(
        "vl_saliency.attn.get_saliency_qk",
        mock_get_saliency_qk,
    )
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
        saliency=dummy_context,
        extra="kwarg",
    )
    assert torch.equal(output, q)
    assert weights is not None
    assert torch.equal(weights, torch.ones(1, 2, 3, 3))
    assert torch.equal(dummy_context.updated_value, torch.tensor([99.0]))
