import torch

from vl_saliency.lib.cam import agcam, gradcam


def test_gradcam(dummy_trace, get_attn_grad_maps):
    result = dummy_trace.map(token=2, mode=gradcam)
    attn, grad = get_attn_grad_maps(dummy_trace, token=2)

    grad = torch.relu(grad)
    expected = grad * attn

    assert torch.allclose(result.tensor(), expected)


def test_agcam(dummy_trace, get_attn_grad_maps):
    result = dummy_trace.map(token=2, mode=agcam)
    attn, grad = get_attn_grad_maps(dummy_trace, token=2)

    attn = torch.sigmoid(attn)
    grad = torch.relu(grad)
    expected = grad * attn

    assert torch.allclose(result.tensor(), expected)
