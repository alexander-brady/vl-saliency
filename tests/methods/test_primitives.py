import torch

from vl_saliency.methods.primitives import attn_raw, grad_raw
from vl_saliency.methods.registry import resolve


def test_attn_raw_identity_and_sigmoid_flag():
    attn = torch.tensor([[[ -10.0, 0.0, 10.0 ]]])
    grad = torch.zeros_like(attn)

    out_no_sig = attn_raw(attn, grad, sigmoid=False)
    assert torch.allclose(out_no_sig, attn)

    out_sig = attn_raw(attn, grad, sigmoid=True)
    ref = torch.sigmoid(attn)
    assert torch.allclose(out_sig, ref)


def test_grad_raw_identity_and_relu_and_abs_flags():
    grad = torch.tensor([[[ -2.0, -1.0, 0.0, 1.0, 2.0 ]]])
    attn = torch.zeros_like(grad)

    # identity
    out_id = grad_raw(attn, grad)
    assert torch.allclose(out_id, grad)

    # relu only
    out_relu = grad_raw(attn, grad, relu=True)
    ref_relu = torch.relu(grad)
    assert torch.allclose(out_relu, ref_relu)

    # abs only
    out_abs = grad_raw(attn, grad, abs=True)
    ref_abs = grad.abs()
    assert torch.allclose(out_abs, ref_abs)

    # relu + abs (relu already non-negative, so same as relu)
    out_ra = grad_raw(attn, grad, relu=True, abs=True)
    assert torch.allclose(out_ra, ref_relu)


def test_methods_do_not_modify_inputs_inplace():
    grad = torch.randn(2, 3, 4)
    attn = torch.randn(2, 3, 4)
    grad_copy = grad.clone()
    attn_copy = attn.clone()

    _ = attn_raw(attn, grad, sigmoid=True)
    _ = grad_raw(attn, grad, relu=True, abs=True)

    assert torch.allclose(grad, grad_copy)
    assert torch.allclose(attn, attn_copy)


def test_registry_resolve_attn_and_grad_raw():
    a = resolve("attn_raw")
    g = resolve("grad_raw")
    assert callable(a) and callable(g)
    attn = torch.ones(1,1,2)
    grad = torch.ones_like(attn)
    assert torch.allclose(a(attn, grad), attn_raw(attn, grad))
    assert torch.allclose(g(attn, grad), grad_raw(attn, grad))