import torch

from vl_saliency.methods.cam import agcam, gradcam
from vl_saliency.methods.registry import resolve


def test_gradcam_shape_and_relu_on_grad():
    L, H, P = 3, 4, 5  # num_layers, num_heads, seq_len/patches
    attn = torch.randn(L, H, P)
    grad = torch.randn(L, H, P)  # contains negatives

    out = gradcam(attn, grad)

    assert out.shape == (L, H, P)
    # Expected behavior: out == attn * relu(grad), so negatives in grad contribute 0
    relu_grad = torch.relu(grad)
    assert torch.allclose(out, attn * relu_grad, atol=0, rtol=0)


def test_agcam_shape_relu_on_grad_and_sigmoid_on_attn_extremes():
    L, H, P = 2, 3, 4
    # Make attn contain large magnitudes to test sigmoid squashing (~0 and ~1)
    attn = torch.tensor(
        [[[-20.0, 0.0, 20.0, 5.0]] * H] * L
    )  # broadcasted pattern per layer/head
    grad = torch.tensor([[[-1.0, 0.5, 2.0, -3.0]] * H] * L)

    out = agcam(attn, grad)

    assert out.shape == (L, H, P)

    # Reference computation
    relu_grad = torch.relu(grad)
    sig_attn = torch.sigmoid(attn)
    ref = relu_grad * sig_attn

    assert torch.allclose(out, ref, atol=1e-6)

    # Spot-check behavior:
    # - grad < 0 -> zero regardless of attn
    assert torch.allclose(out[..., 0], torch.zeros_like(out[..., 0]))
    # - large negative attn -> ~0 multiplier
    assert torch.all(
        out[..., 0] == 0
    )  # already zero from grad, but also confirms no negatives leak
    # - mid gradients with ~0.5 sigmoid at 0.0
    assert torch.allclose(out[..., 1], relu_grad[..., 1] * 0.5, atol=1e-6)
    # - large positive attn -> ~1 multiplier
    assert torch.allclose(out[..., 2], relu_grad[..., 2] * 1.0, atol=1e-6)


def test_methods_do_not_modify_inputs_inplace():
    L, H, P = 2, 2, 3
    attn = torch.randn(L, H, P)
    grad = torch.randn(L, H, P)

    attn_clone = attn.clone()
    grad_clone = grad.clone()

    _ = gradcam(attn, grad)
    _ = agcam(attn, grad)

    # Inputs remain unchanged (no in-place ops on user-provided tensors)
    assert torch.allclose(attn, attn_clone)
    assert torch.allclose(grad, grad_clone)


def test_registry_resolve_gradcam_and_agcam():
    g = resolve("gradcam")
    a = resolve("agcam")
    assert callable(g) and callable(a)

    # quick smoke check via registry-call equivalence
    attn = torch.tensor([[[1.0, -2.0, 3.0]]])
    grad = torch.tensor([[[-1.0, 2.0, -3.0]]])

    out_g_direct = gradcam(attn, grad)
    out_g_registry = g(attn, grad)
    assert torch.allclose(out_g_direct, out_g_registry)

    out_a_direct = agcam(attn, grad)
    out_a_registry = a(attn, grad)
    assert torch.allclose(out_a_direct, out_a_registry)
