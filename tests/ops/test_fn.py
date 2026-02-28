import torch

from vl_saliency.ops import absolute, normalize, relu, sigmoid, softmax, square


def test_basic_ops(sample):
    scores, mask = sample

    assert torch.allclose(relu(scores, mask), torch.relu(scores))
    assert torch.allclose(absolute(scores, mask), torch.abs(scores))
    assert torch.allclose(square(scores, mask), scores**2)
    assert torch.allclose(sigmoid(scores, mask), torch.sigmoid(scores))


def test_normalize_range(sample):
    scores, mask = sample
    out = normalize(scores, mask)

    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.shape == scores.shape


def test_softmax_mask(sample):
    scores, mask = sample
    out = softmax(scores, mask)

    # Softmax sums to 1 over last two dims
    assert torch.allclose(out.flatten(start_dim=-2).sum(-1), torch.ones(1))

    # Masked values should be zero if masked out
    mask[..., 0, 0] = False
    out = softmax(scores, mask)
    assert torch.all(out[..., 0, 0] == 0.0)
