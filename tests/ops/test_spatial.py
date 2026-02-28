import pytest
import torch

from vl_saliency.ops.spatial import Binarize, GaussianSmoothing, SoftBinarize, Upscale

# ------- Binarize -------


def test_binarize_fixed_threshold(sample):
    x, mask = sample
    op = Binarize(threshold=1.5)
    out = op(x, mask)

    expected = (x > 1.5).float()
    assert torch.allclose(out, expected)


def test_binarize_mean_threshold_no_mask(sample):
    x, _ = sample
    op = Binarize(threshold="mean")
    out = op(x)

    threshold = x.mean()
    expected = (x > threshold).float()
    assert torch.allclose(out, expected)


def test_binarize_mean_threshold(sample):
    x, mask = sample
    op = Binarize(threshold="mean")
    out = op(x, mask)

    threshold = x[mask].mean()
    expected = (x > threshold).float()
    assert torch.allclose(out, expected)


def test_binarize_empty_mask(sample):
    x, mask = sample
    mask = torch.zeros_like(mask, dtype=torch.bool)  # Empty mask
    op = Binarize(threshold="mean")
    out = op(x, mask)

    expected = (x > 0.0).float()  # With empty mask, threshold defaults to 0.0
    assert torch.allclose(out, expected)


# ------- SoftBinarize -------


def test_soft_binarize_shape_and_range(sample):
    x, mask = sample
    op = SoftBinarize(threshold=1.0, softness=10.0)
    out = op(x, mask)

    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_soft_binarize_mean_threshold(sample):
    x, mask = sample
    op = SoftBinarize(threshold="mean", softness=10.0)
    out = op(x, mask)

    threshold = x[mask].mean()
    expected = torch.sigmoid((x - threshold) * 10.0)
    assert torch.allclose(out, expected)


def test_soft_binarize_empty_mask(sample):
    x, mask = sample
    mask = torch.zeros_like(mask, dtype=torch.bool)  # Empty mask
    op = SoftBinarize(threshold="mean", softness=10.0)
    out = op(x, mask)

    expected = torch.sigmoid((x - 0.0) * 10.0)  # Threshold defaults to 0.0
    assert torch.allclose(out, expected)


def test_soft_binarize_no_mask(sample):
    x, _ = sample
    op = SoftBinarize(threshold="mean", softness=10.0)
    out = op(x)

    threshold = x.mean()
    expected = torch.sigmoid((x - threshold) * 10.0)
    assert torch.allclose(out, expected)


def test_soft_binarize_differentiable(sample):
    x, mask = sample
    x.requires_grad_()
    op = SoftBinarize(threshold=1.0, softness=10.0)
    out = op(x, mask)

    out.sum().backward()  # Should not raise an error
    assert x.grad is not None


# ------- GaussianSmoothing -------


def test_gaussian_smoothing_shape(sample):
    x, mask = sample
    op = GaussianSmoothing(kernel_size=3, sigma=1.0)
    out = op(x, mask)

    assert out.shape == x.shape


def test_gaussian_requires_odd_kernel():
    with pytest.raises(ValueError):
        GaussianSmoothing(kernel_size=4, sigma=1.0)


# ------- Upscale -------


def test_upscale_changes_size(sample):
    x, mask = sample
    op = Upscale(height=4, width=4)
    out = op(x, mask)

    assert out.shape[-2:] == (4, 4)
