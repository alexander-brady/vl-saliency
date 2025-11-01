import pytest
import torch
from torch.nn.functional import interpolate

from vl_saliency.transforms.spatial import Binarize, GaussianSmoothing, Upscale

# ------------------------------ Binarize Tests ------------------------------ #


def test_binarize_mean(smap):
    binarizer = Binarize(threshold="mean")
    binarized_map = smap >> binarizer

    tensor = smap.tensor()
    threshold_value = tensor.mean().item()
    expected_binarized = (tensor >= threshold_value).float()

    assert torch.equal(binarized_map.tensor(), expected_binarized)


def test_binarize_value(smap):
    threshold_value = 0.5
    binarizer = Binarize(threshold=threshold_value)
    binarized_map = smap >> binarizer

    tensor = smap.tensor()
    expected_binarized = (tensor >= threshold_value).float()

    assert torch.equal(binarized_map.tensor(), expected_binarized)


# -------------------------- GaussianSmoothing Tests ------------------------- #


def test_gaussian_smoothing(smap):
    smap = smap.agg()  # Ensure smap is 2D [1, 1, H, W]

    sigma = 2.0
    smoother = GaussianSmoothing(sigma=sigma)
    smoothed_map = smap >> smoother

    from scipy.ndimage import gaussian_filter

    tensor = smap.tensor().squeeze().numpy()
    expected_smoothed = gaussian_filter(tensor, sigma=sigma)
    expected_smoothed_tensor = torch.from_numpy(expected_smoothed).unsqueeze(0).unsqueeze(0)

    assert torch.allclose(smoothed_map.tensor(), expected_smoothed_tensor, atol=1e-6)


def test_gaussian_smoothing_invalid_input(smap):
    tensor = smap.tensor().squeeze()
    assert tensor.dim() != 2  # Ensure smap is not 2D

    with pytest.raises(ValueError):
        GaussianSmoothing(sigma=-1.0)
        smap.apply(GaussianSmoothing(sigma=1.0))


# ------------------------------ Upscale Tests ------------------------------ #


def test_upscale(smap):
    smap = smap.agg()  # Ensure smap is 2D [1, 1, H, W]

    upscaler = Upscale(64, 64)
    upscaled_map = smap >> upscaler

    tensor = smap.tensor()
    expected_upscaled = interpolate(tensor, size=(64, 64), mode="bilinear", align_corners=False)

    assert torch.allclose(upscaled_map.tensor(), expected_upscaled, atol=1e-6)
    assert upscaled_map.tensor().shape == (1, 1, 64, 64)


def test_upscale_invalid_input(smap):
    tensor = smap.tensor()
    assert tensor.dim() != 4 or tensor.size(0) != 1 or tensor.size(1) != 1  # Ensure smap is not [1, 1, H, W]

    with pytest.raises(ValueError):
        Upscale(64, 64)
        smap.apply(Upscale(64, 64))


def test_upscale_height_width(smap):
    smap = smap.agg()  # Ensure smap is 2D [1, 1, H, W]

    target_height = 80
    target_width = 120
    upscaler = Upscale(width=target_width, height=target_height)
    upscaled_map = smap >> upscaler

    tensor = smap.tensor()
    expected_upscaled = interpolate(tensor, size=(target_height, target_width), mode="bilinear", align_corners=False)

    assert torch.allclose(upscaled_map.tensor(), expected_upscaled, atol=1e-6)
    assert upscaled_map.tensor().shape == (1, 1, target_height, target_width)
