from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def binarize(map: torch.Tensor, threshold: float | Literal["mean"] = "mean") -> torch.Tensor:
    """Binarize a map using a specified thresholding method.

    Args:
        map (torch.Tensor): Input tensor to binarize.
        threshold (float | Literal["mean"], default="mean"): Thresholding method or value. If "mean", uses the mean of the map.

    Returns:
        torch.Tensor: Binarized tensor.
    """
    if threshold == "mean":
        threshold = map.mean().item()

    binarized_map = (map >= threshold).float()
    return binarized_map


def gaussian_smoothing(map: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian smoothing to a 2D map.

    Args:
        map (torch.Tensor): Input 2D tensor to smooth.
        sigma (float, default=1.0): Standard deviation for Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed tensor.
    """

    map_np = map.detach().cpu().numpy()
    smoothed_np = gaussian_filter(map_np, sigma=sigma)
    smoothed_map = torch.from_numpy(smoothed_np).to(device=map.device, dtype=map.dtype)
    return smoothed_map


def upscale(map: torch.Tensor, x_size: int, y_size: int, mode: str = "bilinear") -> torch.Tensor:
    """Upscale a saliency map to a specified size.

    Args:
        map (torch.Tensor): Input saliency map tensor of shape [1, 1, H, W].
        size (tuple[int, int]): Target size (height, width) for upscaling.
        mode (str, default="bilinear"): Interpolation mode for upscaling.

    Returns:
        torch.Tensor: Upscaled saliency map tensor of shape [1, 1, target_height, target_width].
    """
    upscaled_map = torch.nn.functional.interpolate(map, size=(y_size, x_size), mode=mode, align_corners=False)
    return upscaled_map


def normalize(map: torch.Tensor) -> torch.Tensor:
    """Normalize a saliency map to the range [0, 1].

    Args:
        map (torch.Tensor): Input saliency map tensor.

    Returns:
        torch.Tensor: Normalized saliency map tensor.
    """
    np_map = map.detach().cpu().numpy()
    normalized_map = (np_map - np_map.min()) / (np.ptp(np_map) + 1e-8)

    return torch.from_numpy(normalized_map).to(device=map.device, dtype=map.dtype)
