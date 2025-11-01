from typing import Literal

import torch

from ..core.map import SaliencyMap
from .pipe import Chainable


class Binarize(Chainable):
    """Binarize a map using a specified thresholding method.

    Attributes:
        threshold (float | Literal["mean"], default="mean"): Thresholding method or value. If "mean", uses the mean of the map.
    """

    def __init__(self, threshold: float | Literal["mean"] = "mean"):
        self.threshold = threshold

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        tensor = map.tensor()

        if self.threshold == "mean":
            threshold_value = tensor.mean().item()
        else:
            threshold_value = self.threshold

        binarized = (tensor >= threshold_value).float()  # type: ignore
        return SaliencyMap(binarized)


class GaussianSmoothing(Chainable):
    """Gausian smoothing of a 2D map. Must be applied to maps of shape [1, 1, H, W].

    Arguments:
        sigma (float, default=1.0): Standard deviation for Gaussian kernel.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        tensor = map.tensor().squeeze()

        # Ensure map is [1, 1, H, W]
        if tensor.dim() != 2:
            raise ValueError("Input map must be a 2D tensor.")
        map_np = tensor.numpy()

        from scipy.ndimage import gaussian_filter

        smoothed_np = gaussian_filter(map_np, sigma=self.sigma)
        smoothed_tensor = torch.from_numpy(smoothed_np).to(device=tensor.device, dtype=tensor.dtype)

        smoothed_tensor = smoothed_tensor.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        return SaliencyMap(smoothed_tensor)


class Upscale(Chainable):
    """Upscale a saliency map to a specified size. [1, 1, H, W] -> [1, 1, height, width]

    Attributes:
        height (int): Target height for upscaling.
        width (int): Target width for upscaling.
        mode (str, default="bilinear"): Interpolation mode for upscaling.
    """

    def __init__(self, width: int, height: int, mode: str = "bilinear"):
        self.height = height
        self.width = width
        self.mode = mode

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        tensor = map.tensor()

        # Assert map is [1, 1, H, W]
        if tensor.dim() != 4 or tensor.size(0) != 1 or tensor.size(1) != 1:
            raise ValueError("Input map must be a [1, 1, H, W] tensor.")

        upscaled_tensor = torch.nn.functional.interpolate(
            tensor, size=(self.height, self.width), mode=self.mode, align_corners=False
        )

        return SaliencyMap(upscaled_tensor)
