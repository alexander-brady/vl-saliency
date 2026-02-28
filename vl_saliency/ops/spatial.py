from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float

from vl_saliency.ops.fuse import FusableMixin


class Binarize:
    """Binarizes saliency scores based on a threshold. Can use mean or a fixed value as threshold. Non-differentiable."""

    def __init__(self, threshold: float | Literal["mean"] = "mean"):
        self.threshold: float | Literal["mean"] = threshold

    def __call__(
        self, scores: Float[torch.Tensor, "..."], mask: Bool[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "..."]:
        if self.threshold == "mean":
            if mask is None:
                threshold = scores.mean().detach()
            elif mask.any():
                threshold = scores[mask].mean().detach()
            else:
                threshold = torch.zeros((), device=scores.device, dtype=scores.dtype)
        else:
            threshold = self.threshold

        return (scores > threshold).float()


class SoftBinarize(FusableMixin):
    """Differentiable version of Binarize using a sigmoid function to create a soft mask. The softness parameter controls how close the output is to a hard binary mask."""

    def __init__(self, threshold: float | Literal["mean"] = "mean", softness: float = 1.0):
        self.threshold: float | Literal["mean"] = threshold
        self.softness = softness

    def __call__(
        self, scores: Float[torch.Tensor, "..."], mask: Bool[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "..."]:
        if self.threshold == "mean":
            if mask is None:
                threshold = scores.mean().detach()
            elif mask.any():
                threshold = scores[mask].mean().detach()
            else:
                threshold = torch.zeros((), device=scores.device, dtype=scores.dtype)
        else:
            threshold = self.threshold

        return torch.sigmoid((scores - threshold) * self.softness)


class GaussianSmoothing(FusableMixin):
    """Applies Gaussian smoothing to the saliency map. The kernel size and sigma control the amount of smoothing."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

    def _make_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Creates a 2D Gaussian kernel."""
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / kernel.sum()

    def __call__(
        self, scores: Float[torch.Tensor, "..."], mask: Bool[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "..."]:
        """Applies Gaussian smoothing to the input saliency map. Requires Torchvision."""

        kernel = self._make_gaussian_kernel(  # Build gaussian kernel
            self.kernel_size, self.sigma
        ).to(scores.device, scores.dtype)[None, None, :, :]

        squeeze = scores.ndim < 4  # Check if input is already in (N, C, H, W) format
        if squeeze:
            scores = scores.unsqueeze(1)  # Add channel dimension for layer operations

        C = scores.shape[1]
        weight = kernel.expand(C, 1, -1, -1)  # Expand kernel for all channels
        smoothed = F.conv2d(scores, weight, padding=self.padding, groups=C)

        if squeeze:
            smoothed = smoothed.squeeze(1)  # Remove channel dimension if it was added

        return smoothed


class Upscale(FusableMixin):
    """Upscales the saliency map to a target size using bilinear interpolation."""

    def __init__(self, height: int, width: int, mode: str = "bilinear"):
        self.target_size = (height, width)
        self.mode = mode

    def __call__(
        self, scores: Float[torch.Tensor, "..."], mask: Bool[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "..."]:
        squeeze = scores.ndim < 4  # Check if input is already in (N, C, H, W) format
        if squeeze:
            scores = scores.unsqueeze(1)  # Add channel dimension for layer operations

        upscaled = F.interpolate(scores, size=self.target_size, mode=self.mode, align_corners=False)

        if squeeze:
            upscaled = upscaled.squeeze(1)  # Remove channel dimension if it was added

        return upscaled
